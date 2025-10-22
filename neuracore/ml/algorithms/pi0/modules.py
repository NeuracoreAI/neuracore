"""Gemma MoE model with custom attention."""

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma.modeling_gemma import (
    Cache,
    GemmaAttention,
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class MoeExpertConfig:
    """Configuration for the MoE model."""

    hidden_size: int  # aka width
    intermediate_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    use_cache: bool = False
    hidden_activation: str = "gelu_pytorch_tanh"


class CustomGemmaAttention(GemmaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper.

    Note this is a replica of the GemmaAttention module from the Hugging Face.
    We have to replicate it here to be able to modify the forward pass,
    and expose the query, key, and value states for the mixed attention.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for the CustomGemmaAttention module.

        Args:
            hidden_states: Input hidden states.
            position_embeddings: Position embeddings.
            attention_mask: Attention mask.
            past_key_value: Past key-value cache.
            cache_position: Cache position.
            **kwargs: Additional keyword arguments.

        Returns:
            Output hidden states, attention weights, and past key-value cache.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        self.query_states = query_states = (
            self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        self.key_states = key_states = (
            self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        self.value_states = value_states = (
            self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models;
            # cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` "
                    "does not support `output_attentions=True`. Falling back to "
                    "eager attention. This warning can be removed using the argument "
                    '`attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def o_project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Output projection for the attention module.

        Args:
            hidden_states: Input hidden states.

        Returns:
            torch.Tensor: Output hidden states.
        """
        return self.o_proj(hidden_states)


class GemmaMoELayer(nn.Module):
    """A layer that combines individual Gemma experts with cross-expert attention."""

    def __init__(self, expert_configs: dict[str, MoeExpertConfig], layer_idx: int):
        """Initialize the GemmaMoELayer.

        Args:
            expert_configs: Configuration for the experts.
            layer_idx: Index of the layer.
        """
        super().__init__()
        self.expert_configs = expert_configs
        self.layer_idx = layer_idx

        self.experts = nn.ModuleDict()
        self.rotary_embs = nn.ModuleDict()
        for name, config in expert_configs.items():
            # Create Gemma config for this expert
            gemma_config = GemmaConfig(**asdict(config))
            # Ensure attention implementation is set to eager to avoid None lookups
            # in ALL_ATTENTION_FUNCTIONS during CustomGemmaAttention.forward
            setattr(gemma_config, "_attn_implementation", "eager")
            setattr(gemma_config, "attn_implementation", "eager")
            self.experts[name] = GemmaDecoderLayer(gemma_config, layer_idx)
            self.experts[name].self_attn = CustomGemmaAttention(
                config=gemma_config, layer_idx=layer_idx
            )
            self.rotary_embs[name] = GemmaRotaryEmbedding(config=gemma_config)

    def mix_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        """Compute mixed attention across experts.

        Args:
            queries: Query tensor.
            keys: Key tensor.
            values: Value tensor.
            attention_mask: Attention mask.
            dropout_p: Dropout probability.

        Returns:
            torch.Tensor: Mixed attention output.
        """
        # Compute attention scores
        attn_weights = torch.matmul(queries, keys.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(queries.size(-1))

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout_p, training=self.training
        )

        # Compute mixed attention output
        mixed_output = torch.matmul(attn_weights, values)
        return mixed_output

    def forward(
        self,
        hidden_states: Dict[str, torch.FloatTensor],
        expert_attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        mix_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Dict[str, torch.LongTensor]] = None,
        past_key_values: Optional[Dict[str, DynamicCache]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.FloatTensor]:
        """Forward pass for the GemmaMoELayer.

        Args:
            hidden_states: Input hidden states.
            expert_attention_masks: Attention masks for the experts.
            mix_attention_mask: Mixed attention mask.
            position_ids: Position IDs.
            past_key_values: Past key-value caches.
            use_cache: Whether to use caching.

        Returns:
            Dict[str, torch.FloatTensor]: Output hidden states.
        """
        expert_outputs = {}  # Store the expert outputs
        query_states_all, key_states_all, value_states_all = {}, {}, {}
        for name, states in hidden_states.items():
            pos_ids = position_ids.get(name) if position_ids else None
            past_kv = past_key_values.get(name) if past_key_values else None

            # Get pos embeddings and run through expert
            position_embeddings = self.rotary_embs[name](states, pos_ids)
            expert_output = self.experts[name](
                hidden_states=states,
                attention_mask=(
                    expert_attention_masks[name] if expert_attention_masks else None
                ),
                position_ids=pos_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            expert_outputs[name] = expert_output[0]  # Store the output

            # Store attention states
            query_states_all[name] = self.experts[name].self_attn.query_states
            key_states_all[name] = self.experts[name].self_attn.key_states
            value_states_all[name] = self.experts[name].self_attn.value_states

        # Concatenate for mixed attention
        queries = torch.cat(tuple(query_states_all.values()), dim=2)
        keys = torch.cat(tuple(key_states_all.values()), dim=2)
        values = torch.cat(tuple(value_states_all.values()), dim=2)

        # Run mixed attention
        mixed_output = self.mix_attention(queries, keys, values, mix_attention_mask)

        attn_output = mixed_output.transpose(1, 2).contiguous()
        batch_size = queries.size(0)
        q_lens = [hidden_states.size(1) for hidden_states in hidden_states.values()]
        attn_output = attn_output.view(batch_size, sum(q_lens), -1)

        # Split back per expert
        attn_outputs = torch.split(attn_output, q_lens, dim=1)

        # Combine with expert outputs
        outputs = {}
        for name, states in zip(hidden_states.keys(), attn_outputs):
            proj_mixed = self.experts[name].self_attn.o_project(states)
            # Add expert output as residual
            outputs[name] = expert_outputs[name] + proj_mixed

        return outputs


class GemmaMoE(nn.Module):
    """Main MoE model that uses Gemma experts."""

    def __init__(
        self,
        depth: int,
        expert_configs: dict[str, MoeExpertConfig],
    ):
        """Initialize the GemmaMoE model.

        Args:
            depth: Depth of the MoE model.
            expert_configs: Configuration for the experts.
        """
        super().__init__()
        self.expert_names = list(expert_configs.keys())
        self.expert_configs = expert_configs

        # Create layers with Gemma experts
        self.layers = nn.ModuleList(
            [GemmaMoELayer(expert_configs, i) for i in range(depth)]
        )

        # Create final layer norms for each expert
        self.final_norms = nn.ModuleDict()
        for name, config in expert_configs.items():
            self.final_norms[name] = nn.LayerNorm(config.hidden_size)

        # Track which experts use caching
        self.cache_names = [
            name for name, config in expert_configs.items() if config.use_cache
        ]

    def _init_caches(self) -> Dict[str, DynamicCache]:
        """Initialize caches for the experts.

        Returns:
            Dict[str, DynamicCache]: Initialized caches.
        """
        return {name: DynamicCache() for name in self.cache_names}

    def _normalize_inputs(
        self, hidden_states: Dict[str, torch.FloatTensor]
    ) -> Dict[str, torch.FloatTensor]:
        """Normalize input hidden states.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Dict[str, torch.FloatTensor]: Normalized hidden states.
        """
        normalized = {}
        for name, states in hidden_states.items():
            hidden_size = states.shape[-1]
            normalizer = torch.sqrt(
                torch.tensor(hidden_size, dtype=states.dtype, device=states.device)
            )
            normalized[name] = states * normalizer
        return normalized

    def get_parameters(self, mixture_name: str) -> list:
        """Get the parameters for a specific mixture.

        Args:
            mixture_name: Name of the mixture.

        Returns:
            list: List of parameters.
        """
        params = []
        for layer in self.layers:
            for name, expert in layer.experts.items():
                if name == mixture_name:
                    params.extend([p for p in expert.parameters()])
        return params

    def forward(
        self,
        hidden_states: Dict[str, torch.FloatTensor],
        expert_attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        mix_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[Dict[str, torch.LongTensor]] = None,
        past_key_values: Optional[Dict[str, DynamicCache]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the GemmaMoE model.

        Args:
            hidden_states: Input hidden states.
            expert_attention_masks: Attention masks for the experts.
            mix_attention_mask: Mixed attention mask.
            position_ids: Position IDs.
            past_key_values: Past key-value caches.
            use_cache: Whether to use caching.

        Returns:
            hidden_states: Output hidden states.
        """
        # Initialize caches if needed
        if past_key_values is None and use_cache:
            past_key_values = self._init_caches()

        # Normalize inputs
        hidden_states = self._normalize_inputs(hidden_states)

        # Process through layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                expert_attention_masks=expert_attention_masks,
                mix_attention_mask=mix_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        # Apply final layer norms
        hidden_states = {
            name: self.final_norms[name](states)
            for name, states in hidden_states.items()
        }
        return hidden_states


class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embedding module used for time embedding.

    This module implements sinusoidal positional embeddings for time steps,
    commonly used in diffusion models and transformers.
    """

    def __init__(self, dim: int):
        """Initialize the SinusoidalPosEmb module.

        Args:
            dim: Dimension of the positional embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor, max_period: float = 10000.0) -> torch.Tensor:
        """Forward pass for the SinusoidalPosEmb module.

        Args:
            t: Input tensor.
            max_period: Maximum period for the sinusoidal embedding.

        Returns:
            torch.Tensor: Positional embeddings.
        """
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb).to(t.dtype)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Action encoder for the Pi0 model."""

    def __init__(self, action_dim: int, width: int):
        """Initialize the ActionEncoder module.

        Args:
            action_dim: Dimension of the action space.
            width: Width of the encoder.
        """
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width)
        self.linear_2 = nn.Linear(2 * width, width)
        self.nonlinearity = nn.SiLU()
        self.linear_3 = nn.Linear(width, width)

    def forward(
        self, action: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the ActionEncoder module.

        Args:
            action: Input action tensor.
            time_emb: Time embedding tensor.

        Returns:
            torch.Tensor: Encoded action tensor.
        """
        emb = self.linear_1(action)  # [B, H, W]
        if time_emb is not None:
            time_emb_full = time_emb.unsqueeze(1).expand(-1, action.size(1), -1)
        else:
            time_emb_full = torch.zeros_like(emb)
        emb = torch.cat([time_emb_full, emb], dim=-1)  # [B, H, W * 2]
        emb = self.nonlinearity(self.linear_2(emb))  # [B, H, W]
        emb = self.linear_3(emb)  # [B, H, W]
        return emb  # [B, H, W]
