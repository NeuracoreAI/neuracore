"""Gemma MoE model with custom attention."""

# cspell:ignore openpi adarms layernorm

import math
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Literal, Optional, Tuple, cast

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import (
    Cache,
    GemmaAttention,
    GemmaConfig,
    GemmaDecoderLayer,
    GemmaForCausalLM,
    GemmaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
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
    """1D sinusoidal positional embedding module based on OpenPI's implementation.

    Key features:
    - Uses explicit period boundaries (min_period, max_period).
    - Includes 2pi scaling (inputs are treated as normalized physical time).
    - Uses float64 for internal frequency calculations to avoid numerical divergence.
    """

    def __init__(self, dim: int):
        """Initialize the SinusoidalPosEmb module.

        Args:
            dim: Dimension of the positional embedding.
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dimension ({dim}) must be divisible by 2")
        self.dim = dim

    def forward(
        self, time: torch.Tensor, min_period: float = 1.0, max_period: float = 10000.0
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            time: Input tensor of shape (batch_size, ).
            min_period: The minimum period of the sine waves.
            max_period: The maximum period of the sine waves.

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, dim).
        """
        # 1. Validation (Strict shape check per OpenPI spec)
        if time.ndim != 1:
            raise ValueError(
                "The time tensor is expected to be of shape `(batch_size, )`."
            )

        device = time.device
        half_dim = self.dim // 2

        # We perform the linspace and power calculation in float64 to ensure
        # the geometric progression is numerically stable.
        fraction = torch.linspace(
            0.0, 1.0, half_dim, dtype=torch.float64, device=device
        )

        # Calculate periods: Geometric progression from min_period to max_period
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = (1.0 / period) * 2 * math.pi

        # time: (B, ) -> (B, 1)
        # scaling_factor: (D/2, ) -> (1, D/2)
        # Result: (B, D/2)
        sin_input = time[:, None] * scaling_factor[None, :]

        # 5. Trigonometric Operations & Concatenation
        emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)

        # Cast back to the input dtype (e.g., float32 or bfloat16) before returning
        return emb.to(dtype=time.dtype)


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


class GemmaVariantConfig:
    """Configuration for Gemma model variants."""

    def __init__(
        self,
        width: int,
        depth: int,
        mlp_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Store Gemma variant hyperparameters."""
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(
    variant: str,
) -> GemmaVariantConfig:  # see openpi `gemma.py: get_config`
    """Return config for specified Gemma variant."""
    if variant == "gemma_300m":
        return GemmaVariantConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaVariantConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for PI0.

    Adapted from OpenPI's PaliGemmaWithExpertModel.
    """

    def __init__(
        self,
        vlm_config: GemmaVariantConfig,
        action_expert_config: GemmaVariantConfig,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> None:
        """Initialize the VLM and action expert models."""
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = (
            vlm_config.width if use_adarms[0] else None
        )
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ) -> None:
        """Cast parameters to requested precision, preserving key layers in float32."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return image embeddings from the vision tower."""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return token embeddings from the language model."""
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor | None] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor | None], Cache | None]:
        """Forward pass through VLM and action expert."""
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided for forward pass.")
        adarms_cond = cast(list[torch.Tensor | None], adarms_cond)
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            if inputs_embeds[0] is None:
                raise ValueError(
                    "inputs_embeds[0] must be provided when mixing experts."
                )
            first_embed = inputs_embeds[0]
            second_embed = inputs_embeds[1]
            # Both embeddings are guaranteed to be present in this branch
            non_optional_embeds: list[torch.Tensor] = [first_embed, second_embed]
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            current_embeds = non_optional_embeds

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    current_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        current_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    current_embeds = compute_layer_complete(
                        layer_idx,
                        current_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(
                inputs_embeds: list[torch.Tensor],
                adarms_cond: list[torch.Tensor | None],
            ) -> list[torch.Tensor]:
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    current_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(current_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None,
    adarms_cond: list[torch.Tensor | None],
    paligemma: PaliGemmaForConditionalGeneration,
    gemma_expert: GemmaForCausalLM,
) -> list[torch.Tensor]:
    """Compute a full transformer layer with shared attention across experts."""
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(
            hidden_states, cond=adarms_cond[i]
        )  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = (
            layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        key_state = (
            layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        value_state = (
            layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(
            hidden_states, out_emb, gates[i]
        )  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(
            after_first_residual, out_emb, gate
        )  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds
