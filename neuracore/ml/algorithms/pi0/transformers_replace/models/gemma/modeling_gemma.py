"""Gemma model implementation for the PI0 transformers library modification.

This module implements the Gemma transformer architecture adapted for the
Neuracore PI0 algorithm. The model supports adaptive RMS normalization (ADARMS)
for conditional generation, rotary position embeddings (RoPE), and various
attention backends including flash attention.

The implementation includes:
- GemmaRMSNorm: Root mean square layer normalization with optional adaptive
  conditioning for ADARMS
- GemmaMLP: Multi-layer perceptron with gated activation
- GemmaRotaryEmbedding: Rotary position embeddings for relative position encoding
- GemmaAttention: Multi-headed self-attention with support for grouped query
  attention
- GemmaDecoderLayer: Transformer decoder layer with self-attention and MLP
- GemmaModel: Full decoder-only transformer model
- GemmaForCausalLM: Causal language modeling head on top of GemmaModel

This file started life as auto-generated code from the upstream transformers
Gemma model and is now maintained here and adapted for the Neuracore PI0
implementation.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import LossKwargs, auto_docstring, can_return_tuple, logging
from .configuration_gemma import GemmaConfig

logger = logging.get_logger(__name__)


class GemmaRMSNorm(nn.Module):
    """Root mean square layer normalization with optional adaptive conditioning.

    This module implements RMS normalization as used in the Gemma architecture.
    When a condition dimension is provided, it supports adaptive RMS normalization
    (ADARMS) which modulates the normalization using a learned dense layer that
    produces scale, shift, and gate parameters from a conditioning vector.

    Args:
        dim: Hidden dimension size
        eps: Small epsilon value for numerical stability (default: 1e-6)
        cond_dim: Optional condition dimension for adaptive normalization
    """

    def __init__(
        self, dim: int, eps: float = 1e-6, cond_dim: int | None = None
    ) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            # self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization of input tensor.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor with same shape as input.
        """
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply RMS normalization with optional adaptive conditioning.

        Args:
            x: Input tensor of shape [batch, seq_len, dim] or [batch, dim]
            cond: Optional condition tensor of shape [batch, cond_dim] for
                adaptive normalization. If provided, must match cond_dim.

        Returns:
            Tuple of (normalized_tensor, gate_tensor). The gate tensor is None
            for standard RMSNorm and contains the gate values for ADARMS.
        """
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return (
                normed_inputs.to(dtype),
                None,
            )  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(
                f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}"
            )

        # self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly:
        # [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)
        # Convert normed_inputs to model dtype

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(
            torch.float32
        )

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        """Return a string representation of the module configuration.

        Returns:
            String describing the module parameters.
        """
        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
        return repr_str


class GemmaMLP(nn.Module):
    """Multi-layer perceptron with gated activation for Gemma.

    This MLP uses a gated activation pattern where the input is projected
    through both a gate projection and an up projection, then combined via
    element-wise multiplication before being projected down.

    Args:
        config: Gemma configuration containing hidden_size, intermediate_size,
            and hidden_act activation function.
    """

    def __init__(self, config: GemmaConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP transformation with gated activation.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]
        """
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class GemmaRotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) for relative position encoding.

    This module implements rotary position embeddings that encode relative
    position information directly into the attention mechanism. Supports
    various RoPE scaling types for handling longer sequences.

    Args:
        config: Gemma configuration containing max_position_embeddings and
            optional rope_scaling configuration
        device: Target device for buffer registration
    """

    def __init__(
        self, config: GemmaConfig, device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    # Power user: used with advanced RoPE types (e.g. dynamic rope).
    @dynamic_rope_update
    def forward(
        self, x: torch.Tensor, position_ids: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine embeddings for rotary position encoding.

        Args:
            x: Input tensor used to determine device and dtype
            position_ids: Position indices of shape [batch_size, seq_len]

        Returns:
            Tuple of (cos_emb, sin_emb) tensors of shape
            [batch_size, seq_len, head_dim] matching the dtype of x.
        """
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input for RoPE.

    This function splits the last dimension in half and rotates the two halves,
    which is used in the rotary position embedding computation.

    Args:
        x: Input tensor with shape [..., hidden_dim]

    Returns:
        Rotated tensor with same shape as input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be broadcast to q and k. For
            example, cos[position_ids] and sin[position_ids] have shape
            [batch_size, seq_len, head_dim]. If q and k have shape
            [batch_size, heads, seq_len, head_dim], then setting
            unsqueeze_dim=1 makes cos[position_ids] and sin[position_ids]
            broadcastable to q and k. If q and k have shape
            [batch_size, seq_len, heads, head_dim], set unsqueeze_dim=2.

    Returns:
        `tuple(torch.Tensor)` comprising the rotated query and key tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads across attention groups for grouped query attention.

    This function implements grouped query attention (GQA) by repeating key/value
    heads to match the number of query heads. This is more efficient than
    multi-head attention when num_key_value_heads < num_attention_heads.

    Args:
        hidden_states: Key or value tensor of shape
            [batch, num_key_value_heads, seq_len, head_dim]
        n_rep: Number of repetitions (num_attention_heads // num_key_value_heads)

    Returns:
        Repeated tensor of shape [batch, num_attention_heads, seq_len, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    """Apply a gated residual connection.

    Args:
        x: Input tensor (residual).
        y: Output tensor to be added.
        gate: Optional gate tensor to modulate the addition.

    Returns:
        The gated residual sum.
    """
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation of scaled dot-product attention.

    This function computes attention using standard matrix operations without
    optimized kernels. Used as a fallback when flash attention is not available.

    Args:
        module: Attention module containing num_key_value_groups attribute
        query: Query tensor of shape [batch, num_heads, seq_len, head_dim]
        key: Key tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        value: Value tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        attention_mask: Optional attention mask of shape
            [batch, 1, seq_len, seq_len]
        scaling: Attention scaling factor (typically 1/sqrt(head_dim))
        dropout: Dropout probability for attention weights
        **kwargs: Additional keyword arguments (unused)

    Returns:
        Tuple of (attention_output, attention_weights) where:
        - attention_output: [batch, seq_len, num_heads, head_dim]
        - attention_weights: [batch, num_heads, seq_len, seq_len]
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class GemmaAttention(nn.Module):
    """Multi-headed self-attention with rotary position embeddings.

    This module implements scaled dot-product attention with support for:
    - Grouped query attention (GQA) for efficient key/value caching
    - Rotary position embeddings (RoPE) for relative position encoding
    - Multiple attention backends (eager, flash attention, SDPA)
    - Causal masking for autoregressive generation

    Args:
        config: Gemma configuration containing attention parameters
        layer_idx: Layer index for cache management
    """

    def __init__(self, config: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute self-attention with rotary position embeddings.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) tensors for RoPE
            attention_mask: Optional attention mask of shape
                [batch, 1, seq_len, seq_len]
            past_key_value: Optional cache for key/value states
            cache_position: Optional position indices for cache updates
            use_cache: Whether to update and return the cache
            **kwargs: Additional attention backend arguments

        Returns:
            Tuple of (attention_output, attention_weights) where:
            - attention_output: [batch, seq_len, hidden_size]
            - attention_weights: [batch, num_heads, seq_len, seq_len] or None
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Use cache if provided
        if past_key_value is not None:
            if use_cache:
                # sin and cos are specific to RoPE models; cache_position is
                # needed for the static cache.
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat(
                    [past_key_value[self.layer_idx][0], key_states], dim=2
                )
                value_states = torch.cat(
                    [past_key_value[self.layer_idx][1], value_states], dim=2
                )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
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


class GemmaDecoderLayer(GradientCheckpointingLayer):
    """Transformer decoder layer with self-attention and MLP.

    This layer implements a standard transformer decoder block with:
    - Pre-attention layer normalization (with optional ADARMS)
    - Multi-headed self-attention with RoPE
    - Gated residual connection
    - Post-attention layer normalization (with optional ADARMS)
    - Feed-forward MLP with gated activation
    - Gated residual connection

    Supports gradient checkpointing for memory-efficient training.

    Args:
        config: Gemma configuration containing layer parameters
        layer_idx: Layer index for attention cache management
    """

    def __init__(self, config: GemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        cond_dim = (
            getattr(config, "adarms_cond_dim", None)
            if getattr(config, "use_adarms", False)
            else None
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: None | (
            tuple[torch.Tensor, torch.Tensor]
        ) = None,  # necessary, but kept here for BC
        adarms_cond: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor] | tuple[torch.FloatTensor, torch.FloatTensor | None]:
        """Apply the decoder layer transformation.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional cache for key/value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to update and return the cache
            cache_position: Optional position indices for cache updates
            position_embeddings: Optional precomputed RoPE embeddings
            adarms_cond: Optional condition tensor for ADARMS
            **kwargs: Additional attention backend arguments

        Returns:
            Tuple containing:
            - hidden_states: [batch, seq_len, hidden_size]
            - attention_weights: Optional [batch, num_heads, seq_len, seq_len]
        """
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = _gated_residual(residual, hidden_states, gate)

        # Fully Connected
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        hidden_states = _gated_residual(residual, hidden_states, gate)

        if output_attentions:
            return (hidden_states, self_attn_weights)
        return (hidden_states,)


@auto_docstring
class GemmaPreTrainedModel(PreTrainedModel):
    """Base class for Gemma models.

    This class provides common functionality for all Gemma model variants,
    including weight initialization and support for various attention backends
    and optimization features.
    """

    config_class = GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different module types.

        Args:
            module: PyTorch module to initialize
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, GemmaRMSNorm):
            if hasattr(module, "weight"):
                module.weight.data.fill_(1.0)


@auto_docstring
class GemmaModel(GemmaPreTrainedModel):
    """Gemma decoder-only transformer model.

    This model implements a standard decoder-only transformer architecture
    with token embeddings, multiple decoder layers, and final layer normalization.
    Supports adaptive RMS normalization (ADARMS) for conditional generation.

    Args:
        config: Gemma configuration specifying model architecture
    """

    def __init__(self, config: GemmaConfig) -> None:
        """Initialize the Gemma model.

        Args:
            config: Gemma configuration containing model hyperparameters
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        cond_dim = (
            getattr(config, "adarms_cond_dim", None)
            if getattr(config, "use_adarms", False)
            else None
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )
        self.rotary_emb = GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input token embeddings.

        Returns:
            Embedding layer for input tokens
        """
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input token embeddings.

        Args:
            value: New embedding layer to use for input tokens
        """
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        """Run the forward pass for the decoder.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the input tokens.
            position_ids (`torch.LongTensor`, *optional*):
                Position indices for the input tokens.
            past_key_values (`Cache`, *optional*):
                Cached key/value states for faster decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Precomputed input embeddings.
            use_cache (`bool`, *optional*):
                Whether to return key/value cache.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            cache_position (`torch.LongTensor`, *optional*):
                Positions used for cache updates.
            adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
                Condition for ADARMS.
            **kwargs:
                Additional attention-related keyword arguments.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # embed positions
        hidden_states = inputs_embeds
        # Convert to bfloat16 if the first layer uses bfloat16
        if (
            len(self.layers) > 0
            and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to
        # become 55.5.
        # See https://github.com/huggingface/transformers/pull/29402
        torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        # hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states: tuple[torch.Tensor, ...] | None = (
            () if output_hidden_states else None
        )
        all_self_attns: tuple[torch.Tensor, ...] | None = (
            () if output_attentions else None
        )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                assert all_hidden_states is not None
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                assert all_self_attns is not None
                all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    """Type alias for keyword arguments accepted by GemmaForCausalLM.

    Combines flash attention kwargs and loss kwargs for type checking.
    """


@auto_docstring
class GemmaForCausalLM(GemmaPreTrainedModel, GenerationMixin):
    """Gemma model with a causal language modeling head.

    This model adds a language modeling head on top of the GemmaModel decoder,
    enabling next-token prediction for autoregressive text generation. Supports
    efficient logit computation via logits_to_keep parameter.

    Args:
        config: Gemma configuration specifying model architecture
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: GemmaConfig) -> None:
        """Initialize the causal language model.

        Args:
            config: Gemma configuration containing model hyperparameters
        """
        super().__init__(config)
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the input token embeddings.

        Returns:
            Embedding layer for input tokens
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set the input token embeddings.

        Args:
            value: New embedding layer to use for input tokens
        """
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Return the output token embeddings.

        Returns:
            Linear layer mapping hidden states to vocabulary logits
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set the output token embeddings.

        Args:
            new_embeddings: New linear layer for output embeddings
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: GemmaModel) -> None:
        """Set the decoder module.

        Args:
            decoder: GemmaModel instance to use as decoder
        """
        self.model = decoder

    def get_decoder(self) -> GemmaModel:
        """Return the decoder module.

        Returns:
            The underlying GemmaModel decoder
        """
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        adarms_cond: torch.Tensor | None = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        """Run the forward pass for causal language modeling.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for the input tokens.
            position_ids (`torch.LongTensor`, *optional*):
                Position indices for the input tokens.
            past_key_values (`Cache`, *optional*):
                Cached key/value states for faster decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Precomputed input embeddings.
            labels (`torch.LongTensor`, *optional*):
                Labels for computing the masked language modeling loss with
                shape `(batch_size, sequence_length)`. Indices should either be
                in `[0, ..., config.vocab_size]` or -100 (see `input_ids`
                docstring). Tokens with indices set to `-100` are ignored
                (masked), so the loss is computed only for tokens with labels in
                `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                Whether to return key/value cache.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            cache_position (`torch.LongTensor`, *optional*):
                Positions used for cache updates.
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                Number of logits to compute or indices to keep.
            adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
                Condition for ADARMS.
            **kwargs:
                Additional attention or loss-related keyword arguments.

        Example:
            ```python
            >>> from transformers import AutoTokenizer, GemmaForCausalLM

            >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
            >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

            >>> prompt = "What is your favorite condiment?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(
            ...     generate_ids,
            ...     skip_special_tokens=True,
            ...     clean_up_tokenization_spaces=False,
            ... )[0]
            "What is your favorite condiment?"
            ```
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            adarms_cond=adarms_cond,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we
        # are not computing the loss.
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "GemmaModel",
    "GemmaForCausalLM",
    "GemmaPreTrainedModel",
]
