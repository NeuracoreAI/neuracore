"""Core PyTorch modules for the GR00T N1.6 algorithm.

This module implements the major architectural components of the GR00T N1.6
Vision-Language-Action model:

  - VLMBackbone: Eagle-based vision-language backbone (Cosmos-Reason-2B)
  - MLPConnector: LayerNorm connector between VLM and DiT
  - DiTActionHead: 32-layer Diffusion Transformer for action generation

The DiT architecture uses interleaved self-attention and cross-attention
blocks with Adaptive Layer Normalization (AdaLN) conditioned on the
denoising timestep. Cross-attention blocks attend to the VLM context
embeddings, while self-attention blocks refine the state-action tokens.
In the default GR00T N1.6 configuration, cross-attention blocks alternate
between non-image tokens and image tokens instead of always attending to
the full VLM sequence.

Reference: NVIDIA GR00T N1.6 technical report.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch.nn.attention import SDPBackend, sdpa_kernel

# Re-export dataclass so callers can customise the bundled Eagle config
# without touching JSON files.
from .eagle_config.eagle3_vl_dataconfig import Eagle3VLDataConfig  # noqa: E402

logger = logging.getLogger(__name__)


def _patch_transformers_union_type_bug() -> None:
    """Patch transformers 4.53.x auto_docstring crash on Python 3.10+ union types.

    ``transformers.utils.args_doc._process_parameter_type`` accesses
    ``param.annotation.__name__`` which does not exist on ``types.UnionType``
    (the ``X | Y`` syntax).  This monkey-patches the function so that union
    type annotations are converted to their string representation first.
    The patch is idempotent.

    Can be removed once transformers is upgraded to >=5.0.0.
    """
    import types as _types

    from transformers.utils import args_doc as _args_doc

    if getattr(_args_doc, "_union_type_patched", False):
        return
    _orig = _args_doc._process_parameter_type

    def _safe_process_parameter_type(param: Any, param_name: str, func: Any) -> Any:
        if isinstance(param.annotation, _types.UnionType):
            param = param.replace(annotation=str(param.annotation))
        return _orig(param, param_name, func)

    _args_doc._process_parameter_type = _safe_process_parameter_type
    _args_doc._union_type_patched = True


_patch_transformers_union_type_bug()


### Architecture constants (from GR00T N1.6 default config)
# DiT defaults
DEFAULT_NUM_DIT_LAYERS = 32
DEFAULT_NUM_ATTENTION_HEADS = 32
DEFAULT_ATTENTION_HEAD_DIM = 48
DEFAULT_DIT_OUTPUT_DIM = 1024
DEFAULT_DIT_DROPOUT = 0.2

# VLM backbone defaults
DEFAULT_BACKBONE_EMBEDDING_DIM = 2048
DEFAULT_SELECT_LAYER = 16

# Timestep bucketing for discretized conditioning
DEFAULT_NUM_TIMESTEP_BUCKETS = 1000

# Positional embedding for state-action tokens
DEFAULT_MAX_SEQ_LEN = 1024


### Hardware-specific attention backend selection
def _is_spark_sm121() -> bool:
    """Detect NVIDIA Spark (compute capability 12.1) GPUs."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (12, 1)


def _should_force_math_sdpa() -> bool:
    """Return True if we must use the math SDPA backend.

    Spark sm12.1 hits a broken PyTorch mem-efficient SDPA kernel dispatch.
    Override via env var GR00T_DIT_SDPA_MODE=math|default.
    """
    override = os.environ.get("GR00T_DIT_SDPA_MODE")
    if override == "math":
        return True
    if override == "default":
        return False
    return _is_spark_sm121()


def _get_attn_implementation() -> str:
    """Return the best available attention implementation.

    Prefers ``flash_attention_2`` when the ``flash_attn`` package is installed,
    otherwise falls back to PyTorch's built-in ``sdpa``.
    """
    try:
        import flash_attn  # noqa: F401

        logger.info("flash_attn detected – using flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        logger.info("flash_attn not found – falling back to sdpa")
        return "sdpa"


def _sdpa_context() -> AbstractContextManager[None]:
    """Return a context manager that selects the correct SDPA backend.

    On most hardware this is a no-op. On Spark sm12.1 it forces the safe
    math backend to avoid numerical noise from the mem-efficient kernel.
    """
    if not _should_force_math_sdpa():
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


### Pretrained weight loading utilities
def _load_pretrained_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Download and load a GR00T N1.6 state dict from a checkpoint.

    Supports both local paths and HuggingFace Hub model identifiers.
    Handles both single-file and sharded safetensors checkpoints.

    Args:
        model_path: Local directory path or HuggingFace model ID
            (e.g., "nvidia/GR00T-N1.6-3B").

    Returns:
        Full state dict mapping parameter names to tensors.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded from the given path.
    """
    from safetensors.torch import load_file

    # --- Try local single-file checkpoint ---
    local_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(local_file):
        logger.info("Loading checkpoint from local path: %s", local_file)
        return load_file(local_file)

    # --- Try local sharded checkpoint ---
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info("Loading sharded checkpoint from: %s", model_path)
        return _load_sharded_safetensors(model_path, index_file)

    # --- Fall back to HuggingFace Hub download ---
    try:
        from huggingface_hub import hf_hub_download

        # Try single-file first
        try:
            weight_path = hf_hub_download(model_path, "model.safetensors")
            logger.info("Downloaded single-file checkpoint from HF Hub.")
            return load_file(weight_path)
        except Exception:
            pass

        # Try sharded checkpoint
        index_path = hf_hub_download(model_path, "model.safetensors.index.json")
        logger.info("Downloaded sharded checkpoint index from HF Hub.")
        return _load_sharded_safetensors_from_hub(model_path, index_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from '{model_path}'. "
            f"Ensure the path exists locally or is a valid HuggingFace model ID. "
            f"Error: {e}"
        ) from e


def _load_sharded_safetensors(
    model_dir: str, index_file: str
) -> dict[str, torch.Tensor]:
    """Load a sharded safetensors checkpoint from a local directory.

    Args:
        model_dir: Directory containing the shard files.
        index_file: Path to the model.safetensors.index.json file.

    Returns:
        Combined state dict from all shards.
    """
    from safetensors.torch import load_file

    with open(index_file) as f:
        index = json.load(f)

    # Collect unique shard filenames
    shard_files = set(index["weight_map"].values())
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(model_dir, shard_file)
        state_dict.update(load_file(shard_path))
    return state_dict


def _load_sharded_safetensors_from_hub(
    model_path: str, index_path: str
) -> dict[str, torch.Tensor]:
    """Load a sharded safetensors checkpoint from HuggingFace Hub.

    Args:
        model_path: HuggingFace model identifier.
        index_path: Local path to the downloaded index JSON file.

    Returns:
        Combined state dict from all downloaded shards.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    with open(index_path) as f:
        index = json.load(f)

    shard_files = set(index["weight_map"].values())
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(shard_files):
        local_path = hf_hub_download(model_path, shard_file)
        state_dict.update(load_file(local_path))
    return state_dict


def _load_config_json(model_path: str) -> dict:
    """Load config.json from a GR00T N1.6 checkpoint.

    Args:
        model_path: Local directory path or HuggingFace model ID.

    Returns:
        Parsed config dictionary. Returns empty dict if not found.
    """
    # Try local first
    local_file = os.path.join(model_path, "config.json")
    if os.path.exists(local_file):
        with open(local_file) as f:
            return json.load(f)

    # Try HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, "config.json")
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        logger.warning("Could not load config.json from %s", model_path)
        return {}


def _build_eagle_config_from_dataclass(
    dataconfig: Eagle3VLDataConfig | None = None,
) -> Any:
    """Build an ``Eagle3_VLConfig`` from a dataclass (no JSON on disk needed).

    Args:
        dataconfig: Dataclass holding Eagle3-VL configuration values.
            If *None*, the default ``Eagle3VLDataConfig()`` is used.

    Returns:
        An ``Eagle3_VLConfig`` instance ready for ``AutoModel.from_config``.
    """
    from .eagle_config.configuration_eagle3_vl import Eagle3_VLConfig

    if dataconfig is None:
        dataconfig = Eagle3VLDataConfig()

    cfg_dict = dataconfig.to_dict()
    # Remove fields that Eagle3_VLConfig doesn't accept directly
    for key in (
        "auto_map",
        "architectures",
        "model_type",
        "dynamic_image_size",
        "max_dynamic_tiles",
        "min_dynamic_tiles",
        "mlp_connector_layers",
        "use_pixel_shuffle",
        "use_thumbnail",
        "transformers_version",
    ):
        cfg_dict.pop(key, None)
    return Eagle3_VLConfig(**cfg_dict)


def _add_action_step_positional_encoding(
    *,
    action_tokens: torch.Tensor,
    action_position_embedding: nn.Embedding | None,
    max_action_pos_embeddings: int,
) -> torch.Tensor:
    """Add learned action-step positional encoding to action tokens."""
    if action_position_embedding is None:
        return action_tokens

    action_horizon = action_tokens.shape[1]
    if action_horizon > max_action_pos_embeddings:
        raise ValueError(
            f"Action horizon {action_horizon} exceeds "
            f"max_action_pos_embeddings={max_action_pos_embeddings}"
        )

    position_ids = torch.arange(action_horizon, device=action_tokens.device)
    position_embs = action_position_embedding(position_ids).unsqueeze(0)
    return action_tokens + position_embs.to(dtype=action_tokens.dtype)


### DiT building blocks (ported from GR00T N1.6 source)


class TimestepEncoder(nn.Module):
    """Encode discrete timesteps into continuous embeddings.

    Uses sinusoidal projection followed by a learned MLP to produce
    timestep embeddings for AdaLN conditioning in the DiT.

    Architecture:
        Timesteps(256) -> TimestepEmbedding(256, embedding_dim)

    Args:
        embedding_dim: Output embedding dimension (should match DiT inner_dim).
    """

    def __init__(self, embedding_dim: int):
        """Initialize timestep encoder with sinusoidal projection and MLP."""
        super().__init__()
        # Sinusoidal projection: scalar timestep -> 256-dim vector
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        # Learned MLP: 256-dim -> embedding_dim
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encode timesteps into embeddings.

        Args:
            timesteps: Discrete timestep values (B,).

        Returns:
            Timestep embeddings (B, embedding_dim).
        """
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        return timesteps_emb


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep embeddings.

    Applies standard layer normalization, then modulates the output with
    learned scale and shift parameters derived from the timestep embedding.
    This allows the DiT to adapt its normalization at each denoising step.

    Architecture:
        temb -> SiLU -> Linear -> (scale, shift)
        output = LayerNorm(x) * (1 + scale) + shift

    Args:
        embedding_dim: Dimension of the input and timestep embeddings.
        norm_elementwise_affine: Whether LayerNorm has learnable affine params.
        norm_eps: Epsilon for LayerNorm numerical stability.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        """Initialize adaptive layer norm with timestep modulation."""
        super().__init__()
        # Project timestep embedding to (scale, shift) pair
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        # Standard layer norm (no affine since AdaLN provides scale/shift)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor (B, T, D).
            temb: Timestep embedding (B, D).

        Returns:
            Normalized and modulated tensor (B, T, D).
        """
        # Compute adaptive scale and shift from timestep embedding
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        # Apply: norm(x) * (1 + scale) + shift
        # scale/shift are (B, D), broadcast over sequence dim T
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class BasicTransformerBlock(nn.Module):
    """Single transformer block for the DiT architecture.

    Each block contains:
      1. AdaLN-conditioned attention (self-attention or cross-attention)
      2. Feed-forward network with residual connection

    When cross_attention_dim is provided, the attention layer performs
    cross-attention to encoder hidden states (VLM context). Otherwise,
    it performs self-attention over the input sequence.

    Args:
        dim: Hidden dimension of the block.
        num_attention_heads: Number of attention heads.
        attention_head_dim: Dimension per attention head.
        dropout: Dropout probability.
        cross_attention_dim: If set, enables cross-attention to encoder states.
        activation_fn: Activation function for the feed-forward network.
        attention_bias: Whether attention layers use bias.
        upcast_attention: Whether to upcast attention to float32.
        norm_type: Normalization type ("ada_norm" for AdaLN).
        norm_elementwise_affine: Whether norms have learnable affine params.
        norm_eps: Epsilon for normalization layers.
        final_dropout: Whether to apply dropout after attention.
        positional_embeddings: Type of positional embeddings ("sinusoidal" or None).
        num_positional_embeddings: Max sequence length for positional embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
    ):
        """Initialize transformer block with attention and feed-forward layers."""
        super().__init__()

        # Optional sinusoidal positional embeddings
        if positional_embeddings == "sinusoidal":
            if num_positional_embeddings is None:
                raise ValueError(
                    "num_positional_embeddings must be set when using "
                    "sinusoidal positional embeddings."
                )
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # --- Block 1: Attention with AdaLN ---
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(
                dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=True,
        )

        # Optional dropout after attention
        self.attn_dropout = nn.Dropout(dropout) if final_dropout else None

        # --- Block 2: Feed-forward with residual ---
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )

        # Store norm_type for forward dispatch
        self.norm_type = norm_type

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process one transformer block.

        Args:
            hidden_states: Input tensor (B, T, D).
            attention_mask: Self-attention mask (B, T, T) or None.
            encoder_hidden_states: Cross-attention context (B, S, D_cross) or None.
            encoder_attention_mask: Cross-attention mask (B, S) or None.
            temb: Timestep embedding for AdaLN (B, D) or None.

        Returns:
            Output tensor (B, T, D).
        """
        # --- Attention ---
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # Add sinusoidal positional embeddings if configured
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # Attention: cross-attention if encoder states provided, else self-attention
        with _sdpa_context():
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=(
                    encoder_attention_mask
                    if encoder_hidden_states is not None
                    else attention_mask
                ),
            )

        if self.attn_dropout is not None:
            attn_output = self.attn_dropout(attn_output)

        # Residual connection
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # --- Feed-forward ---
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        # Residual connection
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


### DiT Action Head


class DiTActionHead(nn.Module):
    """Diffusion Transformer (DiT) action head for GR00T N1.6.

    A 32-layer transformer that generates actions via flow matching.
    Uses interleaved self-attention and cross-attention blocks:
      - Even-indexed blocks: cross-attention to VLM context
      - Odd-indexed blocks: self-attention over state-action tokens
    All blocks use Adaptive Layer Normalization (AdaLN) conditioned on
    the denoising timestep.

    The DiT takes pre-embedded state-action tokens and VLM context,
    and outputs refined representations that are decoded into velocity
    predictions by the action decoder MLP.

    Args:
        num_layers: Number of transformer blocks (default: 32).
        num_attention_heads: Number of attention heads (default: 32).
        attention_head_dim: Dimension per head (default: 48).
        output_dim: Output projection dimension (default: 1024).
        cross_attention_dim: Dimension of VLM context (default: 2048).
        dropout: Dropout probability (default: 0.2).
        interleave_self_attention: Whether to alternate self/cross attention.
        use_alternate_vl_dit: Whether cross-attention blocks should alternate
            between non-image tokens and image tokens, matching upstream
            AlternateVLDiT.
        attend_text_every_n_blocks: Alternation cadence for the upstream
            AlternateVLDiT schedule.
        num_timestep_buckets: Number of discrete timestep bins (default: 1000).
    """

    def __init__(
        self,
        num_layers: int = DEFAULT_NUM_DIT_LAYERS,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
        attention_head_dim: int = DEFAULT_ATTENTION_HEAD_DIM,
        output_dim: int = DEFAULT_DIT_OUTPUT_DIM,
        cross_attention_dim: int = DEFAULT_BACKBONE_EMBEDDING_DIM,
        dropout: float = DEFAULT_DIT_DROPOUT,
        interleave_self_attention: bool = True,
        use_alternate_vl_dit: bool = True,
        attend_text_every_n_blocks: int = 2,
        num_timestep_buckets: int = DEFAULT_NUM_TIMESTEP_BUCKETS,
    ):
        """Initialize DiT action head with interleaved transformer blocks."""
        super().__init__()

        # inner_dim = num_heads * head_dim = 32 * 48 = 1536
        self.inner_dim = num_attention_heads * attention_head_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.interleave_self_attention = interleave_self_attention
        self.use_alternate_vl_dit = use_alternate_vl_dit
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.num_timestep_buckets = num_timestep_buckets

        if self.use_alternate_vl_dit and not self.interleave_self_attention:
            raise ValueError("AlternateVLDiT requires interleave_self_attention=True.")

        # Timestep encoding: discrete integer -> continuous embedding
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim)

        # Build transformer blocks with interleaved attention pattern
        all_blocks = []
        for idx in range(num_layers):
            # Odd blocks: self-attention (no cross-attention dim)
            # Even blocks: cross-attention to VLM context
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = None if use_self_attn else cross_attention_dim

            all_blocks.append(
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu-approximate",
                    attention_bias=True,
                    upcast_attention=False,
                    norm_type="ada_norm",
                    norm_elementwise_affine=False,
                    norm_eps=1e-5,
                    positional_embeddings=None,
                    num_positional_embeddings=None,
                    final_dropout=True,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output projection: inner_dim -> output_dim
        # Uses AdaLN-style modulation followed by linear projection
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.output_dim)

        self.gradient_checkpointing = False

    @property
    def input_dim(self) -> int:
        """Input embedding dimension (= num_heads * head_dim = 1536)."""
        return self.inner_dim

    @property
    def hidden_dim(self) -> int:
        """Alias for output_dim, used by the main model class."""
        return self.output_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the DiT forward pass.

        Args:
            hidden_states: Pre-embedded state-action tokens
                (B, 1 + action_horizon, inner_dim).
            timestep: Discretized denoising timestep (B,) as integers.
            encoder_hidden_states: VLM context embeddings
                (B, num_vl_tokens, cross_attention_dim).
            encoder_attention_mask: Boolean mask identifying valid VLM tokens
                (B, num_vl_tokens).
            image_mask: Boolean mask identifying image tokens inside the VLM
                context (B, num_vl_tokens). Required when
                ``use_alternate_vl_dit`` is enabled.

        Returns:
            Output tensor (B, 1 + action_horizon, output_dim).
        """
        # Encode timestep for AdaLN conditioning
        temb = self.timestep_encoder(timestep)

        # Ensure contiguous memory layout for efficient attention
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(dtype=torch.bool)
        if image_mask is not None:
            image_mask = image_mask.to(dtype=torch.bool)

        # Process through interleaved transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            is_self_attn = idx % 2 == 1 and self.interleave_self_attention
            enc_states = None if is_self_attn else encoder_hidden_states
            enc_mask = None

            if not is_self_attn:
                enc_mask = self._select_encoder_attention_mask(
                    block_idx=idx,
                    encoder_attention_mask=encoder_attention_mask,
                    image_mask=image_mask,
                )

            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,  # attention_mask
                    enc_states,
                    enc_mask,
                    temb,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=enc_states,
                    encoder_attention_mask=enc_mask,
                    temb=temb,
                )

        # Output projection with AdaLN-style modulation
        # shift/scale from timestep embedding modulate the final LayerNorm
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        return self.proj_out_2(hidden_states)

    def _select_encoder_attention_mask(
        self,
        block_idx: int,
        encoder_attention_mask: torch.Tensor | None,
        image_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Choose the VLM token subset for a cross-attention block.

        When AlternateVLDiT is enabled, even-indexed cross-attention blocks
        alternate between non-image and image tokens. If the selected subset is
        empty for a sample, fall back to the full valid-token mask to avoid
        degenerate all-masked attention rows.
        """
        if not self.use_alternate_vl_dit:
            return encoder_attention_mask

        if image_mask is None:
            raise ValueError("image_mask is required when use_alternate_vl_dit=True.")

        full_mask = (
            encoder_attention_mask
            if encoder_attention_mask is not None
            else torch.ones_like(image_mask, dtype=torch.bool)
        )
        image_attention_mask = image_mask & full_mask
        non_image_attention_mask = (~image_mask) & full_mask

        if block_idx % (2 * self.attend_text_every_n_blocks) == 0:
            selected_mask = non_image_attention_mask
        else:
            selected_mask = image_attention_mask

        has_selected_tokens = selected_mask.any(dim=-1, keepdim=True)
        return torch.where(has_selected_tokens, selected_mask, full_mask)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        use_alternate_vl_dit: bool | None = None,
        attend_text_every_n_blocks: int | None = None,
    ) -> DiTActionHead:
        """Load DiT weights from a GR00T N1.6 checkpoint.

        Extracts action_head.model.* weights from the checkpoint and
        loads them into a freshly constructed DiT module.

        Args:
            model_path: Local path or HuggingFace model ID for the
                GR00T N1.6 checkpoint.
            use_alternate_vl_dit: Optional override for AlternateVLDiT mode.
                When None, use the checkpoint config value.
            attend_text_every_n_blocks: Optional override for the alternating
                token schedule. When None, use the checkpoint config value.

        Returns:
            DiTActionHead with pretrained weights loaded.
        """
        # Read config to get architecture hyperparameters
        config = _load_config_json(model_path)
        dit_cfg = config.get("diffusion_model_cfg", {})

        dit = cls(
            num_layers=dit_cfg.get("num_layers", DEFAULT_NUM_DIT_LAYERS),
            num_attention_heads=dit_cfg.get(
                "num_attention_heads", DEFAULT_NUM_ATTENTION_HEADS
            ),
            attention_head_dim=dit_cfg.get(
                "attention_head_dim", DEFAULT_ATTENTION_HEAD_DIM
            ),
            output_dim=dit_cfg.get("output_dim", DEFAULT_DIT_OUTPUT_DIM),
            cross_attention_dim=config.get(
                "backbone_embedding_dim", DEFAULT_BACKBONE_EMBEDDING_DIM
            ),
            dropout=dit_cfg.get("dropout", DEFAULT_DIT_DROPOUT),
            interleave_self_attention=dit_cfg.get("interleave_self_attention", True),
            use_alternate_vl_dit=(
                config.get("use_alternate_vl_dit", True)
                if use_alternate_vl_dit is None
                else use_alternate_vl_dit
            ),
            attend_text_every_n_blocks=(
                config.get("attend_text_every_n_blocks", 2)
                if attend_text_every_n_blocks is None
                else attend_text_every_n_blocks
            ),
            num_timestep_buckets=config.get(
                "num_timestep_buckets", DEFAULT_NUM_TIMESTEP_BUCKETS
            ),
        )

        # Load and filter state dict for DiT weights
        full_state_dict = _load_pretrained_state_dict(model_path)
        dit_prefix = "action_head.model."
        dit_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith(dit_prefix):
                new_key = key[len(dit_prefix) :]
                dit_state_dict[new_key] = value

        missing, unexpected = dit.load_state_dict(dit_state_dict, strict=False)
        if missing:
            logger.warning("DiT missing keys: %s", missing)
        if unexpected:
            logger.warning("DiT unexpected keys: %s", unexpected)

        logger.info(
            "Loaded DiT action head from %s (%d parameters)",
            model_path,
            sum(p.numel() for p in dit.parameters()),
        )
        return dit


### VLM Backbone
class VLMBackbone(nn.Module):
    """Eagle vision-language backbone for GR00T N1.6.

    Wraps the Eagle-Block2A-2B-v2 model (based on SigLip2 vision encoder
    + Qwen3-1.7B LLM) to produce visual-language embeddings that serve
    as cross-attention context for the DiT action head.

    The backbone processes interleaved image and text tokens through the
    vision encoder, MLP projector, and a truncated LLM (first
    ``select_layer`` layers) to produce rich multi-modal embeddings.

    Args:
        select_layer: Number of LLM layers to keep (default: 16).
        backbone_embedding_dim: Output embedding dimension (default: 2048).
    """

    def __init__(
        self,
        select_layer: int = DEFAULT_SELECT_LAYER,
        backbone_embedding_dim: int = DEFAULT_BACKBONE_EMBEDDING_DIM,
    ):
        """Initialize VLM backbone with empty model placeholder."""
        super().__init__()
        self.select_layer = select_layer
        self.backbone_embedding_dim = backbone_embedding_dim
        # The Eagle model is set by from_pretrained
        self.model: nn.Module | None = None
        self._image_token_index: int | None = None

    @property
    def image_token_index(self) -> int:
        """Token ID used as placeholder for image features in input_ids."""
        if self._image_token_index is not None:
            return self._image_token_index
        if self.model is not None and hasattr(self.model, "config"):
            return self.model.config.image_token_index
        raise ValueError("image_token_index not available; load model first.")

    @property
    def llm_layers(self) -> nn.ModuleList:
        """Access the LLM transformer layers for optional finetuning."""
        assert self.model is not None, "VLM backbone not initialized."
        return self.model.language_model.model.layers

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the VLM backbone forward pass.

        Args:
            pixel_values: Preprocessed images (B, C, H, W).
            input_ids: Token IDs with image placeholders (B, L).
            attention_mask: Attention mask (B, L).

        Returns:
            Tuple of:
              - hidden_states: VLM output embeddings
                  (B, L, backbone_embedding_dim).
              - attention_mask: Boolean attention mask (B, L).
              - image_mask: Boolean mask identifying image tokens (B, L).
        """
        if self.model is None:
            raise RuntimeError("VLM backbone not initialized. Call from_pretrained.")

        # Eagle's Siglip2 embedding layer iterates over pixel_values and calls
        # convert_images_to_patches(each, ...) which unpacks (B, C, H, W).
        # When pixel_values is a single (N, C, H, W) tensor, iterating yields
        # 3D tensors (C, H, W) causing a shape mismatch. Convert to a list of
        # (1, C, H, W) tensors so each element has the expected 4 dimensions.
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 4:
            pixel_values = [img.unsqueeze(0) for img in pixel_values]

        # Run Eagle model forward pass and extract hidden states
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        # Use hidden states from the last layer
        hidden_states = outputs["hidden_states"][-1]

        # Create masks for downstream use
        image_mask = input_ids == self.image_token_index
        attn_mask = attention_mask == 1

        return hidden_states, attn_mask, image_mask

    @classmethod
    def from_random_init(
        cls,
        model_path: str,
        select_layer: int = DEFAULT_SELECT_LAYER,
    ) -> VLMBackbone:
        """Create VLM backbone with random weights (no pretrained loading).

        Downloads only the Eagle architecture config and creates the model
        with random initialization. Useful for training from scratch.

        Args:
            model_path: Local path or HuggingFace model ID for the
                GR00T N1.6 checkpoint (used to read architecture config).
            select_layer: Number of LLM layers to keep.

        Returns:
            VLMBackbone with randomly initialized weights.
        """
        from .eagle_config.modeling_eagle3_vl import Eagle3_VLForConditionalGeneration

        config = _load_config_json(model_path)
        eagle_model_name = config.get("model_name", "nvidia/Eagle-Block2A-2B-v2")
        backbone_embedding_dim = config.get(
            "backbone_embedding_dim", DEFAULT_BACKBONE_EMBEDDING_DIM
        )
        select_layer = config.get("select_layer", select_layer)

        obj = cls(
            select_layer=select_layer,
            backbone_embedding_dim=backbone_embedding_dim,
        )

        # Build Eagle config: use the Python dataclass for the bundled default,
        # fall back to AutoConfig.from_pretrained for external models.
        attn_impl = _get_attn_implementation()
        if eagle_model_name == "nvidia/Eagle-Block2A-2B-v2":
            logger.info(
                "Creating Eagle architecture from dataclass defaults (random init)"
            )
            eagle_config = _build_eagle_config_from_dataclass()
        else:
            from transformers import AutoConfig

            logger.info(
                "Creating Eagle architecture from %s (random init)", eagle_model_name
            )
            eagle_config = AutoConfig.from_pretrained(
                eagle_model_name, trust_remote_code=True
            )
        eagle_config._attn_implementation = attn_impl
        if hasattr(eagle_config, "vision_config"):
            eagle_config.vision_config._attn_implementation = attn_impl
        obj.model = Eagle3_VLForConditionalGeneration(eagle_config)
        obj._image_token_index = eagle_config.image_token_index

        while len(obj.model.language_model.model.layers) > select_layer:
            obj.model.language_model.model.layers.pop(-1)

        logger.info(
            "Created VLM backbone with random init (%d layers, %d parameters)",
            select_layer,
            sum(p.numel() for p in obj.parameters()),
        )
        return obj

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        select_layer: int = DEFAULT_SELECT_LAYER,
    ) -> VLMBackbone:
        """Load VLM backbone from a GR00T N1.6 checkpoint.

        Creates an Eagle model architecture and loads the backbone weights
        from the GR00T checkpoint. The LLM is truncated to ``select_layer``
        layers since the DiT uses intermediate (not final) representations.

        Args:
            model_path: Local path or HuggingFace model ID for the
                GR00T N1.6 checkpoint.
            select_layer: Number of LLM layers to keep.

        Returns:
            VLMBackbone with pretrained weights loaded.
        """
        from transformers import AutoConfig

        from .eagle_config.modeling_eagle3_vl import Eagle3_VLForConditionalGeneration

        # Read GR00T config for backbone settings
        config = _load_config_json(model_path)
        eagle_model_name = config.get("model_name", "nvidia/Eagle-Block2A-2B-v2")
        backbone_embedding_dim = config.get(
            "backbone_embedding_dim", DEFAULT_BACKBONE_EMBEDDING_DIM
        )
        select_layer = config.get("select_layer", select_layer)

        obj = cls(
            select_layer=select_layer,
            backbone_embedding_dim=backbone_embedding_dim,
        )

        # Create Eagle architecture with empty weights.
        # Use the Python dataclass for the bundled default, fall back to
        # AutoConfig.from_pretrained for external models.
        attn_impl = _get_attn_implementation()
        if eagle_model_name == "nvidia/Eagle-Block2A-2B-v2":
            logger.info("Creating Eagle architecture from dataclass defaults")
            eagle_config = _build_eagle_config_from_dataclass()
        else:
            logger.info("Creating Eagle architecture from %s", eagle_model_name)
            eagle_config = AutoConfig.from_pretrained(
                eagle_model_name, trust_remote_code=True
            )
        eagle_config._attn_implementation = attn_impl
        if hasattr(eagle_config, "vision_config"):
            eagle_config.vision_config._attn_implementation = attn_impl
        obj.model = Eagle3_VLForConditionalGeneration(eagle_config)
        obj._image_token_index = eagle_config.image_token_index

        # Truncate LLM layers to select_layer
        while len(obj.model.language_model.model.layers) > select_layer:
            obj.model.language_model.model.layers.pop(-1)

        # Load backbone weights from GR00T checkpoint
        full_state_dict = _load_pretrained_state_dict(model_path)
        backbone_prefix = "backbone.model."
        backbone_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith(backbone_prefix):
                new_key = key[len(backbone_prefix) :]
                backbone_state_dict[new_key] = value

        missing, unexpected = obj.model.load_state_dict(
            backbone_state_dict, strict=False
        )
        # Filter out expected missing keys (layers we truncated)
        missing = [k for k in missing if "language_model.model.layers." not in k]
        if missing:
            logger.warning("VLM backbone missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("VLM backbone unexpected keys: %s", unexpected[:10])

        logger.info(
            "Loaded VLM backbone (%d layers, %d parameters)",
            select_layer,
            sum(p.numel() for p in obj.parameters()),
        )
        return obj


# MLP Connector
class TinyVLMBackbone(nn.Module):
    """Minimal VLM backbone for unit testing without HuggingFace downloads.

    Replaces the full Eagle VLM with a simple linear projection that
    produces correctly-shaped outputs. This avoids trust_remote_code,
    large model downloads, and GPU requirements during testing.

    Args:
        backbone_embedding_dim: Output embedding dimension (default: 64).
        num_llm_layers: Number of dummy LLM layers for the llm_layers
            property (default: 4).
    """

    def __init__(
        self,
        backbone_embedding_dim: int = 64,
        num_llm_layers: int = 4,
    ):
        """Initialize tiny VLM with embedding projection and dummy layers."""
        super().__init__()
        self.backbone_embedding_dim = backbone_embedding_dim
        # Simple projection from pixel space to embedding space
        # Input: flattened 3x448x448 patch -> we just use a small projection
        self.image_proj = nn.Linear(3 * 16 * 16, backbone_embedding_dim)
        self.text_proj = nn.Embedding(200000, backbone_embedding_dim)
        # Dummy LLM layers for the llm_layers property (needed for VLM finetuning)
        self._llm_layers = nn.ModuleList([
            nn.Linear(backbone_embedding_dim, backbone_embedding_dim)
            for _ in range(num_llm_layers)
        ])
        self._image_token_index = 151669

    @property
    def image_token_index(self) -> int:
        """Token ID used as placeholder for image features in input_ids."""
        return self._image_token_index

    @property
    def llm_layers(self) -> nn.ModuleList:
        """Access the dummy LLM layers for optional finetuning."""
        return self._llm_layers

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run tiny VLM forward pass.

        Produces correctly-shaped outputs by projecting text tokens
        through an embedding layer. Image tokens are replaced with
        a simple projection of downsampled pixel patches.

        Args:
            pixel_values: Preprocessed images (B*num_cam, 3, 448, 448).
            input_ids: Token IDs with image placeholders (B, L).
            attention_mask: Attention mask (B, L).

        Returns:
            Tuple of (hidden_states, attn_mask, image_mask).
        """
        B, L = input_ids.shape

        # Start with text embeddings for all tokens
        hidden_states = self.text_proj(input_ids)  # (B, L, D)

        # Create masks
        image_mask = input_ids == self._image_token_index
        attn_mask = attention_mask == 1

        return hidden_states, attn_mask, image_mask


class MLPConnector(nn.Module):
    """Vision-language LayerNorm connector between VLM and DiT.

    Applies LayerNorm to the VLM backbone output features before they
    are used as cross-attention context in the DiT. This corresponds
    to the ``vlln`` (vision-language layer norm) module in the original
    GR00T N1.6 architecture.

    Args:
        embedding_dim: Dimension of the VLM output embeddings (default: 2048).
    """

    def __init__(self, embedding_dim: int = DEFAULT_BACKBONE_EMBEDDING_DIM):
        """Initialize MLP connector with layer normalization."""
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, vl_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to VLM embeddings.

        Args:
            vl_embeddings: VLM output features
                (B, num_tokens, embedding_dim).

        Returns:
            Normalized features (B, num_tokens, embedding_dim).
        """
        return self.layer_norm(vl_embeddings)

    @classmethod
    def from_pretrained(cls, model_path: str) -> MLPConnector:
        """Load MLPConnector weights from a GR00T N1.6 checkpoint.

        Extracts the action_head.vlln.* weights from the checkpoint.

        Args:
            model_path: Local path or HuggingFace model ID.

        Returns:
            MLPConnector with pretrained weights loaded.
        """
        config = _load_config_json(model_path)
        embedding_dim = config.get(
            "backbone_embedding_dim", DEFAULT_BACKBONE_EMBEDDING_DIM
        )

        connector = cls(embedding_dim=embedding_dim)

        # Load and filter state dict for connector weights
        full_state_dict = _load_pretrained_state_dict(model_path)
        vlln_prefix = "action_head.vlln."
        connector_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith(vlln_prefix):
                # Map action_head.vlln.weight -> layer_norm.weight
                new_key = key[len(vlln_prefix) :]
                connector_state_dict[f"layer_norm.{new_key}"] = value

        if connector_state_dict:
            missing, unexpected = connector.load_state_dict(
                connector_state_dict, strict=False
            )
            if missing:
                logger.warning("MLPConnector missing keys: %s", missing)
            if unexpected:
                logger.warning("MLPConnector unexpected keys: %s", unexpected)
            logger.info("Loaded MLP connector from %s", model_path)
        else:
            logger.warning("No vlln weights found in checkpoint; using random init.")

        return connector
