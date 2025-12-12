"""PI0 reference implementation (fully local, no external lerobot deps)."""

# cspell:ignore OPENPI adarms layernorm silu huggingface openpi

from __future__ import annotations

import builtins
import logging
import math
<<<<<<< HEAD
from collections.abc import Callable
from dataclasses import asdict, dataclass
=======
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Iterator, Literal, Optional, TypeVar
>>>>>>> 673255f (temp add)

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

T = TypeVar("T")

try:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
    )
    from transformers.utils import cached_file
except Exception:  # pragma: no cover
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None
    cached_file = None

try:
    from safetensors.torch import load_file
except Exception:  # pragma: no cover
    load_file = None

ACTION = "action"
OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
OBS_STATE = "observation.state"
OPENPI_ATTENTION_MASK_VALUE = -1e9

logger = logging.getLogger(__name__)


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    device = torch.device(device)
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    bsize: int,
    device: torch.device | str,
) -> Tensor:
    """Sample beta-distributed scalars."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Convert 1D padding/attention masks into a causal 2D mask."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Right-pad the last dimension of a tensor to ``new_dim``."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def _align_mask_length(mask_1d: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or trim a 1D mask to a target length."""
    current_len = mask_1d.shape[0]
    if current_len == target_len:
        return mask_1d
    if current_len < target_len:
        pad = torch.zeros(
            target_len - current_len, device=mask_1d.device, dtype=mask_1d.dtype
        )
        return torch.cat([mask_1d, pad], dim=0)
    return mask_1d[:target_len]

def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize image to target size with padding (channels-first or last)."""
    if images.shape[-1] <= 4:  # assume channels-last
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)
    return padded_images
>>>>>>> 673255f (temp add)


def compute_layer_complete(
    layer_idx: int,
    inputs_embeds: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    adarms_cond: list[Tensor | None],
    paligemma: PaliGemmaForConditionalGeneration,
    gemma_expert: GemmaForCausalLM,
) -> list[Tensor]:
    """Run a single transformer layer jointly across prefix/suffix branches."""
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        if adarms_cond[i] is None:
            hidden_states = layer.input_layernorm(hidden_states)[0]  # noqa: PLW2901
            gate = None
        else:
            hidden_states, gate = layer.input_layernorm(
                hidden_states, cond=adarms_cond[i]
            )  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = (
            layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
<<<<<<< HEAD

        # Compute mixed attention output
        mixed_output = torch.matmul(attn_weights, values)
        return mixed_output

    def forward(
        self,
        hidden_states: dict[str, torch.FloatTensor],
        expert_attention_masks: dict[str, torch.Tensor] | None = None,
        mix_attention_mask: torch.Tensor | None = None,
        position_ids: dict[str, torch.LongTensor] | None = None,
        past_key_values: dict[str, DynamicCache] | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.FloatTensor]:
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

    def _init_caches(self) -> dict[str, DynamicCache]:
        """Initialize caches for the experts.

        Returns:
            Dict[str, DynamicCache]: Initialized caches.
        """
        return {name: DynamicCache() for name in self.cache_names}

    def _normalize_inputs(
        self, hidden_states: dict[str, torch.FloatTensor]
    ) -> dict[str, torch.FloatTensor]:
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
        hidden_states: dict[str, torch.FloatTensor],
        expert_attention_masks: dict[str, torch.Tensor] | None = None,
        mix_attention_mask: torch.Tensor | None = None,
        position_ids: dict[str, torch.LongTensor] | None = None,
        past_key_values: dict[str, DynamicCache] | None = None,
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
        self, action: torch.Tensor, time_emb: torch.Tensor | None = None
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
=======
        key_state = (
            layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        value_state = (
            layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
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
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = modeling_gemma._gated_residual(
            hidden_states, out_emb, gates[i]
        )  # noqa: SLF001
        after_first_residual = out_emb.clone()
        if adarms_cond[i] is None:
            out_emb = layer.post_attention_layernorm(out_emb)[0]
            gate = None
        else:
            out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = modeling_gemma._gated_residual(
            after_first_residual, out_emb, gate
        )  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds
>>>>>>> 673255f (temp add)


class GemmaConfig:
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
        """Initialize a Gemma configuration."""
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:
    """Return the GemmaConfig for a known variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for PI0."""

    def __init__(
        self,
        vlm_config: GemmaConfig,
        action_expert_config: GemmaConfig,
        use_adarms: tuple[bool, bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> None:
        """Initialize the joint vision-language and action expert model."""
        if use_adarms is None:
            use_adarms = (False, False)
        super().__init__()

        paligemma_config = CONFIG_MAPPING["paligemma"]()
        paligemma_config._vocab_size = 257152  # noqa: SLF001
        paligemma_config.image_token_index = 257152
        paligemma_config.text_config.hidden_size = vlm_config.width
        paligemma_config.text_config.intermediate_size = vlm_config.mlp_dim
        paligemma_config.text_config.num_attention_heads = vlm_config.num_heads
        paligemma_config.text_config.head_dim = vlm_config.head_dim
        paligemma_config.text_config.num_hidden_layers = vlm_config.depth
        paligemma_config.text_config.num_key_value_heads = vlm_config.num_kv_heads
        paligemma_config.text_config.hidden_activation = "gelu_pytorch_tanh"
        paligemma_config.text_config.torch_dtype = "float32"
        paligemma_config.text_config.vocab_size = 257152
        paligemma_config.text_config.use_adarms = use_adarms[0]
        paligemma_config.text_config.adarms_cond_dim = (
            vlm_config.width if use_adarms[0] else None
        )
        paligemma_config.vision_config.intermediate_size = 4304
        paligemma_config.vision_config.projection_dim = 2048
        paligemma_config.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config.vision_config.torch_dtype = "float32"

        action_expert_config_gemma = CONFIG_MAPPING["gemma"](
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

        self.paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_gemma)
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ) -> None:
        """Move parameters to bfloat16, keeping sensitive ones in float32."""
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
        """Embed an input image using the vision tower."""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens with the language model tokenizer."""
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[torch.FloatTensor] | None]:
        """Forward pass for prefix (vision/lang) and suffix (action) branches."""
        adarms: list[torch.Tensor | None]
        if adarms_cond is None:
            adarms = [None, None]
        else:
            adarms = adarms_cond
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms[0],
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
                adarms_cond=adarms[1],
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            def compute_final_norms(
                inputs_embeds: list[Tensor], adarms_cond: list[Tensor | None]
            ) -> list[Tensor]:
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI0Pytorch(nn.Module):
    """Core PI0 PyTorch model."""

    def __init__(self, config: PI0Config):
        """Initialize the PI0 model and projection heads."""
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=config.use_adarms,
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(
            config.max_action_dim, action_expert_config.width
        )
        self.action_out_proj = nn.Linear(
            action_expert_config.width, config.max_action_dim
        )

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.width, action_expert_config.width
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(  # type: ignore[method-assign]
                self.sample_actions, mode=config.compile_mode
            )
            self.forward = torch.compile(  # type: ignore[method-assign]
                self.forward, mode=config.compile_mode
            )

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on all submodules."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            True
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on all submodules."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def _apply_checkpoint(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Optionally wrap a function call with PyTorch checkpointing."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        """Expand 2D masks to 4D and apply OPENPI fill value."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(
        self, shape: torch.Size | tuple[int, ...], device: torch.device
    ) -> Tensor:
        """Sample standard normal noise."""
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Sample diffusion time steps."""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = (
            time_beta * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed image and language prefix tokens and build attention masks."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1).to(dtype=torch.bool)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_t = _align_mask_length(att_masks_t, pad_masks.shape[1])
        bsize = pad_masks.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, att_masks_t.shape[0])
        return embs, pad_masks, att_masks_t

    def embed_suffix(
        self, state: Tensor, noisy_actions: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        """Embed state, actions, and timestep for the suffix branch."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state: Tensor) -> Tensor:
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions: Tensor) -> Tensor:
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb: Tensor) -> Tensor:
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks_t = _align_mask_length(att_masks_t, pad_masks.shape[1])
        att_masks_t = att_masks_t[None, :].expand(bsize, att_masks_t.shape[0])

        return embs, pad_masks, att_masks_t, adarms_cond

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Compute denoising loss for a batch."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs: Tensor,
            suffix_embs: Tensor,
            att_2d_masks_4d: Tensor,
            position_ids: Tensor,
            adarms_cond: Tensor | None,
        ) -> Tensor:
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out: Tensor) -> Tensor:
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Run Euler denoising to sample an action chunk."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        paligemma_lm_config = self.paligemma_with_expert.paligemma.language_model.config
        paligemma_lm_config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def denoise_step(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values: list[torch.FloatTensor] | None,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Single Euler step used during sampling."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        gemma_config = self.paligemma_with_expert.gemma_expert.model.config
        gemma_config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        assert suffix_out is not None
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


@dataclass
class PI0Config:
    """Configuration for the PI0 model and training hyperparameters."""

    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: Literal["bfloat16", "float32"] = "float32"
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    use_adarms: tuple[bool, bool] = (False, False)
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None
    input_features: dict = field(default_factory=dict)
    output_features: dict = field(default_factory=dict)
    image_features: list[str] = field(default_factory=list)

    def validate_features(self) -> None:
        """Validate configured feature dimensions."""
        if self.device is None:
            self.device = "cpu"

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")


class PI0Policy:
    """Lightweight policy wrapper without external deps."""

    config_class = PI0Config
    name = "pi0"

    def __init__(self, config: PI0Config, **kwargs: Any):
        """Construct a PI0 policy and initialize the underlying model."""
        self.config = config
        self.model = PI0Pytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)

    @classmethod
    def from_pretrained(
        cls: builtins.type["PI0Policy"],
        pretrained_name_or_path: str | Path | None = None,
        *,
        config: Optional[PI0Config] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "PI0Policy":
        """Load a pretrained PI0 model from HuggingFace Hub or local path.

        Args:
            pretrained_name_or_path: HuggingFace repo id (e.g. "lerobot/pi0_base")
                or local path. Defaults to "lerobot/pi0_base".
            config: Optional PI0Config. If None, uses default config.
            strict: Whether to strictly enforce state dict key matching.
            **kwargs: Additional arguments passed to huggingface_hub.

        Returns:
            PI0Policy with loaded weights.
        """
        # Default to the official pi0_base weights from Physical Intelligence / LeRobot
        if pretrained_name_or_path is None:
            pretrained_name_or_path = "lerobot/pi0_base"
            logging.warning(
                "No pretrained model path provided; using default pi0_base model"
            )
        if config is None:
            config = PI0Config()
        model = cls(config, **kwargs)

        if cached_file is None or load_file is None:
            logging.warning(
                "transformers/safetensors not available; loading weights skipped"
            )
            return model

        try:
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                resume_download=kwargs.get("resume_download"),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token") or kwargs.get("use_auth_token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            original_state_dict = load_file(resolved_file)
            logging.info("Loaded state dict from %s", resolved_file)
        except Exception as e:
            logging.warning(
                "Could not load state dict from %s: %s", pretrained_name_or_path, e
            )
            return model

        fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict)
        # # Remove 'model.' prefix if present (checkpoint may have it, but PI0Pytorch
        # # doesn't)
        remapped_state_dict = {}
        for key, value in fixed_state_dict.items():
            # Remove 'model.' prefix if it exists at the start
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
            else:
                new_key = key
            remapped_state_dict[new_key] = value

        missing_keys, unexpected_keys = model.load_state_dict(
            fixed_state_dict, strict=False
        )
        if missing_keys:
            logging.warning("Missing keys when loading state dict: %s", missing_keys)
        if unexpected_keys:
            logging.warning(
                "Unexpected keys when loading state dict: %s", unexpected_keys
            )
        tie_key = (
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        )
        if tie_key in missing_keys:
            paligemma = model.model.paligemma_with_expert.paligemma
            if model._tie_or_copy_language_embeddings(paligemma):
                print("Tied language embeddings to lm_head weight")
                missing_keys = [key for key in missing_keys if key != tie_key]
        logging.warning(
            "Missing keys after tying language embeddings: %s", missing_keys
        )
        logging.info(
            "Successfully loaded pretrained PI0 weights from %s",
            pretrained_name_or_path,
        )
        return model

    def _tie_or_copy_language_embeddings(
        self, paligemma: PaliGemmaForConditionalGeneration
    ) -> bool:
        """Tie or copy language embeddings to lm_head weight."""
        language_model = getattr(
            getattr(paligemma, "model", None), "language_model", None
        )
        lm_head = getattr(paligemma, "lm_head", None)
        if language_model is None or lm_head is None:
            return False

        embed_tokens = getattr(language_model, "embed_tokens", None)
        lm_head_weight = getattr(lm_head, "weight", None)
        if embed_tokens is None or lm_head_weight is None:
            return False

        embed_weight = getattr(embed_tokens, "weight", None)
        if embed_weight is None or embed_weight.shape != lm_head_weight.shape:
            return False

        with torch.no_grad():
            embed_weight.copy_(lm_head_weight)

        if hasattr(paligemma, "tie_weights"):
            paligemma.tie_weights()

        tied_embed = getattr(language_model.embed_tokens, "weight", None)
        return (
            tied_embed is not None
            and tied_embed.data_ptr() == lm_head_weight.data_ptr()
        )

    def _fix_pytorch_state_dict_keys(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                (
                    r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
                    r"(input_layernorm|post_attention_layernorm)\.weight"
                ),
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config,
                    "use_adarms",
                    False,
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config,
                    "use_adarms",
                    False,
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi0
            # non-pi05 model expects action_time_mlp_*; checkpoint might use time_mlp_*
            if key.startswith("time_mlp_in."):
                new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."):
                new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might include this; current model expects different
                # structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def to(self, device: torch.device | str) -> "PI0Policy":
        """Move underlying model to the specified device."""
        self.model.to(device)
        self.config.device = device
        return self

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate load_state_dict to the underlying torch module."""
        return self.model.load_state_dict(*args, **kwargs)

    def parameters(self) -> Iterator[nn.Parameter]:
        """Expose parameters of the wrapped model."""
        return self.model.parameters()

    def sample_actions(self, *args: Any, **kwargs: Any) -> Tensor:
        """Sample actions via the wrapped model."""
        return self.model.sample_actions(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward to the wrapped model."""
        return self.model.forward(*args, **kwargs)
