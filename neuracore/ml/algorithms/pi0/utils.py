"""Utility functions and configuration for the PI0 algorithm.

This module provides helper functions for flow matching, attention mask
construction, and image preprocessing used by the PI0 model. It also
defines the PI0Config dataclass for model configuration.
"""

# cspell:ignore OPENPI adarms

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

# Constant used for attention masking in the OPENPI implementation
OPENPI_ATTENTION_MASK_VALUE = -1e9


@dataclass(slots=True)
class PI0Config:
    """Configuration for the PI0 model and training hyperparameters.

    Attributes:
        paligemma_variant: PaliGemma model size ("gemma_300m" or "gemma_2b").
        action_expert_variant: Action expert model size ("gemma_300m" or "gemma_2b").
        dtype: Model precision ("bfloat16" or "float32").
        chunk_size: Number of action steps predicted per inference.
        max_state_dim: Maximum dimension for state input vectors.
        max_action_dim: Maximum dimension for action output vectors.
        num_inference_steps: Number of Euler steps for action denoising.
        use_adarms: Whether to use adaptive RMSNorm for (VLM, action expert).
        time_sampling_beta_alpha: Alpha parameter for beta distribution time sampling.
        time_sampling_beta_beta: Beta parameter for beta distribution time sampling.
        time_sampling_scale: Scale factor for sampled time values.
        time_sampling_offset: Offset added to sampled time values.
        min_period: Minimum period for sinusoidal time embeddings.
        max_period: Maximum period for sinusoidal time embeddings.
        gradient_checkpointing: Whether to enable gradient checkpointing.
        compile_model: Whether to compile the model with torch.compile.
        compile_mode: Compilation mode for torch.compile.
        device: Device to place the model on.
        input_features: Mapping of input feature names to dimensions.
        output_features: Mapping of output feature names to dimensions.
        image_features: List of image feature names used as input.
    """

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
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.device is None:
            self.device = "cpu"

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(
                f"Invalid action_expert_variant: {self.action_expert_variant}"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")


def _get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a device-compatible dtype.

    Some devices don't support certain dtypes (e.g., MPS doesn't support
    float64, CPU doesn't efficiently support bfloat16). This function
    returns a safe fallback dtype.

    Args:
        target_dtype: Desired dtype
        device_type: Device type string ("cpu", "mps", "cuda")

    Returns:
        Compatible dtype for the given device.
    """
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def _create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Create sinusoidal positional embeddings for diffusion timesteps.

    Uses logarithmically-spaced frequencies between min_period and max_period
    to create rich time representations for the flow matching model.

    Args:
        time: Diffusion timesteps [batch_size]
        dimension: Embedding dimension (must be even)
        min_period: Minimum frequency period
        max_period: Maximum frequency period
        device: Target device

    Returns:
        Sinusoidal embeddings [batch_size, dimension].

    Raises:
        ValueError: If dimension is odd or time tensor has wrong shape.
    """
    device = torch.device(device)
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = _get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def _sample_beta(
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    bsize: int,
    device: torch.device | str,
) -> Tensor:
    """Sample from beta distribution for time sampling.

    Args:
        alpha: Beta distribution alpha parameter
        beta: Beta distribution beta parameter
        bsize: Number of samples to draw
        device: Target device

    Returns:
        Beta-distributed samples [bsize].
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def _make_att_2d_masks(
    pad_masks: torch.Tensor, att_masks: torch.Tensor
) -> torch.Tensor:
    """Build causal 2D attention masks from padding and attention masks.

    Combines padding information with causal masking to create the final
    attention mask used by transformer layers.

    Args:
        pad_masks: Padding mask [batch_size, seq_len]
        att_masks: Attention mask [batch_size, seq_len]

    Returns:
        Combined causal mask [batch_size, seq_len, seq_len].

    Raises:
        ValueError: If input masks don't have 2 dimensions.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Right-pad tensor's last dimension to target size.

    Args:
        vector: Input tensor
        new_dim: Target size for last dimension

    Returns:
        Padded tensor, or original if already large enough.
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def build_lr_lambda(
    actual_warmup_steps: int,
    actual_decay_steps: int,
    decay_lr: float,
    optimizer_lr: float,
) -> Callable[[int], float]:
    """Create a learning rate scheduler lambda with warmup and cosine decay.

    Args:
        actual_warmup_steps: Warmup steps after any scaling.
        actual_decay_steps: Cosine decay steps after any scaling.
        decay_lr: Final learning rate after decay.
        optimizer_lr: Base optimizer learning rate.

    Returns:
        Callable that maps the current step to a LR multiplier.
    """

    def linear_warmup(step: int) -> float:
        if step <= 0:
            return 1 / (actual_warmup_steps + 1)
        frac = 1 - step / actual_warmup_steps
        return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

    def cosine_decay(step: int) -> float:
        step = min(step, actual_decay_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
        alpha = decay_lr / optimizer_lr
        return (1 - alpha) * cosine + alpha

    def lr_lambda(current_step: int) -> float:
        if current_step < actual_warmup_steps:
            return linear_warmup(current_step)
        return cosine_decay(current_step)

    return lr_lambda


def _align_mask_length(mask_1d: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad or trim a 1D mask to target length.

    Args:
        mask_1d: Input mask tensor
        target_len: Desired length

    Returns:
        Mask tensor with exactly target_len elements.
    """
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
    """Resize images to target size while preserving aspect ratio.

    Resizes the image to fit within the target dimensions while maintaining
    aspect ratio, then pads to reach exact target size. Automatically
    detects channels-first vs channels-last format.

    Args:
        images: Input images (B, C, H, W) or (B, H, W, C)
        height: Target height
        width: Target width
        mode: Interpolation mode

    Returns:
        Resized and padded images in original format.

    Raises:
        ValueError: If image dtype is not uint8 or float32.
    """
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
