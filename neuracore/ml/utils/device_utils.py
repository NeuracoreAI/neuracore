"""Device allocation utils."""

import logging

import psutil
import torch

logger = logging.getLogger(__name__)

MIXED_PRECISION_DTYPES = {"bf16": torch.bfloat16}


def get_default_device(gpu_index: int | None = None) -> torch.device:
    """Get the default device for PyTorch operations.

    Args:
        gpu_index: The index of the GPU to use (if available).

    Returns:
        The default torch.device object.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_index}" if gpu_index else "cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_count() -> int:
    """Return a positive CPU count even if psutil reports None."""
    return psutil.cpu_count() or 1


def resolve_autocast_dtype(
    mixed_precision: str | None, device: torch.device
) -> torch.dtype | None:
    """Resolve a mixed-precision setting to an autocast dtype.

    Returns None when the setting is off, or when the device cannot support it,
    so an unsupported request degrades to float32 training rather than failing.

    Args:
        mixed_precision: Requested precision, or None for float32.
        device: Device training will run on.

    Returns:
        The autocast dtype, or None to train in float32.

    Raises:
        ValueError: If mixed_precision names an unsupported precision.
    """
    if mixed_precision is None:
        return None
    if mixed_precision not in MIXED_PRECISION_DTYPES:
        raise ValueError(
            f"mixed_precision must be one of "
            f"{sorted(MIXED_PRECISION_DTYPES)} or None, got {mixed_precision!r}"
        )
    if device.type != "cuda":
        logger.warning(
            "mixed_precision=%s ignored: autocast needs CUDA, device is %s.",
            mixed_precision,
            device.type,
        )
        return None
    if not torch.cuda.is_bf16_supported():
        logger.warning(
            "mixed_precision=%s ignored: this GPU does not support bfloat16.",
            mixed_precision,
        )
        return None
    return MIXED_PRECISION_DTYPES[mixed_precision]
