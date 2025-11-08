"""Device allocation utils."""

from typing import Optional

import torch


def get_default_device(gpu_index: Optional[int] = None) -> torch.device:
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
