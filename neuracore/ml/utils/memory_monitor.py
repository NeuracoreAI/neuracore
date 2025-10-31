"""Monitor CPU RAM and GPU VRAM usage to prevent out of memory errors."""

import logging
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)


class OutOfMemoryError(Exception):
    """Custom exception for GPU or RAM out of memory."""

    def __init__(self, message: str, device: str = "unknown"):
        """Init.

        Args:
            message: Error message describing the out of memory condition
            device: Device where the error occurred (e.g., "CPU", "GPU:0")
        """
        super().__init__(f"Out of Memory on {device}: {message}")
        self.device = device


class MemoryMonitor:
    """Monitor CPU RAM and GPU VRAM usage."""

    def __init__(
        self,
        max_ram_utilization: float = 0.9,
        max_gpu_utilization: float = 0.9,
        gpu_id: Optional[int] = 0,
    ):
        """Init.

        Args:
            max_ram_utilization: Fraction of system RAM allowed (e.g., 0.9 = 90%)
            max_gpu_utilization: Fraction of GPU VRAM allowed
            gpu_id: Which GPU to monitor. None means no GPU monitoring.
        """
        self.max_ram_utilization = max_ram_utilization
        self.max_gpu_utilization = max_gpu_utilization
        self.gpu_id = gpu_id

    def check_memory(self, log: bool = False) -> None:
        """Raise OutOfMemoryError if memory is close to system limits.

        Args:
            log: Whether to log memory usage information
        Raises:
            OutOfMemoryError: If RAM or GPU memory usage exceeds limits
        """
        ram = psutil.virtual_memory()
        ram_used_ratio = ram.used / ram.total
        if log:
            ram_used_gb = ram.used / (1024**3)
            ram_total_gb = ram.total / (1024**3)
            logger.info(f"RAM Usage: {ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB")
        if ram_used_ratio > self.max_ram_utilization:
            raise OutOfMemoryError(
                f"RAM usage {ram_used_ratio*100:.1f}% exceeded limit! "
                "Consider using a smaller batch size or a more powerful machine."
            )

        if self.gpu_id is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024**3)
            gpu_mem_total = torch.cuda.get_device_properties(
                self.gpu_id
            ).total_memory / (1024**3)
            if log:
                logger.info(
                    f"GPU Reserved: {gpu_mem_reserved:.2f} GB / {gpu_mem_total:.2f} GB"
                )
            gpu_used_ratio = gpu_mem_reserved / gpu_mem_total
            if gpu_used_ratio > self.max_gpu_utilization:
                raise OutOfMemoryError(
                    f"GPU memory usage {gpu_used_ratio*100:.1f}% exceeded limit! "
                    "Consider using a smaller batch size or a more powerful machine."
                )
