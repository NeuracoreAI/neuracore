"""Host and accelerator utilisation metrics for training runs."""

import logging
import shutil
from pathlib import Path

import psutil
import torch

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3

# Scalars logged under this prefix are about the machine, not the model, and
# group separately from train/ and val/ in TensorBoard.
SYSTEM_METRIC_PREFIX = "system"


class SystemMetricsCollector:
    """Samples how hard the machine is working.

    Answers the question a loss curve cannot: whether a slow epoch is the model
    or the machine. A GPU sitting at low utilisation while CPU is pinned means
    the input pipeline is the bottleneck, not the network.

    Sampling costs a few syscalls plus one NVML query, so it is meant to be
    called every N steps rather than every step. See ``system_log_freq`` on the
    trainer.
    """

    def __init__(self, device: torch.device, disk_path: Path | None = None) -> None:
        """Initialize the collector.

        Args:
            device: Device being trained on. Accelerator metrics are collected
                only when this is a CUDA device.
            disk_path: Filesystem to report usage for. Defaults to the cache
                directory, which is what fills up during training.
        """
        self.device = device
        self.disk_path = disk_path
        self._collect_gpu = device.type == "cuda" and torch.cuda.is_available()
        self._utilization_available = self._collect_gpu
        self._gpu_total_bytes = 0
        if self._collect_gpu:
            self._gpu_total_bytes = torch.cuda.get_device_properties(
                device
            ).total_memory
        # cpu_percent reports the average since the previous call, so the first
        # one is always 0.0. Prime it here and let the first real sample cover
        # the interval up to it.
        psutil.cpu_percent(interval=None)

    def collect(self) -> dict[str, float]:
        """Return the current utilisation figures.

        Returns:
            Metric name to value. Names are unprefixed; the caller adds
            :data:`SYSTEM_METRIC_PREFIX`.
        """
        metrics: dict[str, float] = {
            # Average since the previous call, so this covers the interval
            # between samples rather than an instant.
            "cpu_utilization_percent": psutil.cpu_percent(interval=None),
            "ram_utilization_percent": psutil.virtual_memory().percent,
        }

        if self.disk_path is not None:
            try:
                usage = shutil.disk_usage(self.disk_path)
                metrics["disk_utilization_percent"] = (
                    usage.used / usage.total * 100 if usage.total else 0.0
                )
            except OSError:
                # The path may not exist yet on a fresh machine.
                pass

        if self._collect_gpu:
            reserved_bytes = torch.cuda.memory_reserved(self.device)
            metrics["gpu_memory_reserved_gb"] = reserved_bytes / BYTES_PER_GB
            if self._gpu_total_bytes:
                metrics["gpu_memory_reserved_percent"] = (
                    reserved_bytes / self._gpu_total_bytes * 100
                )
            utilization = self._gpu_utilization()
            if utilization is not None:
                metrics["gpu_utilization_percent"] = utilization

        return metrics

    def _gpu_utilization(self) -> float | None:
        """Percent of the sample period with a kernel running, via NVML.

        Returns None, and stops trying, if NVML is unavailable -- it needs
        driver access that some container configurations do not grant, and
        utilisation is a nice-to-have rather than something worth failing or
        log-spamming over.
        """
        if not self._utilization_available:
            return None
        try:
            return float(torch.cuda.utilization(self.device))
        except Exception:
            logger.warning(
                "GPU utilization unavailable; omitting it from system metrics.",
                exc_info=True,
            )
            self._utilization_available = False
            return None
