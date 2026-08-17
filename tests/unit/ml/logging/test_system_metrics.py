"""Tests for host and accelerator utilisation metrics."""

from unittest.mock import patch

import torch

from neuracore.ml.logging.system_metrics import (
    SYSTEM_METRIC_PREFIX,
    SystemMetricsCollector,
)


def test_reports_host_metrics_on_any_device():
    metrics = SystemMetricsCollector(torch.device("cpu")).collect()

    assert 0.0 <= metrics["cpu_utilization_percent"] <= 100.0
    assert 0.0 < metrics["ram_utilization_percent"] <= 100.0


def test_omits_accelerator_metrics_without_a_gpu():
    """A CPU or MPS run must simply not report GPU figures, not fail."""
    metrics = SystemMetricsCollector(torch.device("cpu")).collect()

    assert not any(name.startswith("gpu_") for name in metrics)


def test_disk_usage_is_reported_for_the_given_path(tmp_path):
    metrics = SystemMetricsCollector(torch.device("cpu"), disk_path=tmp_path).collect()

    assert 0.0 <= metrics["disk_utilization_percent"] <= 100.0


def test_disk_usage_is_omitted_when_the_path_is_unreadable(tmp_path):
    """A missing cache directory must not take the metrics down with it."""
    collector = SystemMetricsCollector(torch.device("cpu"), disk_path=tmp_path)

    with patch(
        "neuracore.ml.logging.system_metrics.shutil.disk_usage",
        side_effect=OSError("gone"),
    ):
        metrics = collector.collect()

    assert "disk_utilization_percent" not in metrics
    assert "cpu_utilization_percent" in metrics


def test_gpu_metrics_are_reported_when_cuda_is_present():
    collector = SystemMetricsCollector(torch.device("cpu"))
    # Stand in for a CUDA device without needing one.
    collector._collect_gpu = True
    collector._utilization_available = True
    collector._gpu_total_bytes = 8 * 1024**3

    with (
        patch("torch.cuda.memory_reserved", return_value=2 * 1024**3),
        patch("torch.cuda.utilization", return_value=73),
    ):
        metrics = collector.collect()

    assert metrics["gpu_memory_reserved_gb"] == 2.0
    assert metrics["gpu_memory_reserved_percent"] == 25.0
    assert metrics["gpu_utilization_percent"] == 73.0


def test_unavailable_nvml_is_tolerated_and_not_retried():
    """Utilisation needs driver access some containers withhold.

    Losing it should cost the one metric, not the run, and should not
    re-raise and re-log on every subsequent sample.
    """
    collector = SystemMetricsCollector(torch.device("cpu"))
    collector._collect_gpu = True
    collector._utilization_available = True
    collector._gpu_total_bytes = 0

    with (
        patch("torch.cuda.memory_reserved", return_value=0),
        patch("torch.cuda.utilization", side_effect=RuntimeError("no NVML")) as nvml,
    ):
        first = collector.collect()
        second = collector.collect()

    assert "gpu_utilization_percent" not in first
    assert "gpu_utilization_percent" not in second
    assert nvml.call_count == 1, "a failed NVML query should not be retried"
    # The memory figures still come through.
    assert "gpu_memory_reserved_gb" in second


def test_prefix_groups_separately_from_model_metrics():
    """These are machine metrics, so they must not land under train/ or val/."""
    assert SYSTEM_METRIC_PREFIX
    assert SYSTEM_METRIC_PREFIX not in ("train", "val")
    # _log_scalars joins with a slash, so carrying one here would double it.
    assert not SYSTEM_METRIC_PREFIX.endswith("/")
