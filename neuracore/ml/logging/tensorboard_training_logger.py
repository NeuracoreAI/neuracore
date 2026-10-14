"""TensorBoard-based training logger for local training.

This module provides a unified logging interface similar to Weights & Biases
but using TensorBoard for visualization.
"""

import json
import logging
import math
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from neuracore.ml.logging.training_logger import TrainingLogger

logger = logging.getLogger(__name__)


SMALL_TENSOR_NUMEL = 256 * 1024
GROUP_NUMEL = 2 * 2**20
GROUP_MAX_TENSORS = 4096


def compute_histogram_counts_and_stats(
    tensors: list[torch.Tensor], bucket_edges: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a histogram of each tensor and compute its summary statistics.

    A histogram splits the value range into buckets and records how many values
    fall in each bucket. Bucket i holds the values from bucket_edges[i] up to, but
    not including, bucket_edges[i + 1]. All work runs on the device of the tensors
    and never waits for the device, so the training loop keeps running.

    Args:
        tensors: Tensors to summarize, all on one device.
        bucket_edges: Ascending bucket edges, as in SummaryWriter.default_bins.

    Returns:
        A tuple of two tensors on the device of the input. The first holds the
        int64 count of values per bucket, with shape (len(tensors), num_buckets).
        The second holds the float32 min, max, sum and sum of squares of each
        tensor, in that order, with shape (len(tensors), 4). Values below the first
        edge, values at or above the last edge, and NaN fall in no bucket.
    """
    device = tensors[0].device
    num_buckets = len(bucket_edges) - 1
    slots = num_buckets + 2

    # Place the bucket edges on the device as float32.
    edges = torch.tensor(bucket_edges, dtype=torch.float32).to(
        device, non_blocking=True
    )

    # Flatten each tensor to a 1-D view.
    flat = [tensor.detach().reshape(-1) for tensor in tensors]

    # Group consecutive small tensors, and give each large tensor its own group.
    groups: list[list[int]] = []
    current: list[int] = []
    current_numel = 0
    for index, values in enumerate(flat):
        if values.numel() >= SMALL_TENSOR_NUMEL:
            groups.append([index])
            continue
        current.append(index)
        current_numel += values.numel()
        if current_numel >= GROUP_NUMEL or len(current) == GROUP_MAX_TENSORS:
            groups.append(current)
            current, current_numel = [], 0
    if current:
        groups.append(current)

    # Upload the tensor sizes in group order and the input order in one copy.
    grouped_order = [i for indices in groups for i in indices]
    sizes_and_order = torch.cat([
        torch.tensor([flat[i].numel() for i in grouped_order]),
        torch.argsort(torch.tensor(grouped_order)),
    ]).to(device, non_blocking=True)
    sizes, order = sizes_and_order.chunk(2)

    counts, stats = [], []
    offset = 0
    for indices in groups:
        num_tensors = len(indices)
        is_large = flat[indices[0]].numel() >= SMALL_TENSOR_NUMEL

        # Join the group's values as float32 and map each value to its tensor.
        if is_large:
            values = flat[indices[0]].float()
        else:
            values = torch.cat([flat[i].float() for i in indices])
            owner = torch.repeat_interleave(
                torch.arange(num_tensors, device=device),
                sizes[offset : offset + num_tensors],
                output_size=values.numel(),
            )
        offset += num_tensors

        # Find each value's slot: 0 below the edges, 1 to num_buckets inside, and
        # num_buckets + 1 above the edges or NaN.
        keys = torch.bucketize(values, edges, right=True)
        if not is_large:
            keys = owner * slots + keys

        # Count the values per tensor and slot, then keep the bucket slots.
        group_counts = torch.zeros(
            num_tensors * slots, dtype=torch.int64, device=device
        )
        group_counts.scatter_add_(
            0, keys, torch.ones(1, dtype=torch.int64, device=device).expand_as(keys)
        )
        counts.append(group_counts.view(num_tensors, slots)[:, 1 : num_buckets + 1])

        # Reduce min, max, sum and sum of squares per tensor.
        if is_large:
            low, high = torch.aminmax(values)
            group_stats = torch.stack(
                [low, high, values.sum(), torch.dot(values, values)]
            )[None]
        else:
            group_stats = torch.zeros(num_tensors, 4, device=device)
            group_stats[:, 0] = math.inf
            group_stats[:, 1] = -math.inf
            group_stats[:, 0].scatter_reduce_(0, owner, values, "amin")
            group_stats[:, 1].scatter_reduce_(0, owner, values, "amax")
            group_stats[:, 2].index_add_(0, owner, values)
            group_stats[:, 3].index_add_(0, owner, values * values)
        stats.append(group_stats)

    # Restore the input order.
    return torch.cat(counts)[order], torch.cat(stats)[order]


class HistogramsPendingCopy(NamedTuple):
    """Histograms of several tensors that the background writer logs.

    The counts and stats may still be copying to the host. The writer waits for
    the copied event before it reads them.

    Attributes:
        names: Histogram name of each tensor.
        numels: Number of values in each tensor.
        counts: Bucket counts, one row per tensor.
        stats: Min, max, sum and sum of squares, one row per tensor.
        step: Training step of the histograms.
        walltime: Creation time of the histograms, in seconds since the epoch.
        copied: CUDA event that completes when counts and stats reach the host,
            or None when they already live on the CPU.
    """

    names: list[str]
    numels: list[int]
    counts: torch.Tensor
    stats: torch.Tensor
    step: int
    walltime: float
    copied: torch.cuda.Event | None


class TensorboardTrainingLogger(TrainingLogger):
    """TensorBoard-based logger for local training."""

    def __init__(
        self,
        log_dir: Path,
        run_name: str | None = None,
        sync_interval: int = 60,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save logs. If None, creates temp directory.
            run_name: Name of the run. Defaults to a timestamp-based name.
            sync_interval: Interval in seconds for cloud sync (if enabled).
        """
        self.sync_interval = sync_interval
        self.run_name = run_name or f"run_{int(time.time())}"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self._histogram_writer = ThreadPoolExecutor(max_workers=1)
        self._histogram_writes: list[Future[None]] = []
        logger.info(f"TensorBoard logger initialized: {self.log_dir}")

    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            name: Name of the metric (e.g., "train/loss", "val/accuracy").
            value: Scalar value to log.
            step: Training step.
        """
        self.writer.add_scalar(name, value, step)

    def log_image(
        self,
        name: str,
        image: np.ndarray | torch.Tensor,
        step: int,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image.

        Args:
            name: Name for the image.
            image: Image data as numpy array or torch tensor.
            step: Training step.
            dataformats: Format of the image data (e.g., "CHW", "HWC").
        """
        self.writer.add_image(name, image, step, dataformats=dataformats)

    def log_images(
        self,
        name: str,
        images: np.ndarray | torch.Tensor,
        step: int,
        dataformats: str = "NCHW",
    ) -> None:
        """Log multiple images as a grid.

        Args:
            name: Name for the image grid.
            images: Batch of images as numpy array or torch tensor.
            step: Training step.
            dataformats: Format of the image data (e.g., "NCHW").
        """
        self.writer.add_images(name, images, step, dataformats=dataformats)

    def log_histogram(
        self,
        name: str,
        values: np.ndarray | torch.Tensor | Mapping[str, torch.Tensor],
        step: int,
    ) -> None:
        """Log a histogram of values, or one histogram per tensor of a mapping.

        A mapping logs the histogram of each tensor under the name "<name>/<key>"
        without slowing the training loop. Compute the bucket counts and
        statistics on the device of the tensors, start copying them to the host,
        and write them to TensorBoard on a background thread. Raise a failed write
        on the next call once the write has finished, and on flush or close.

        Args:
            name: Name for the histogram, or the prefix of the histogram names when
                values is a mapping.
            values: Values to create the histogram from, as numpy array or torch
                tensor, or a mapping from histogram name suffix to tensor.
            step: Training step.
        """
        if not isinstance(values, Mapping):
            self.writer.add_histogram(name, values, step)
            return
        self._raise_histogram_write_errors()
        if not values:
            return
        tensors = list(values.values())
        counts, stats = compute_histogram_counts_and_stats(
            tensors, self.writer.default_bins
        )
        copied = None
        if counts.is_cuda:
            counts = counts.to("cpu", non_blocking=True)
            stats = stats.to("cpu", non_blocking=True)
            copied = torch.cuda.Event()
            copied.record()
        pending = HistogramsPendingCopy(
            names=[f"{name}/{key}" for key in values],
            numels=[tensor.numel() for tensor in tensors],
            counts=counts,
            stats=stats,
            step=step,
            walltime=time.time(),
            copied=copied,
        )
        self._histogram_writes.append(
            self._histogram_writer.submit(self._write_histograms, pending)
        )

    def _raise_histogram_write_errors(self, wait: bool = False) -> None:
        """Raise the error of a failed background histogram write.

        Args:
            wait: Wait for unfinished writes and check them too. Otherwise skip
                the unfinished writes.
        """
        writes, self._histogram_writes = self._histogram_writes, []
        for write in writes:
            if wait or write.done():
                write.result()
            else:
                self._histogram_writes.append(write)

    def _write_histograms(self, pending: HistogramsPendingCopy) -> None:
        """Write histograms to TensorBoard the way SummaryWriter.add_histogram does.

        Wait until the counts and stats reach the host, drop the empty buckets at
        both ends as add_histogram does, and log each histogram with
        add_histogram_raw.

        Args:
            pending: Counts and statistics to write.

        Raises:
            ValueError: A tensor has no values, or no value fell in a bucket.
        """
        if pending.copied is not None:
            pending.copied.synchronize()
        bucket_edges = np.asarray(self.writer.default_bins)
        for name, numel, counts, (low, high, total, squares) in zip(
            pending.names,
            pending.numels,
            pending.counts.numpy(),
            pending.stats.tolist(),
        ):
            if numel == 0:
                raise ValueError("The input has no element.")
            cum_counts = np.cumsum(np.greater(counts, 0))
            start, end = np.searchsorted(
                cum_counts, [0, cum_counts[-1] - 1], side="right"
            )
            start, end = int(start), int(end) + 1
            counts = (
                counts[start - 1 : end]
                if start > 0
                else np.concatenate([[0], counts[:end]])
            )
            limits = bucket_edges[start : end + 1]
            if counts.size == 0 or limits.size == 0:
                raise ValueError("The histogram is empty, please file a bug report.")
            self.writer.add_histogram_raw(
                name,
                min=low,
                max=high,
                num=numel,
                sum=total,
                sum_squares=squares,
                bucket_limits=limits.tolist(),
                bucket_counts=counts.tolist(),
                global_step=pending.step,
                walltime=pending.walltime,
            )

    def log_text(self, name: str, text: str, step: int) -> None:
        """Log text data.

        Args:
            name: Name for the text data.
            text: Text string to log.
            step: Training step.
        """
        self.writer.add_text(name, text, step)

    def log_hyperparameters(
        self,
        hparams: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log hyperparameters and optionally metrics.

        Args:
            hparams: Dictionary of hyperparameters.
            metrics: Optional dictionary of metrics to associate with hparams.
        """
        # Convert any non-primitive types to strings for TensorBoard compatibility
        clean_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                clean_hparams[key] = value
            else:
                clean_hparams[key] = str(value)

        self.writer.add_hparams(clean_hparams, metrics or {})

        # Also save as JSON for easier access
        config_file = self.log_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(hparams, f, indent=2, default=str)

    def log_model_graph(
        self, model: torch.nn.Module, input_to_model: torch.Tensor | None = None
    ) -> None:
        """Log the model computational graph.

        Args:
            model: PyTorch model to log.
            input_to_model: Example input tensor for the model.
        """
        self.writer.add_graph(model, input_to_model)

    def flush(self) -> None:
        """Force flush any pending writes to disk."""
        self._raise_histogram_write_errors(wait=True)
        self.writer.flush()

    def close(self) -> None:
        """Close the logger and clean up resources."""
        try:
            self._raise_histogram_write_errors(wait=True)
        finally:
            self._histogram_writer.shutdown()
            self.writer.close()

    def get_log_dir(self) -> str:
        """Get the log directory path."""
        return str(self.log_dir)
