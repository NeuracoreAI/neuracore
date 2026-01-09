"""Auto-tuner for finding the optimal batch size for model training."""

import gc
import logging
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from neuracore.ml import BatchedTrainingOutputs, BatchedTrainingSamples, NeuracoreModel
from neuracore.ml.utils.memory_monitor import MemoryMonitor, OutOfMemoryError

logger = logging.getLogger(__name__)


class BatchSizeAutotuner:
    """Auto-tuner for finding the optimal batch size for model training."""

    def __init__(
        self,
        dataset: Dataset,
        model: NeuracoreModel,
        model_kwargs: dict[str, Any],
        dataloader_kwargs: dict[str, Any] | None = None,
        min_batch_size: int = 8,
        max_batch_size: int = 512,
        num_iterations: int = 3,
    ):
        """Initialize the batch size auto-tuner.

        Args:
            dataset: Dataset to use for testing
            model: Model to use for testing
            model_kwargs: Arguments to pass to model constructor
            dataloader_kwargs: Additional arguments for the DataLoader
            min_batch_size: Minimum batch size to try
            max_batch_size: Maximum batch size to try
            num_iterations: Number of iterations to run for each batch size
        """
        self.dataset = dataset
        self.model_kwargs = model_kwargs
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.num_iterations = num_iterations
        self.device = model.device
        self.last_peak_memory_gb: float | None = None
        self.last_gpu_memory_gb: float | None = None

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Autotuning batch size is only supported on GPUs.")
        self.model = model

        # Validate batch size ranges
        if min_batch_size > max_batch_size:
            raise ValueError(
                f"min_batch_size ({min_batch_size}) must be "
                f"<= max_batch_size ({max_batch_size})"
            )

        # Validate dataset size
        if len(dataset) < min_batch_size:
            raise ValueError(
                f"Dataset size ({len(dataset)}) is smaller "
                f"than min_batch_size ({min_batch_size})"
            )

    def find_optimal_batch_size(self) -> int:
        """Find the optimal batch size using binary search.

        Returns:
            The optimal batch size
        """
        logger.info(
            "Finding optimal batch size between "
            f"{self.min_batch_size} and {self.max_batch_size}"
        )

        # Binary search approach
        low = self.min_batch_size
        high = self.max_batch_size
        optimal_batch_size = low  # Start conservative

        while low <= high:
            mid = (low + high) // 2
            success = self._test_batch_size(mid)

            if success:
                # This batch size works, try a larger one
                optimal_batch_size = mid
                low = mid + 1
            else:
                # This batch size failed, try a smaller one
                high = mid - 1

        # Reduce by 30% to be safe
        reduced_batch_size = int(optimal_batch_size * 0.70)
        logger.info(
            f"Optimal batch size found {optimal_batch_size}, "
            f"Reducing it by 30% to {reduced_batch_size}"
        )

        logging.info(f"Testing the selected batch size {reduced_batch_size}")

        # Re-test the reduced size and, if it fails, keep shrinking until it fits.
        candidate = reduced_batch_size
        while candidate >= self.min_batch_size:
            if self._test_batch_size(candidate):
                return candidate
            logger.info(
                "Reduced batch size %s failed on re-test; trying %s",
                candidate,
                candidate - 1,
            )
            candidate -= 1

        raise OutOfMemoryError(
            "Unable to find a valid batch size after safety reduction.",
            device=str(self.device),
        )

    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if a specific batch size works.

        Args:
            batch_size: Batch size to test

        Returns:
            True if the batch size works, False if it causes OOM error
        """
        logger.info(f"Testing batch size: {batch_size}")

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Batch size testing is only supported on GPUs.")

        base_state = None
        try:
            memory_monitor = MemoryMonitor(
                max_ram_utilization=0.8, max_gpu_utilization=1.0
            )

            # Snapshot model weights so optimizer steps during tuning don't
            # leak into training.
            base_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }

            # Create dataloader
            dataloader_kwargs = {**self.dataloader_kwargs, "batch_size": batch_size}
            data_loader = DataLoader(self.dataset, **dataloader_kwargs)

            # Get a batch that we can reuse
            batch: BatchedTrainingSamples = next(iter(data_loader))
            batch = batch.to(self.device)

            # Fresh optimizers each trial so optimizer state is
            # representative but discarded.
            optimizers = self.model.configure_optimizers()

            # Track peak memory for this test.
            torch.cuda.reset_peak_memory_stats(self.device)

            for i in range(self.num_iterations):

                memory_monitor.check_memory(log=True)

                if len(batch) < batch_size:
                    logger.info(f"Skipping batch size {batch_size} - not enough data")
                    return False

                # Forward pass
                self.model.train()

                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize(self.device)
                start_time = time.time()
                outputs: BatchedTrainingOutputs = self.model.training_step(batch)
                torch.cuda.synchronize(self.device)
                loss = sum(outputs.losses.values()).mean()
                torch.cuda.synchronize(self.device)

                # Backward pass
                loss.backward()
                torch.cuda.synchronize(self.device)

                # Keep optimizer.step() to reflect real training peak memory,
                # but optimizers are recreated per test and model weights
                # restored after.
                for optimizer in optimizers:
                    optimizer.step()
                torch.cuda.synchronize(self.device)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = None

                end_time = time.time()
                logger.info(
                    f"  Iteration {i+1}/{self.num_iterations} - "
                    f"Time: {end_time - start_time:.4f}s, "
                    f"Loss: {loss.item():.4f}"
                )

                # Explicitly drop graph references to avoid lingering allocations.
                del outputs
                del loss

            torch.cuda.synchronize(self.device)
            peak_mem_bytes = torch.cuda.max_memory_allocated(self.device)
            self.last_peak_memory_gb = peak_mem_bytes / (1024**3)
            self.last_gpu_memory_gb = self.last_peak_memory_gb
            logger.info(
                "Batch size %s succeeded ✓ (peak GPU memory: %.2f GB)",
                batch_size,
                self.last_peak_memory_gb,
            )
            return True

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(e).lower()
            ):
                torch.cuda.synchronize(self.device)
                logger.info(f"Batch size {batch_size} failed due to OOM error ✗")
                return False
            raise

        except OutOfMemoryError:
            logger.info(f"Batch size {batch_size} failed due to RAM OOM error ✗")
            return False

        finally:
            # Restore model weights so tuning does not alter training.
            if base_state is not None:
                try:
                    self.model.load_state_dict(base_state)
                except Exception:
                    logger.exception(
                        "Failed to restore model weights after tuning trial."
                    )
            self.model.zero_grad(set_to_none=True)

            # Drop references and clean CUDA allocator to reduce
            # fragmentation between tests.
            try:
                del batch
            except Exception:
                pass
            try:
                del optimizers
            except Exception:
                pass

            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def find_optimal_batch_size(
    dataset: Dataset,
    model: NeuracoreModel,
    model_kwargs: dict[str, Any],
    dataloader_kwargs: dict[str, Any] | None = None,
    min_batch_size: int = 8,
    max_batch_size: int = 512,
) -> int:
    """Find the optimal batch size for a given model and dataset.

    Args:
        dataset: Dataset to use for testing
        model: Model to use for testing
        model_kwargs: Arguments to pass to model constructor
        dataloader_kwargs: Additional arguments for the DataLoader
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try

    Returns:
        The optimal batch size
    """
    autotuner = BatchSizeAutotuner(
        dataset=dataset,
        model=model,
        model_kwargs=model_kwargs,
        dataloader_kwargs=dataloader_kwargs,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
    )

    return autotuner.find_optimal_batch_size()
