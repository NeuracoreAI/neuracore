"""Auto-tuner for finding the optimal batch size for model training."""

import gc
import logging
import time
from typing import Any, Dict, Optional

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
        model_kwargs: Dict[str, Any],
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
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

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Autotuning batch size is only supported on GPUs.")
        self.model = model

        # create optimizers
        self.optimizers = self.model.configure_optimizers()

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

            # Clean up memory
            torch.cuda.empty_cache()
            gc.collect()

        # Reduce by 15% to be safe
        reduced_batch_size = int(optimal_batch_size * 0.70)
        logger.info(
            f"Optimal batch size found {optimal_batch_size}, "
            f"Reducing it by 30% to {reduced_batch_size}"
        )
        return reduced_batch_size

    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if a specific batch size works.

        Args:
            batch_size: Batch size to test

        Returns:
            True if the batch size works, False if it causes OOM error
        """
        logger.info(f"Testing batch size: {batch_size}")

        try:
            memory_monitor = MemoryMonitor(
                max_ram_utilization=0.8, max_gpu_utilization=1.0
            )

            # Create dataloader
            dataloader_kwargs = {**self.dataloader_kwargs, "batch_size": batch_size}
            data_loader = DataLoader(self.dataset, **dataloader_kwargs)

            # Get a batch that we can reuse
            batch: BatchedTrainingSamples = next(iter(data_loader))

            for i in range(self.num_iterations):

                memory_monitor.check_memory()

                if len(batch) < batch_size:
                    logger.info(f"Skipping batch size {batch_size} - not enough data")
                    return False

                # Convert batch to device
                batch = batch.to(self.device)

                # Forward pass
                self.model.train()

                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                start_time = time.time()
                outputs: BatchedTrainingOutputs = self.model.training_step(batch)
                loss = sum(outputs.losses.values()).mean()

                # Backward pass
                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()

                end_time = time.time()
                logger.info(
                    f"  Iteration {i+1}/{self.num_iterations} - "
                    f"Time: {end_time - start_time:.4f}s, "
                    f"Loss: {loss.item():.4f}"
                )

            logger.info(f"Batch size {batch_size} succeeded ✓")
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"Batch size {batch_size} failed due to OOM error ✗")
                # Clean up memory after OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            else:
                # Re-raise if it's not an OOM error
                raise

        except OutOfMemoryError:
            logger.info(f"Batch size {batch_size} failed due to RAM OOM error ✗")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def find_optimal_batch_size(
    dataset: Dataset,
    model: NeuracoreModel,
    model_kwargs: Dict[str, Any],
    dataloader_kwargs: Optional[Dict[str, Any]] = None,
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
