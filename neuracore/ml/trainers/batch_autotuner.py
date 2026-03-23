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
        print(
            f"[DEBUG] BatchSizeAutotuner.find_optimal_batch_size: "
            f"range=[{self.min_batch_size}, {self.max_batch_size}], "
            f"device={self.device}, dataset_len={len(self.dataset)}, "
            f"num_iterations={self.num_iterations}"
        )

        # Binary search approach
        low = self.min_batch_size
        high = self.max_batch_size
        optimal_batch_size = low  # Start conservative
        search_step = 0

        while low <= high:
            mid = (low + high) // 2
            search_step += 1
            print(
                f"[DEBUG] Binary search step {search_step}: "
                f"low={low}, high={high}, testing mid={mid}"
            )
            success = self._test_batch_size(mid)

            if success:
                # This batch size works, try a larger one
                optimal_batch_size = mid
                low = mid + 1
                print(f"[DEBUG]   -> SUCCESS, optimal so far={optimal_batch_size}")
            else:
                # This batch size failed, try a smaller one
                high = mid - 1
                print("[DEBUG]   -> FAILED, reducing search range")

        # Reduce by 30% to be safe
        reduced_batch_size = int(optimal_batch_size * 0.70)
        msg = (
            f"Optimal batch size found {optimal_batch_size}, "
            f"Reducing it by 30% to {reduced_batch_size}"
        )
        logger.info(msg)
        print(f"[DEBUG] {msg}")

        logging.info(f"Testing the selected batch size {reduced_batch_size}")

        # Re-test the reduced size and, if it fails, keep shrinking until it fits.
        candidate = reduced_batch_size
        while candidate >= self.min_batch_size:
            print(f"[DEBUG] Re-testing reduced candidate={candidate}")
            if self._test_batch_size(candidate):
                print(f"[DEBUG] Final batch size: {candidate}")
                return candidate
            logger.info(
                "Reduced batch size %s failed on re-test; trying %s",
                candidate,
                candidate - 1,
            )
            print(
                f"[DEBUG] Reduced batch size {candidate} failed on re-test; "
                f"trying {candidate - 1}"
            )
            candidate -= 1

        print("[DEBUG] FATAL: Unable to find a valid batch size after safety reduction")
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
        print(f"[DEBUG] _test_batch_size({batch_size}): ENTERED")

        if not torch.cuda.is_available() or "cuda" not in self.device.type:
            raise ValueError("Batch size testing is only supported on GPUs.")

        base_state = None
        try:
            import psutil

            ram = psutil.virtual_memory()
            print(
                f"[DEBUG]   RAM before test: "
                f"{ram.used / (1024**3):.2f}/{ram.total / (1024**3):.2f} GB "
                f"({ram.percent}%)"
            )
            print(
                f"[DEBUG]   GPU mem before test: "
                f"alloc={torch.cuda.memory_allocated(self.device) / (1024**3):.4f} GB, "
                f"reserved={torch.cuda.memory_reserved(self.device) / (1024**3):.4f} GB"
            )

            memory_monitor = MemoryMonitor(
                max_ram_utilization=0.8, max_gpu_utilization=1.0
            )

            print("[DEBUG]   Snapshotting model weights...")
            base_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            print("[DEBUG]   Snapshot done")

            # Create dataloader
            dataloader_kwargs = {**self.dataloader_kwargs, "batch_size": batch_size}
            print(
                f"[DEBUG]   Creating DataLoader with kwargs: "
                f"batch_size={batch_size}, "
                f"other_keys={[k for k in dataloader_kwargs if k != 'batch_size']}"
            )
            data_loader = DataLoader(self.dataset, **dataloader_kwargs)

            # Get a batch that we can reuse
            print("[DEBUG]   Loading first batch from DataLoader...")
            batch: BatchedTrainingSamples = next(iter(data_loader))
            print(
                f"[DEBUG]   Batch loaded: batch_size={batch.batch_size}, "
                f"len={len(batch)}"
            )

            print(f"[DEBUG]   Moving batch to {self.device}...")
            batch = batch.to(self.device)
            print(
                f"[DEBUG]   Batch on device. "
                f"GPU allocated"
                f"={torch.cuda.memory_allocated(self.device) / (1024**3):.4f} GB"
            )

            # Fresh optimizers each trial
            print("[DEBUG]   Configuring optimizers...")
            optimizers = self.model.configure_optimizers()
            print(f"[DEBUG]   Optimizers configured: {len(optimizers)} optimizer(s)")

            # Track peak memory for this test.
            torch.cuda.reset_peak_memory_stats(self.device)

            for i in range(self.num_iterations):
                print(f"[DEBUG]   Iteration {i+1}/{self.num_iterations} starting...")

                memory_monitor.check_memory(log=True)

                if len(batch) < batch_size:
                    logger.info(f"Skipping batch size {batch_size} - not enough data")
                    print(
                        f"[DEBUG]   Batch too small"
                        f": {len(batch)} < {batch_size}, skipping"
                    )
                    return False

                # Forward pass
                self.model.train()

                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)

                torch.cuda.synchronize(self.device)
                start_time = time.time()
                print("[DEBUG]   Running forward pass (training_step)...")
                outputs: BatchedTrainingOutputs = self.model.training_step(batch)
                torch.cuda.synchronize(self.device)
                print(
                    f"[DEBUG]   Forward pass done. "
                    f"Losses: {list(outputs.losses.keys())}"
                )
                loss = sum(outputs.losses.values()).mean()
                torch.cuda.synchronize(self.device)

                # Backward pass
                print("[DEBUG]   Running backward pass...")
                loss.backward()
                torch.cuda.synchronize(self.device)
                print("[DEBUG]   Backward pass done")

                for optimizer in optimizers:
                    optimizer.step()
                torch.cuda.synchronize(self.device)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad = None

                end_time = time.time()
                iter_msg = (
                    f"  Iteration {i+1}/{self.num_iterations} - "
                    f"Time: {end_time - start_time:.4f}s, "
                    f"Loss: {loss.item():.4f}"
                )
                logger.info(iter_msg)
                print(f"[DEBUG] {iter_msg}")

                # Explicitly drop graph references to avoid lingering allocations.
                del outputs
                del loss

            torch.cuda.synchronize(self.device)
            peak_mem_bytes = torch.cuda.max_memory_allocated(self.device)
            self.last_peak_memory_gb = peak_mem_bytes / (1024**3)
            self.last_gpu_memory_gb = self.last_peak_memory_gb
            msg = (
                f"Batch size {batch_size} succeeded (peak GPU memory: "
                f"{self.last_peak_memory_gb:.2f} GB)"
            )
            logger.info(msg)
            print(f"[DEBUG]   {msg}")
            return True

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if (
                isinstance(e, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(e).lower()
            ):
                torch.cuda.synchronize(self.device)
                print(
                    f"[DEBUG]   Batch size {batch_size} CUDA/Runtime OOM: "
                    f"{type(e).__name__}: {e}"
                )
                logger.info(f"Batch size {batch_size} failed due to OOM error")
                return False
            print(
                f"[DEBUG]   Batch size {batch_size} RuntimeError (NOT OOM): "
                f"{type(e).__name__}: {e}"
            )
            logger.error(
                "[DEBUG] _test_batch_size RuntimeError (not OOM)", exc_info=True
            )
            raise

        except OutOfMemoryError as e:
            print(f"[DEBUG]   Batch size {batch_size} RAM OOM: {e}")
            logger.info(f"Batch size {batch_size} failed due to RAM OOM error")
            return False

        except Exception as e:
            print(
                f"[DEBUG]   Batch size {batch_size} UNEXPECTED error: "
                f"{type(e).__name__}: {e}"
            )
            logger.error("[DEBUG] _test_batch_size unexpected exception", exc_info=True)
            raise

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

            # Drop references and clean CUDA allocator
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
            if torch.cuda.is_available():
                print(
                    f"[DEBUG]   _test_batch_size({batch_size}) cleanup done. "
                    f"GPU allocated="
                    f"{torch.cuda.memory_allocated(self.device) / (1024**3):.4f} GB"
                )
            else:
                print(
                    f"[DEBUG]   _test_batch_size({batch_size}) cleanup done (no CUDA)"
                )


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
