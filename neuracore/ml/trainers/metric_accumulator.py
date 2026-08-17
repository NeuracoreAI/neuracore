"""Accumulation of per-step losses and metrics over a training epoch."""

import torch

from neuracore.ml import BatchedTrainingOutputs


class MetricAccumulator:
    """Running sums of per-step losses and metrics for one epoch.

    Tensor values are summed on whichever device they arrive on and read back
    once, when the epoch ends. Summing on the host instead would mean a
    blocking device-to-host copy per loss and per metric on every single step,
    which serialises the training loop against the accelerator.
    """

    def __init__(self) -> None:
        """Initialize empty accumulators."""
        self._loss_sums: dict[str, torch.Tensor | float] = {}
        self._metric_sums: dict[str, torch.Tensor | float] = {}
        self._loss_steps = 0
        self._metric_steps = 0

    @staticmethod
    def _add(
        sums: dict[str, torch.Tensor | float], values: dict[str, torch.Tensor]
    ) -> None:
        """Fold one step's values into the running totals.

        Detaches first: these are held for a whole epoch, and keeping them
        attached would pin every step's autograd graph alive with them.
        """
        for key, value in values.items():
            contribution = value.detach() if isinstance(value, torch.Tensor) else value
            sums[key] = sums[key] + contribution if key in sums else contribution

    def update(self, batch_output: BatchedTrainingOutputs) -> None:
        """Add one step's losses and metrics to the running totals.

        Args:
            batch_output: Outputs from a single training or validation step.
        """
        if batch_output.losses:
            self._add(self._loss_sums, batch_output.losses)
            self._loss_steps += 1
        if batch_output.metrics:
            self._add(self._metric_sums, batch_output.metrics)
            self._metric_steps += 1

    @staticmethod
    def _finalize(
        sums: dict[str, torch.Tensor | float], steps: int
    ) -> dict[str, float]:
        """Divide the totals through and bring them back to the host."""
        if steps == 0:
            return {}
        return {
            key: (total.item() if isinstance(total, torch.Tensor) else float(total))
            / steps
            for key, total in sums.items()
        }

    def averages(self) -> tuple[dict[str, float], dict[str, float]]:
        """Return the epoch-averaged losses and metrics as plain floats.

        This is the only point at which values cross back from the device, so
        it belongs at an epoch boundary rather than inside the step loop.

        Returns:
            ``(losses, metrics)``, each empty if nothing was accumulated.
        """
        return (
            self._finalize(self._loss_sums, self._loss_steps),
            self._finalize(self._metric_sums, self._metric_steps),
        )
