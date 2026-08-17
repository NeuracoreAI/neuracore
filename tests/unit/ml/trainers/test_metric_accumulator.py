"""Tests for epoch metric accumulation."""

import torch

from neuracore.ml import BatchedTrainingOutputs
from neuracore.ml.trainers.metric_accumulator import MetricAccumulator


def test_averages_losses_and_metrics_over_the_epoch():
    accumulator = MetricAccumulator()
    for step in range(4):
        accumulator.update(
            BatchedTrainingOutputs(
                losses={"l1": torch.tensor(float(step)), "l2": torch.tensor(2.0)},
                metrics={"m": torch.tensor(float(step) * 3)},
            )
        )

    losses, metrics = accumulator.averages()

    assert losses == {"l1": 1.5, "l2": 2.0}
    assert metrics == {"m": 4.5}


def test_empty_accumulator_yields_nothing():
    assert MetricAccumulator().averages() == ({}, {})


def test_values_are_summed_without_leaving_the_device():
    """The point of this class is avoiding a device sync on every step.

    Reading a tensor back with .item() per loss per step is what it replaces,
    so the running totals must stay tensors until the epoch ends.
    """
    accumulator = MetricAccumulator()
    accumulator.update(
        BatchedTrainingOutputs(losses={"l": torch.tensor(1.0)}, metrics={})
    )
    accumulator.update(
        BatchedTrainingOutputs(losses={"l": torch.tensor(3.0)}, metrics={})
    )

    assert isinstance(accumulator._loss_sums["l"], torch.Tensor)
    assert accumulator.averages()[0] == {"l": 2.0}


def test_accumulated_tensors_are_detached():
    """Holding graph-attached tensors for a whole epoch would leak the graph."""
    accumulator = MetricAccumulator()
    weight = torch.tensor(2.0, requires_grad=True)

    accumulator.update(BatchedTrainingOutputs(losses={"l": weight * 3}, metrics={}))

    assert not accumulator._loss_sums["l"].requires_grad


def test_metrics_are_counted_independently_of_losses():
    """A model that reports losses but no metrics must still average cleanly."""
    accumulator = MetricAccumulator()
    accumulator.update(
        BatchedTrainingOutputs(losses={"l": torch.tensor(4.0)}, metrics={})
    )
    accumulator.update(
        BatchedTrainingOutputs(losses={"l": torch.tensor(2.0)}, metrics={})
    )

    losses, metrics = accumulator.averages()

    assert losses == {"l": 3.0}
    assert metrics == {}
