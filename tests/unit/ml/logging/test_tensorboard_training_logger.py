"""Tests for TensorboardTrainingLogger histogram logging."""

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.tensorboard.summary import histogram

from neuracore.ml.logging.tensorboard_training_logger import TensorboardTrainingLogger

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
    ),
]


def assert_matches_add_histogram(call, values, bucket_edges):
    expected = histogram("expected", values, bucket_edges).value[0].histo
    assert call.kwargs["bucket_limits"] == list(expected.bucket_limit)
    assert call.kwargs["bucket_counts"] == list(expected.bucket)
    assert call.kwargs["num"] == expected.num
    assert call.kwargs["min"] == expected.min
    assert call.kwargs["max"] == expected.max
    scale = values.double().abs().sum().item()
    assert call.kwargs["sum"] == pytest.approx(
        expected.sum, abs=1e-5 * scale, nan_ok=True
    )
    if expected.sum_squares <= torch.finfo(torch.float32).max:
        assert call.kwargs["sum_squares"] == pytest.approx(
            expected.sum_squares, rel=1e-5
        )


@pytest.mark.parametrize("device", DEVICES)
def test_log_histogram_matches_add_histogram(tmp_path, device):
    torch.manual_seed(0)
    training_logger = TensorboardTrainingLogger(log_dir=tmp_path)
    bucket_edges = training_logger.writer.default_bins
    values = {
        "large": torch.randn(300_000),
        "float32": torch.randn(1000, 10),
        "bfloat16": torch.tensor([1e5, -3.0, 0.5]).bfloat16(),
        "zeros": torch.zeros(10),
        "inf": torch.tensor([1.0, float("inf"), -float("inf")]),
        "tiny": torch.tensor([1e-30, -1e-30, 3e-12, 0.0]),
    }

    with patch.object(
        training_logger.writer,
        "add_histogram_raw",
        wraps=training_logger.writer.add_histogram_raw,
    ) as add_histogram_raw:
        training_logger.log_histogram(
            "weights",
            {key: tensor.to(device) for key, tensor in values.items()},
            step=7,
        )
        training_logger.close()

    assert [call.args[0] for call in add_histogram_raw.call_args_list] == [
        f"weights/{key}" for key in values
    ]
    for call, tensor in zip(add_histogram_raw.call_args_list, values.values()):
        assert call.kwargs["global_step"] == 7
        assert_matches_add_histogram(call, tensor.float(), bucket_edges)


@pytest.mark.parametrize("device", DEVICES)
def test_log_histogram_bins_values_before_the_optimizer_step(tmp_path, device):
    layer = nn.Linear(64, 64).to(device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
    layer(torch.randn(4, 64, device=device)).sum().backward()
    weight_before_step = layer.weight.detach().cpu().clone()
    training_logger = TensorboardTrainingLogger(log_dir=tmp_path)

    with patch.object(training_logger.writer, "add_histogram_raw") as add_histogram_raw:
        training_logger.log_histogram("weights", {"layer": layer.weight}, step=0)
        optimizer.step()
        training_logger.close()

    assert_matches_add_histogram(
        add_histogram_raw.call_args,
        weight_before_step,
        training_logger.writer.default_bins,
    )


@pytest.mark.parametrize(
    "invalid_values, message",
    [
        (torch.empty(0), "no element"),
        (torch.full((4,), float("nan")), "histogram is empty"),
    ],
)
def test_failed_histogram_raises_once(tmp_path, invalid_values, message):
    training_logger = TensorboardTrainingLogger(log_dir=tmp_path)

    with patch.object(training_logger.writer, "add_histogram_raw") as add_histogram_raw:
        training_logger.log_histogram(
            "weights",
            {"invalid": invalid_values, "after": torch.ones(4)},
            step=0,
        )
        with pytest.raises(ValueError, match=message):
            training_logger.flush()
        training_logger.log_histogram("weights", {"next": torch.ones(4)}, step=1)
        training_logger.close()

    assert [call.args[0] for call in add_histogram_raw.call_args_list] == [
        "weights/next"
    ]
