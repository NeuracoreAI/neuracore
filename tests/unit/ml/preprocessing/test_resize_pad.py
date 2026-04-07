import pytest
import torch
from neuracore_types import BatchedDepthData, BatchedRGBData

from neuracore.ml.preprocessing.methods.resize_pad import ResizePad


def _rgb(height: int, width: int) -> BatchedRGBData:
    return BatchedRGBData(
        frame=torch.ones((2, 3, 3, height, width), dtype=torch.float32),
        extrinsics=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((2, 3, 3, 3), dtype=torch.float32),
    )


def _depth(height: int, width: int) -> BatchedDepthData:
    return BatchedDepthData(
        frame=torch.ones((2, 3, 1, height, width), dtype=torch.float32),
        extrinsics=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((2, 3, 3, 3), dtype=torch.float32),
    )


def test_resize_pad_rgb_changes_shape_and_preserves_batch_time_channels():
    data = _rgb(80, 120)
    out = ResizePad(size=[200, 160])(data)

    assert out.frame.shape == (2, 3, 3, 200, 160)


def test_resize_pad_depth_changes_shape_and_preserves_batch_time_channels():
    data = _depth(90, 60)
    out = ResizePad(size=[128, 256])(data)

    assert out.frame.shape == (2, 3, 1, 128, 256)


def test_resize_pad_is_noop_when_shape_already_matches():
    data = _rgb(224, 224)
    out = ResizePad(size=[224, 224])(data)

    assert out is data
    assert out.frame.shape == (2, 3, 3, 224, 224)


def test_resize_pad_invalid_size_length_raises():
    data = _rgb(80, 120)
    with pytest.raises(ValueError, match="expects size as"):
        ResizePad(size=[224])(data)


def test_resize_pad_non_positive_sizes_raise():
    data = _depth(80, 120)
    with pytest.raises(ValueError, match="expects positive"):
        ResizePad(size=[0, 224])(data)
    with pytest.raises(ValueError, match="expects positive"):
        ResizePad(size=[224, -1])(data)


def test_resize_pad_rejects_unsupported_batched_type():
    class DummyBatched:
        def __init__(self):
            self.frame = torch.zeros((1, 1, 2, 20, 20), dtype=torch.float32)

    with pytest.raises(TypeError, match="Unsupported batched data type"):
        ResizePad(size=[32, 32])(DummyBatched())  # type: ignore[arg-type]
