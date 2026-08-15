import pytest
import torch
from neuracore_types import BatchedDepthData, BatchedRGBData

from neuracore.ml.preprocessing.methods.color_jitter import ColorJitter


def _rgb(height: int = 32, width: int = 48) -> BatchedRGBData:
    return BatchedRGBData(
        frame=torch.full((2, 3, 3, height, width), 128.0, dtype=torch.float32),
        extrinsics=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((2, 3, 3, 3), dtype=torch.float32),
    )


def test_color_jitter_preserves_shape():
    data = _rgb()
    method = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    out = method(data)

    assert out.frame.shape == (2, 3, 3, 32, 48)
    assert out.frame.min() >= 0.0
    assert out.frame.max() <= 255.0


def test_color_jitter_returns_contiguous_frames():
    """Jitter must hand back the memory layout it was given.

    torchvision's hue adjustment permutes channels and returns a
    non-contiguous view. Collation used to mask that by re-materialising the
    tensor with torch.cat, but this runs on the device after collation now, and
    a non-contiguous frame propagates through the backbone to a Conv2d output
    that the image encoders then fail to flatten with .view.
    """
    torch.manual_seed(0)
    data = _rgb()
    assert data.frame.is_contiguous(), "fixture should start contiguous"

    out = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)(data)

    assert out.frame.is_contiguous()


def test_color_jitter_changes_pixels():
    torch.manual_seed(0)
    data = _rgb()
    original = data.frame.clone()
    out = ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1)(data)

    assert not torch.allclose(out.frame, original)


def test_color_jitter_rejects_non_rgb():
    depth = BatchedDepthData(
        frame=torch.ones((1, 1, 1, 16, 16), dtype=torch.float32),
        extrinsics=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, 1, 3, 3), dtype=torch.float32),
    )
    with pytest.raises(TypeError, match="Unsupported batched data type"):
        ColorJitter()(depth)
