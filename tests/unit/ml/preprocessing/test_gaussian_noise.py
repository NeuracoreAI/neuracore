import pytest
import torch
from neuracore_types import BatchedDepthData, BatchedRGBData

from neuracore.ml.preprocessing.methods.gaussian_noise import GaussianNoise


def _rgb(height: int = 32, width: int = 48) -> BatchedRGBData:
    return BatchedRGBData(
        frame=torch.full((2, 3, 3, height, width), 128.0, dtype=torch.float32),
        extrinsics=torch.zeros((2, 3, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((2, 3, 3, 3), dtype=torch.float32),
    )


def test_gaussian_noise_preserves_shape():
    data = _rgb()
    method = GaussianNoise(std=5.0)

    out = method(data)

    assert out.frame.shape == (2, 3, 3, 32, 48)
    assert out.frame.min() >= 0.0
    assert out.frame.max() <= 255.0


def test_gaussian_noise_changes_pixels():
    torch.manual_seed(0)
    data = _rgb()
    original = data.frame.clone()
    out = GaussianNoise(std=10.0)(data)

    assert not torch.allclose(out.frame, original)


def test_gaussian_noise_std_zero_is_noop():
    data = _rgb()
    original = data.frame.clone()
    out = GaussianNoise(std=0.0)(data)

    assert out is data
    assert torch.equal(out.frame, original)


def test_gaussian_noise_rejects_negative_std():
    with pytest.raises(ValueError, match="non-negative"):
        GaussianNoise(std=-1.0)


def test_gaussian_noise_rejects_non_rgb():
    depth = BatchedDepthData(
        frame=torch.ones((1, 1, 1, 16, 16), dtype=torch.float32),
        extrinsics=torch.zeros((1, 1, 4, 4), dtype=torch.float32),
        intrinsics=torch.zeros((1, 1, 3, 3), dtype=torch.float32),
    )
    with pytest.raises(TypeError, match="Unsupported batched data type"):
        GaussianNoise()(depth)
