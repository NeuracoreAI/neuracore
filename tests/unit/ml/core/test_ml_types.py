"""Tests for device placement of the ML batch types."""

import io

import numpy as np
import torch
from neuracore_types import BatchedDepthData, DataType, DepthCameraData
from torch.utils.data import DataLoader, Dataset

from neuracore.ml import BatchedTrainingSamples


class _DepthSamples(Dataset):
    """Depth samples from a 0.1 mm camera and a 1 mm camera."""

    frames = (
        np.array([[0, 2600], [65535, 1]], dtype=np.uint16),
        np.array([[0, 260], [6553, 1]], dtype=np.uint16),
    )
    scales = (1e-4, 1e-3)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> BatchedDepthData:
        return BatchedDepthData.from_nc_data(
            DepthCameraData(frame=self.frames[index], depth_scale_m=self.scales[index])
        )


def _collate(items: list[BatchedDepthData]) -> BatchedDepthData:
    return BatchedDepthData(
        frame=torch.cat([item.frame for item in items]),
        depth_scale=torch.cat([item.depth_scale for item in items]),
        extrinsics=torch.cat([item.extrinsics for item in items]),
        intrinsics=torch.cat([item.intrinsics for item in items]),
    )


def test_depth_batch_crosses_workers_and_cache_as_uint16_then_metres():
    """uint16 depth crosses workers and the sample cache, then becomes metres."""
    loader = DataLoader(
        _DepthSamples(), batch_size=2, num_workers=2, collate_fn=_collate
    )
    depth = next(iter(loader))
    assert depth.frame.dtype == torch.uint16
    buffer = io.BytesIO()
    torch.save(depth, buffer)
    buffer.seek(0)
    depth = torch.load(buffer, weights_only=False)
    batch = BatchedTrainingSamples(
        inputs={DataType.DEPTH_IMAGES: [depth]},
        inputs_mask={DataType.DEPTH_IMAGES: torch.ones(2, 1)},
        outputs={},
        outputs_mask={},
        batch_size=2,
    )

    metres = batch.to(torch.device("cpu")).inputs[DataType.DEPTH_IMAGES][0].frame

    assert metres.dtype == torch.float32
    for index, (frame, scale) in enumerate(
        zip(_DepthSamples.frames, _DepthSamples.scales, strict=True)
    ):
        expected = torch.from_numpy(frame.astype(np.float32)) * torch.tensor(
            scale, dtype=torch.float32
        )
        assert torch.equal(metres[index, 0, 0], expected)
    assert torch.equal(metres[:, 0, 0, 0, 0], torch.zeros(2))
