"""Tests for device placement of the ML batch types."""

import torch
from neuracore_types import BatchedJointData, BatchedRGBData, DataType

from neuracore.ml import BatchedInferenceInputs, BatchedTrainingSamples


def _rgb() -> BatchedRGBData:
    frame = torch.randint(0, 256, (2, 1, 3, 4, 5), dtype=torch.uint8)
    return BatchedRGBData(
        frame=frame,
        extrinsics=torch.zeros(2, 1, 4, 4),
        intrinsics=torch.zeros(2, 1, 3, 3),
    )


def test_training_samples_to_converts_rgb_frames_to_float32():
    """RGB frames reach the device as float32 with the same pixel values."""
    rgb = _rgb()
    joints = BatchedJointData.sample(batch_size=2, time_steps=1)
    batch = BatchedTrainingSamples(
        inputs={DataType.RGB_IMAGES: [rgb], DataType.JOINT_POSITIONS: [joints]},
        inputs_mask={DataType.RGB_IMAGES: torch.ones(2, 1)},
        outputs={DataType.RGB_IMAGES: [rgb]},
        outputs_mask={DataType.RGB_IMAGES: torch.ones(2, 1)},
        batch_size=2,
    )

    moved = batch.to(torch.device("cpu"))

    for frame in (
        moved.inputs[DataType.RGB_IMAGES][0].frame,
        moved.outputs[DataType.RGB_IMAGES][0].frame,
    ):
        assert frame.dtype == torch.float32
        assert torch.equal(frame, rgb.frame.to(torch.float32))
    assert torch.equal(moved.inputs[DataType.JOINT_POSITIONS][0].value, joints.value)


def test_inference_inputs_to_converts_rgb_frames_to_float32():
    """Inference RGB frames reach the device as float32 with the same values."""
    rgb = _rgb()
    batch = BatchedInferenceInputs(
        inputs={DataType.RGB_IMAGES: [rgb]},
        inputs_mask={DataType.RGB_IMAGES: torch.ones(2, 1)},
        batch_size=2,
    )

    moved = batch.to(torch.device("cpu"))

    frame = moved.inputs[DataType.RGB_IMAGES][0].frame
    assert frame.dtype == torch.float32
    assert torch.equal(frame, rgb.frame.to(torch.float32))
