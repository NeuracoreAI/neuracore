"""Tests for mapping embodiment names onto tensor slots."""

import pytest
import torch
from neuracore_types import BatchedJointData, DataType

from neuracore.ml.utils.embodiment_names import assign_names_to_batches


def _joint(value: float) -> BatchedJointData:
    return BatchedJointData(value=torch.full((1, 1, 1), value))


def test_assign_names_drops_a_slot_whose_mask_is_zero() -> None:
    kept = _joint(0.1)
    dropped = _joint(0.2)

    named = assign_names_to_batches(
        {DataType.JOINT_POSITIONS: [kept, dropped]},
        {DataType.JOINT_POSITIONS: {0: "shoulder", 1: "elbow"}},
        masks={DataType.JOINT_POSITIONS: torch.tensor([1.0, 0.0])},
    )

    assert named[DataType.JOINT_POSITIONS] == {"shoulder": kept}


def test_assign_names_raises_when_a_named_index_is_outside_the_mask() -> None:
    with pytest.raises(IndexError, match="outside the JOINT_POSITIONS mask"):
        assign_names_to_batches(
            {DataType.JOINT_POSITIONS: [_joint(0.1), _joint(0.2)]},
            {DataType.JOINT_POSITIONS: {0: "shoulder", 1: "elbow"}},
            masks={DataType.JOINT_POSITIONS: torch.tensor([1.0])},
        )
