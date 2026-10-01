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


def test_assign_names_drops_extra_padded_tensors() -> None:
    first = _joint(0.1)
    second = _joint(0.2)
    extra = _joint(0.3)

    named = assign_names_to_batches(
        {DataType.JOINT_TARGET_POSITIONS: [first, second, extra]},
        {DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 1: "joint2"}},
    )

    assert list(named[DataType.JOINT_TARGET_POSITIONS].keys()) == ["joint1", "joint2"]
    assert named[DataType.JOINT_TARGET_POSITIONS]["joint1"] is first
    assert named[DataType.JOINT_TARGET_POSITIONS]["joint2"] is second
    assert extra not in named[DataType.JOINT_TARGET_POSITIONS].values()


def test_assign_names_raises_when_data_type_is_missing_from_description() -> None:
    with pytest.raises(
        ValueError,
        match="JOINT_POSITIONS not in embodiment description.",
    ):
        assign_names_to_batches(
            {DataType.JOINT_POSITIONS: [_joint(0.1)]},
            {DataType.JOINT_TARGET_POSITIONS: {0: "joint1"}},
        )


def test_assign_names_raises_when_tensor_list_is_shorter_than_highest_named_index() -> (
    None
):
    with pytest.raises(ValueError, match="Expected at least 3, but got 2"):
        assign_names_to_batches(
            {DataType.JOINT_TARGET_POSITIONS: [_joint(0.1), _joint(0.2)]},
            {DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 2: "joint3"}},
        )


def test_assign_names_keeps_sparse_indices_and_drops_unnamed_slots() -> None:
    at_index_0 = _joint(0.1)
    at_index_1 = _joint(0.2)
    at_index_2 = _joint(0.3)

    named = assign_names_to_batches(
        {DataType.JOINT_TARGET_POSITIONS: [at_index_0, at_index_1, at_index_2]},
        {DataType.JOINT_TARGET_POSITIONS: {0: "joint1", 2: "joint3"}},
    )

    assert named[DataType.JOINT_TARGET_POSITIONS]["joint1"] is at_index_0
    assert named[DataType.JOINT_TARGET_POSITIONS]["joint3"] is at_index_2
    assert at_index_1 not in named[DataType.JOINT_TARGET_POSITIONS].values()
