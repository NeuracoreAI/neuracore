"""Tests for server utility helpers."""

import json

from neuracore_types import DataType, JointData, SynchronizedPoint

from neuracore.core.utils.server import _parse_embodiment_description


def test_parse_embodiment_description_restores_numeric_indices():
    """Parser should convert JSON string indices back to numeric ordering."""
    joint_names = [f"joint{i}" for i in range(12)]
    raw_description = json.dumps(
        {DataType.JOINT_POSITIONS.value: dict(enumerate(joint_names))}
    )

    embodiment_description = _parse_embodiment_description(raw_description)
    sync_point = SynchronizedPoint(
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=float(index))
                for index, name in enumerate(joint_names)
            }
        }
    )

    ordered = sync_point.order(embodiment_description)

    assert list(ordered.data[DataType.JOINT_POSITIONS]) == joint_names
