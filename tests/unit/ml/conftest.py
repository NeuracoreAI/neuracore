"""Shared fixtures for the ML unit tests.

Multi-node (peer-to-peer) prediction: a robot instance can be driven by several
processes at once — one node logs joints, another logs camera frames, and a
third runs the policy. ``get_latest_sync_point`` merges the local process's
streams with what the remote nodes provide, and ``Policy.predict(sync_point=None)``
calls it. The helpers below stub that remote-node seam so the merge-into-predict
path can be exercised without a real WebRTC connection.
"""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
from neuracore_types import (
    DataType,
    JointData,
    NCData,
    ParallelGripperOpenAmountData,
    RGBCameraData,
    SynchronizedPoint,
)

REMOTE_TIMESTAMP = 1234567890.0
REMOTE_JOINT_NAMES = ("joint1", "joint2")
REMOTE_CAMERA_NAME = "top_camera"
REMOTE_GRIPPER_NAMES = ("left_arm", "right_arm")
REMOTE_FRAME = np.full((8, 8, 3), 7, dtype=np.uint8)

RemoteData = dict[DataType, dict[str, NCData]]


def remote_joint_data() -> RemoteData:
    """The payload a joints-only remote node contributes."""
    return {
        DataType.JOINT_POSITIONS: {
            name: JointData(timestamp=REMOTE_TIMESTAMP, value=0.5)
            for name in REMOTE_JOINT_NAMES
        }
    }


def remote_camera_data() -> RemoteData:
    """The payload a camera-only remote node contributes."""
    return {
        DataType.RGB_IMAGES: {
            REMOTE_CAMERA_NAME: RGBCameraData(
                timestamp=REMOTE_TIMESTAMP, frame=REMOTE_FRAME
            )
        }
    }


def remote_gripper_data() -> RemoteData:
    """The payload a gripper-only remote node contributes."""
    return {
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
            name: ParallelGripperOpenAmountData(
                timestamp=REMOTE_TIMESTAMP, open_amount=0.5
            )
            for name in REMOTE_GRIPPER_NAMES
        }
    }


def remote_sync_point(*payloads: RemoteData) -> SynchronizedPoint:
    """Combine the payloads of several remote nodes into one sync point."""
    data: RemoteData = {}
    for payload in payloads:
        data.update(payload)
    return SynchronizedPoint(timestamp=REMOTE_TIMESTAMP, data=data)


@pytest.fixture
def patch_remote_node_data(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., SimpleNamespace]:
    """Return a callable that stubs the remote-node seam.

    The returned callable takes a zero-argument factory — rather than a fixed
    sync point — so a test can vary what the remote nodes deliver between
    successive ``_predict`` attempts.

    Pass ``consume_enabled=False`` to leave live-data consumption switched off,
    which is the negative control: the stub consumer is installed but should
    never be reached.

    The stub consumer is returned so tests can assert on how it was used.
    """

    def patch(
        get_remote_data: Callable[[], SynchronizedPoint],
        *,
        consume_enabled: bool = True,
    ) -> SimpleNamespace:
        consumer = SimpleNamespace(
            calls=0,
            num_remote_nodes=lambda: 1,
            all_remote_nodes_connected=lambda: True,
        )

        def get_latest_data() -> SynchronizedPoint:
            consumer.calls += 1
            return get_remote_data()

        consumer.get_latest_data = get_latest_data

        if consume_enabled:
            # The unit suite globally disables live data (see
            # tests/unit/conftest.py) and EnabledManager.disable() is one-way,
            # so the getter itself is stubbed rather than the manager it
            # returns.
            monkeypatch.setattr(
                "neuracore.core.get_latest_sync_point."
                "get_consume_live_data_enabled_manager",
                lambda: SimpleNamespace(is_disabled=lambda: False),
            )
        # Both get_latest_sync_point and check_remote_nodes_connected read this
        # name out of the same module, so patching here covers both.
        monkeypatch.setattr(
            "neuracore.core.get_latest_sync_point.get_org_nodes_manager",
            lambda org_id: SimpleNamespace(
                get_robot_consumer=lambda robot_id, robot_instance: consumer
            ),
        )
        return consumer

    return patch
