"""Dataset collection, mutation, and verification helpers for ML integration tests."""

import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
from neuracore_types import Dataset as DatasetModel
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import SYNC_PROGRESS_POLL_INTERVAL_S, Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.data.synced_dataset import SynchronizedDataset
from neuracore.core.utils.embodiment_description_utils import (
    merge_cross_embodiment_description,
)
from neuracore.core.utils.http_session import thread_local_session

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
    wait_for_recordings_finalized,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

logger = logging.getLogger(__name__)

RECORDING_POLL_INTERVAL_S = 5


def wait_for_dataset_recording_count(
    dataset_name: str,
    expected_recordings: int,
    timeout_seconds: int = 120,
    poll_seconds: int = 5,
) -> Dataset:
    deadline = time.time() + timeout_seconds
    last_count = None
    last_error = None

    while time.time() < deadline:
        try:
            dataset = nc.get_dataset(name=dataset_name)
            last_count = len(dataset)
            if last_count == expected_recordings:
                return dataset
            last_error = None
        except Exception as e:
            last_error = e

        time.sleep(poll_seconds)

    if last_error is not None:
        raise AssertionError(
            f"Dataset {dataset_name!r} did not become queryable within "
            f"{timeout_seconds} seconds; last error: {last_error}"
        )
    raise AssertionError(
        f"Dataset {dataset_name!r} had {last_count} recordings after "
        f"{timeout_seconds} seconds; expected {expected_recordings}"
    )


def collect_demo_data(
    robot_name: str,
    dataset_name: str,
    *,
    joint_names: tuple[str, ...] | list[str],
    gripper_names: list[str],
    language_label: str,
    nc_cam_name: str,
    pose_sensor_name: str,
    num_episodes: int = 3,
    instance_id: int = 0,
    episode_length_multiplier: int = 1,
    num_cameras: int = 1,
    frequency: float = 20,
    timestamp_jitter_frac: float = 0.05,
) -> Dataset:
    """Collect scripted demonstrations and log them to neuracore.

    Use different instances for different tests since they are run in parallel.
    Increase episode_length_multiplier to inflate episode length by repeating
    the rollout trajectory steps.
    Increase num_cameras to log multiple RGB streams per timestep.
    """
    assert (
        episode_length_multiplier >= 1
    ), f"episode_length_multiplier must be >= 1, got {episode_length_multiplier}"
    assert num_cameras >= 1, f"num_cameras must be >= 1, got {num_cameras}"

    with online_daemon_running():
        assert_exactly_one_daemon_pid()
        nc.connect_robot(
            robot_name=robot_name,
            instance=instance_id,
            urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
            overwrite=False,
        )
        dataset = nc.create_dataset(name=dataset_name)
        for ep_idx in range(num_episodes):
            logger.info(f"Collecting episode {ep_idx + 1}/{num_episodes}")
            action_traj = rollout_policy()
            expanded_action_traj = [
                action_dict
                for action_dict in action_traj
                for _ in range(episode_length_multiplier)
            ]
            nc.start_recording(robot_name=robot_name, instance=instance_id)
            t = time.time()
            timestamp_rng = np.random.default_rng(ep_idx)
            for frame_idx, action_dict in enumerate(expanded_action_traj):
                dt = 1.0 / frequency
                t += dt * float(
                    timestamp_rng.uniform(
                        1.0 - timestamp_jitter_frac, 1.0 + timestamp_jitter_frac
                    )
                )
                joint_positions = {
                    k: v for k, v in action_dict.items() if "gripper" not in k
                }
                joint_torques = {
                    name: float(0.01 * ((index + frame_idx) % 5))
                    for index, name in enumerate(joint_names)
                }
                joint_velocities = {
                    name: float(0.05 * ((index + frame_idx) % 7))
                    for index, name in enumerate(joint_names)
                }
                gripper_open_amounts = {
                    name: float(0.25 + 0.5 * ((frame_idx % 2) == 0))
                    for name in gripper_names
                }
                pose = np.array([0.1 + frame_idx * 0.001, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
                img = np.zeros((84, 84, 3), dtype=np.uint8)
                img.fill(50 + frame_idx % 200)

                nc.log_joint_positions(
                    positions=joint_positions,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_joint_target_positions(
                    target_positions=joint_positions,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_joint_velocities(
                    velocities=joint_velocities,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_joint_torques(
                    torques=joint_torques,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_parallel_gripper_open_amounts(
                    values=gripper_open_amounts,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_parallel_gripper_target_open_amounts(
                    values=gripper_open_amounts,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_pose(
                    name=pose_sensor_name,
                    pose=pose,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_language(
                    name=language_label,
                    language="pick and place",
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
                nc.log_rgb(
                    name=nc_cam_name,
                    rgb=img,
                    timestamp=t,
                    robot_name=robot_name,
                    instance=instance_id,
                )
            nc.stop_recording(wait=True, robot_name=robot_name, instance=instance_id)
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=ep_idx + 1,
                timeout_s=500,
                poll_interval_s=5,
            )
            logger.info(
                f"Episode {ep_idx + 1} recorded ({len(expanded_action_traj)} frames)"
            )
    return dataset


def delete_recording_from_dataset(dataset: Dataset, recording: Recording) -> None:
    """Remove a recording from a dataset via the platform API."""
    session = thread_local_session()
    response = session.delete(
        f"{API_URL}/org/{dataset.org_id}/datasets/{dataset.id}/recording/{recording.id}",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()


def fetch_dataset_model(dataset: Dataset) -> DatasetModel:
    """Fetch full dataset metadata including num_demonstrations and data types."""
    session = thread_local_session()
    response = session.get(
        f"{API_URL}/org/{dataset.org_id}/datasets/{dataset.id}",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()
    return DatasetModel.model_validate(response.json())


def assert_active_recordings(
    dataset: Dataset,
    *,
    expected_count: int,
    expected_types: set[DataType],
    tracked_ids: set[str] | None = None,
) -> set[str]:
    """Assert recording count, datatype presence, and optional ID set equality."""
    assert (
        len(dataset) == expected_count
    ), f"Expected {expected_count} recordings, got {len(dataset)}"
    active_ids: set[str] = set()
    for recording in dataset:
        active_ids.add(str(recording.id))
        missing = expected_types - recording.data_types
        assert not missing, (
            f"Recording {recording.id} missing datatypes "
            f"{sorted(dt.value for dt in missing)}; "
            f"has {sorted(dt.value for dt in recording.data_types)}"
        )
    if tracked_ids is not None:
        assert active_ids == tracked_ids, (
            f"Active recording IDs mismatch.\n"
            f"Expected: {sorted(tracked_ids)}\n"
            f"Got: {sorted(active_ids)}"
        )
    return active_ids


def assert_dataset_metadata(
    dataset: Dataset,
    *,
    expected_count: int,
    expected_common_types: set[DataType],
) -> DatasetModel:
    """Assert dataset metadata reports the expected recording count and types."""
    model = fetch_dataset_model(dataset)
    assert model.num_demonstrations == expected_count, (
        f"Metadata num_demonstrations={model.num_demonstrations}, "
        f"expected {expected_count}"
    )
    common = set(model.common_data_types.keys())
    missing = expected_common_types - common
    assert not missing, (
        f"common_data_types missing {sorted(dt.value for dt in missing)}; "
        f"has {sorted(dt.value for dt in common)}"
    )
    return model


def statistics_fingerprint(stats) -> str:
    """Stable hash of synchronized dataset statistics for refresh checks."""
    payload = stats.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def assert_synced_statistics(
    dataset: Dataset,
    input_desc: dict,
    output_desc: dict,
    *,
    expected_count: int,
    frequency: int = 20,
    finalize_timeout_seconds: float = 300,
    sync_timeout_seconds: float = 600,
    log_prefix: str | None = None,
) -> tuple[SynchronizedDataset, str]:
    """Synchronize dataset, calculate statistics, and assert consistency."""
    prefix = log_prefix or f"[{dataset.name}]"
    recording_ids = {str(recording.id) for recording in dataset}
    logger.info(
        f"{prefix} Waiting for {len(recording_ids)} recordings to finalize "
        f"(timeout={finalize_timeout_seconds}s)"
    )
    wait_for_recordings_finalized(
        dataset.name,
        recording_ids=recording_ids,
        timeout_s=finalize_timeout_seconds,
        poll_interval_s=RECORDING_POLL_INTERVAL_S,
    )
    logger.info(f"{prefix} All recordings finalized")

    cross_embodiment_union = merge_cross_embodiment_description(input_desc, output_desc)
    logger.info(f"{prefix} Starting dataset synchronization at {frequency} Hz")
    synced_model = dataset._synchronize(
        frequency=frequency,
        cross_embodiment_union=cross_embodiment_union,
    )
    total = synced_model.num_demonstrations
    deadline = time.time() + sync_timeout_seconds
    progress = dataset._get_synchronization_progress(synced_model.id)
    processed = progress.num_synchronized_demonstrations
    logger.info(
        f"{prefix} Synchronization progress: {processed}/{total} "
        f"(timeout={sync_timeout_seconds}s)"
    )

    while processed < total:
        if time.time() >= deadline:
            raise AssertionError(
                f"Synchronization timed out after {sync_timeout_seconds}s for "
                f"dataset {dataset.name!r}: {processed}/{total} recordings synced"
            )
        time.sleep(SYNC_PROGRESS_POLL_INTERVAL_S)
        progress = dataset._get_synchronization_progress(synced_model.id)
        new_processed = progress.num_synchronized_demonstrations
        if new_processed != processed:
            processed = new_processed
            logger.info(f"{prefix} Synchronization progress: {processed}/{total}")

    assert (
        not progress.has_failures
    ), f"Synchronization failures: {progress.failed_recording_ids}"
    assert progress.num_synchronized_demonstrations == expected_count, (
        f"Synchronized {progress.num_synchronized_demonstrations} recordings, "
        f"expected {expected_count}"
    )

    logger.info(f"{prefix} Synchronization complete ({processed}/{total})")

    synced = SynchronizedDataset(
        id=synced_model.id,
        dataset=dataset,
        frequency=frequency,
        cross_embodiment_union=cross_embodiment_union,
    )
    assert len(synced) == expected_count, (
        f"Synchronized dataset has {len(synced)} recordings, "
        f"expected {expected_count}"
    )

    logger.info(f"{prefix} Calculating dataset statistics")
    stats = synced.calculate_statistics(
        input_cross_embodiment_description=input_desc,
        output_cross_embodiment_description=output_desc,
    )
    assert stats.dataset_statistics, "Expected non-empty dataset_statistics"

    role_keys = {"input", "output"}
    actual_keys = set(stats.dataset_statistics.keys())
    assert role_keys <= actual_keys, (
        f"Expected dataset_statistics keys {sorted(role_keys)}, "
        f"got {sorted(actual_keys)}"
    )

    expected_input_types = {
        data_type for robot_desc in input_desc.values() for data_type in robot_desc
    }
    expected_output_types = {
        data_type for robot_desc in output_desc.values() for data_type in robot_desc
    }

    input_stats = stats.dataset_statistics["input"]
    output_stats = stats.dataset_statistics["output"]

    for data_type in expected_input_types:
        assert data_type in input_stats, f"Input stats missing {data_type.value}"
    for data_type in expected_output_types:
        assert data_type in output_stats, f"Output stats missing {data_type.value}"

    logger.info(f"{prefix} Dataset statistics calculated and validated")
    return synced, statistics_fingerprint(stats)
