"""Integration tests: dataset synchronization success / param-error / missing-data.

Four independent tests drive a freshly collected dataset through
synchronization to exercise the backend's permanent-failure handling and
post-mutation re-sync:

* :func:`test_dataset_synchronization_success` — synchronization **succeeds**,
* :func:`test_dataset_synchronization_param_error` — **fails on bad parameters**
  (``SynchronizationParameterError``), and
* :func:`test_dataset_synchronization_missing_data` — **fails on missing data**
  (``SynchronizationMissingDataError``), and
* :func:`test_dataset_synchronization_after_mutation` — **re-synchronizes** after
  removing and replacing half the recordings.

Each scenario drives the high-level ``Dataset.synchronize()``: success returns a
``SynchronizedDataset``, while the two failure classes raise a ``DatasetError``
whose message carries the backend's per-recording failure reason.

Target environment
------------------
Prefer **staging**: the real Cloud Tasks queue dispatches the sync-recording
tasks and the permanent-failure path issues real ``delete_task`` purges, so the
backend change is exercised end-to-end. It also works against a **dev** backend
(``ENVIRONMENT=dev``), where tasks run locally; only the Cloud Tasks purge is a
no-op there and must be verified manually.

Each dataset contains ``RECORDINGS_PER_DATASET`` recordings so the
permanent-failure path has sibling tasks to purge.
"""

import contextlib
import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from neuracore.core.exceptions import DatasetError

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.base_env import BimanualViperXTask
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH

from tests.integration.ml.shared.dataset import delete_recording_from_dataset
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
    wait_for_recordings_finalized,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    delete_cloud_robot,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREQUENCY = 20  # Hz the demo data is logged at.
MAX_FRAMES = 30  # Truncate the rollout to keep recordings short.
RECORDINGS_PER_DATASET = 10  # Recordings collected for each dataset.
RECORDINGS_TO_REMOVE = 5  # Recordings removed during the mutation scenario.
RECORDINGS_TO_ADD = 5  # Replacement recordings added after removal.
NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
LANGUAGE_LABEL = "instruction"
RECORDING_STOP_TIMEOUT_SECONDS = 500
RECORDING_FINALIZE_TIMEOUT_SECONDS = 300

# A data type the collector never logs, used to force a missing-data failure.
MISSING_DATA_TYPE = DataType.DEPTH_IMAGES
MISSING_SENSOR_NAME = "depth_angle"

# Synchronization progress polling.
PROGRESS_POLL_SECONDS = 5

JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
)
GRIPPER_NAMES = ["left_gripper", "right_gripper"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _record_one(robot_name: str, instance: int) -> None:
    """Log one short scripted recording's worth of synchronizable streams.

    Logs joint positions/targets, gripper open amounts, an RGB stream and a
    language label per frame at ``FREQUENCY`` Hz with a shared timestamp so the
    streams are synchronizable. Deliberately does NOT log ``MISSING_DATA_TYPE``.
    """
    action_traj = rollout_policy()[:MAX_FRAMES]
    nc.start_recording(robot_name=robot_name, instance=instance)
    t = time.time()
    for frame_idx, action_dict in enumerate(action_traj):
        t += 1.0 / FREQUENCY
        joint_positions = {k: v for k, v in action_dict.items() if "gripper" not in k}
        gripper_open_amounts = {
            name: float(0.25 + 0.5 * ((frame_idx % 2) == 0)) for name in GRIPPER_NAMES
        }
        img = np.zeros((84, 84, 3), dtype=np.uint8)
        img.fill(50 + frame_idx % 200)

        nc.log_joint_positions(
            positions=joint_positions,
            timestamp=t,
            robot_name=robot_name,
            instance=instance,
        )
        nc.log_joint_target_positions(
            target_positions=joint_positions,
            timestamp=t,
            robot_name=robot_name,
            instance=instance,
        )
        nc.log_parallel_gripper_open_amounts(
            values=gripper_open_amounts,
            timestamp=t,
            robot_name=robot_name,
            instance=instance,
        )
        nc.log_language(
            name=LANGUAGE_LABEL,
            language="pick and place",
            timestamp=t,
            robot_name=robot_name,
            instance=instance,
        )
        nc.log_rgb(
            name=NC_CAM_NAME,
            rgb=img,
            timestamp=t,
            robot_name=robot_name,
            instance=instance,
        )
    nc.stop_recording(wait=True, robot_name=robot_name, instance=instance)


def _collect_dataset(robot_name: str, dataset_name: str, instance: int) -> Dataset:
    """Collect ``RECORDINGS_PER_DATASET`` scripted recordings into one dataset."""
    nc.connect_robot(
        robot_name=robot_name,
        instance=instance,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    dataset = nc.create_dataset(name=dataset_name)

    for _ in range(RECORDINGS_PER_DATASET):
        _record_one(robot_name=robot_name, instance=instance)

    wait_for_dataset_ready(
        dataset_name,
        expected_recording_count=RECORDINGS_PER_DATASET,
        timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
        poll_interval_s=PROGRESS_POLL_SECONDS,
    )
    # Synchronization reads finalized traces from the bucket, so block until the
    # recording's end_time is set before returning.
    ready_dataset = nc.get_dataset(dataset_name)
    recording_ids = {str(recording.id) for recording in ready_dataset}
    wait_for_recordings_finalized(
        dataset_name,
        recording_ids=recording_ids,
        timeout_s=RECORDING_FINALIZE_TIMEOUT_SECONDS,
        poll_interval_s=PROGRESS_POLL_SECONDS,
    )
    return dataset


def _add_recordings(
    robot_name: str,
    dataset_name: str,
    instance: int,
    count: int,
    known_recording_ids: set[str],
) -> set[str]:
    """Append ``count`` scripted recordings to an existing dataset."""
    new_ids: set[str] = set()
    expected_count = len(known_recording_ids)

    for _ in range(count):
        _record_one(robot_name=robot_name, instance=instance)
        expected_count += 1
        wait_for_dataset_ready(
            dataset_name,
            expected_recording_count=expected_count,
            timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
            poll_interval_s=PROGRESS_POLL_SECONDS,
        )
        current_ids = {str(recording.id) for recording in nc.get_dataset(dataset_name)}
        added = current_ids - known_recording_ids - new_ids
        assert (
            len(added) == 1
        ), f"Expected exactly one new recording, got {sorted(added)}"
        new_ids.add(added.pop())

    wait_for_recordings_finalized(
        dataset_name,
        recording_ids=new_ids,
        timeout_s=RECORDING_FINALIZE_TIMEOUT_SECONDS,
        poll_interval_s=PROGRESS_POLL_SECONDS,
    )
    return new_ids


def _assert_failure_with_reason(
    excinfo: pytest.ExceptionInfo[DatasetError],
    marker: str,
) -> None:
    """Assert a permanent synchronization failure with a matching reason.

    The backend signals a permanent failure with a 422 whose detail carries the
    per-recording failure reason, and ``Dataset.synchronize()`` surfaces that
    reason verbatim as the ``DatasetError`` message. Assert ``marker`` appears
    in the surfaced reason; it is specific to a failure class, so it also rules
    out an unrelated error (e.g. a 500's "Failed to fetch synchronization
    progress.") that would also raise ``DatasetError``.
    """
    message = str(excinfo.value)
    assert marker in message, f"Expected {marker!r} in failure reason, got: {message!r}"


@contextlib.contextmanager
def _collected_dataset(name_prefix: str) -> Iterator[Dataset]:
    """Yield a freshly collected dataset, cleaning it up on exit.

    Logs in, runs the online daemon while collecting
    ``RECORDINGS_PER_DATASET`` recordings into a uniquely named dataset, then
    deletes the dataset once the caller is done.
    """
    nc.login()
    run_id = uuid.uuid4().hex[:8]
    robot_name = f"sync_it_robot_{run_id}"
    dataset: Dataset | None = None
    try:
        with online_daemon_running():
            assert_exactly_one_daemon_pid()
            dataset = _collect_dataset(
                robot_name=robot_name,
                dataset_name=_unique_name(name_prefix),
                instance=0,
            )
        yield dataset
    finally:
        if dataset is not None:
            try:
                dataset.delete()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to clean up dataset %s", dataset.id)
        delete_cloud_robot(robot_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_synchronization_success() -> None:
    """Sane params (downsample below the data rate) synchronize successfully."""
    with _collected_dataset("sync_it_success") as dataset:
        # synchronize() polls to completion and raises on any failure, so a
        # clean return is the success signal; the result wraps every recording.
        synced = dataset.synchronize(
            frequency=10,
            allow_duplicates=True,
            max_delay_s=1.0,
            cross_embodiment_union=None,
        )
        assert len(synced) == RECORDINGS_PER_DATASET


def test_dataset_synchronization_param_error() -> None:
    """Frequency far above the data rate with no duplicates fails on params.

    The backend rejects this as ``data_type_too_sparse``.
    """
    with _collected_dataset("sync_it_param_error") as dataset:
        with pytest.raises(DatasetError) as excinfo:
            dataset.synchronize(
                frequency=1000,
                allow_duplicates=False,
                max_delay_s=1.0,
                cross_embodiment_union=None,
            )
        _assert_failure_with_reason(excinfo, "too sparse")


def test_dataset_synchronization_missing_data() -> None:
    """Requesting a sensor that was never logged fails on missing data.

    The backend rejects this as ``no_sensors_for_data_type``.
    """
    with _collected_dataset("sync_it_missing_data") as dataset:
        with pytest.raises(DatasetError) as excinfo:
            dataset.synchronize(
                frequency=10,
                allow_duplicates=True,
                max_delay_s=1.0,
                cross_embodiment_union={"": {MISSING_DATA_TYPE: [MISSING_SENSOR_NAME]}},
            )
        _assert_failure_with_reason(excinfo, "No sensors found for data type")


def test_dataset_synchronization_after_mutation() -> None:
    """Synchronize, remove half the recordings, add replacements, re-synchronize."""
    nc.login()
    run_id = uuid.uuid4().hex[:8]
    robot_name = f"sync_it_robot_{run_id}"
    dataset_name = _unique_name("sync_it_mutation")
    dataset: Dataset | None = None
    sync_kwargs = {
        "frequency": 10,
        "allow_duplicates": True,
        "max_delay_s": 1.0,
        "cross_embodiment_union": None,
    }
    try:
        with online_daemon_running():
            assert_exactly_one_daemon_pid()

            logger.info(
                "[STEP 1] Collecting %d recordings into %r",
                RECORDINGS_PER_DATASET,
                dataset_name,
            )
            dataset = _collect_dataset(
                robot_name=robot_name,
                dataset_name=dataset_name,
                instance=0,
            )
            logger.info(
                "[STEP 1] [PASSED] Collected %d recordings into %r",
                len(dataset),
                dataset_name,
            )

            logger.info("[STEP 2] Synchronizing dataset (initial)")
            synced_initial = dataset.synchronize(**sync_kwargs)
            assert len(synced_initial) == RECORDINGS_PER_DATASET
            logger.info(
                "[STEP 2] [PASSED] Initial synchronization complete: "
                "%d recordings synced",
                len(synced_initial),
            )

            logger.info(
                "[STEP 3] Removing %d recordings from %r",
                RECORDINGS_TO_REMOVE,
                dataset_name,
            )
            recordings_to_delete = [
                dataset[index] for index in range(RECORDINGS_TO_REMOVE)
            ]
            deleted_ids = {str(recording.id) for recording in recordings_to_delete}
            for recording in recordings_to_delete:
                logger.info("[STEP 3] Deleting recording %s", recording.id)
                delete_recording_from_dataset(dataset=dataset, recording=recording)

            remaining = RECORDINGS_PER_DATASET - RECORDINGS_TO_REMOVE
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=remaining,
                timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
                poll_interval_s=PROGRESS_POLL_SECONDS,
            )
            dataset = nc.get_dataset(dataset_name)
            surviving_ids = {str(recording.id) for recording in dataset}
            assert deleted_ids.isdisjoint(surviving_ids)
            assert len(surviving_ids) == remaining
            logger.info(
                "[STEP 3] [PASSED] Removed %d recordings; %d remaining",
                RECORDINGS_TO_REMOVE,
                len(surviving_ids),
            )

            logger.info(
                "[STEP 4] Adding %d recordings to %r",
                RECORDINGS_TO_ADD,
                dataset_name,
            )
            new_ids = _add_recordings(
                robot_name=robot_name,
                dataset_name=dataset_name,
                instance=0,
                count=RECORDINGS_TO_ADD,
                known_recording_ids=surviving_ids,
            )
            active_ids = surviving_ids | new_ids
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=RECORDINGS_PER_DATASET,
                timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
                poll_interval_s=PROGRESS_POLL_SECONDS,
            )
            dataset = nc.get_dataset(dataset_name)
            assert len(active_ids) == RECORDINGS_PER_DATASET
            assert deleted_ids.isdisjoint(active_ids)
            assert new_ids <= active_ids
            logger.info(
                "[STEP 4] [PASSED] Added %d recordings; dataset has %d total",
                len(new_ids),
                len(dataset),
            )

            logger.info("[STEP 5] Re-synchronizing dataset after mutation")
            synced_after = dataset.synchronize(**sync_kwargs)
            assert len(synced_after) == RECORDINGS_PER_DATASET
            assert {
                str(recording.id) for recording in synced_after.dataset
            } == active_ids
            logger.info(
                "[STEP 5] [PASSED] Re-synchronization complete: "
                "%d recordings synced",
                len(synced_after),
            )
    finally:
        if dataset is not None:
            try:
                dataset.delete()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to clean up dataset %s", dataset.id)
        delete_cloud_robot(robot_name)
