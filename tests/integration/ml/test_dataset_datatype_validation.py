"""Integration test: dataset datatype validation before training run.

Builds a dataset of 8 recordings (2x joint positions only, 2x joint
velocities only, 2x rgb only, 2x all three), then asserts that requesting
all three input types raises a grouped ValueError before any job is
submitted.
"""

import logging
import re
import time

import numpy as np
import pytest
from neuracore_types import DataType

import neuracore as nc
from neuracore.api.training import get_training_jobs
from neuracore.core.data.dataset import Dataset
from tests.integration.ml.shared.utils import unique_name
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "vx300s_left/waist",
    "vx300s_left/shoulder",
    "vx300s_left/elbow",
    "vx300s_left/forearm_roll",
    "vx300s_left/wrist_angle",
    "vx300s_left/wrist_rotate",
    "vx300s_right/waist",
    "vx300s_right/shoulder",
    "vx300s_right/elbow",
    "vx300s_right/forearm_roll",
    "vx300s_right/wrist_angle",
    "vx300s_right/wrist_rotate",
]
NC_CAM_NAME = "rgb_angle"
ROBOT_NAME = "datatype_validation_test_robot"
ROBOT_INSTANCE = 0
FREQUENCY = 20
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
EPISODE_STEPS = 5
RECORDING_STOP_TIMEOUT_SECONDS = 300
RECORDING_POLL_SECONDS = 5

JOINT_POSITION_SAMPLE = {name: 0.1 for name in JOINT_NAMES}
JOINT_VELOCITY_SAMPLE = {name: 0.05 for name in JOINT_NAMES}
RGB_FRAME = np.full((84, 84, 3), 128, dtype=np.uint8)

REQUESTED_TYPES = {
    DataType.JOINT_POSITIONS,
    DataType.JOINT_VELOCITIES,
    DataType.RGB_IMAGES,
}
EPISODE_DATA_TYPES: list[set[DataType]] = (
    [{DataType.JOINT_POSITIONS}] * 2
    + [{DataType.JOINT_VELOCITIES}] * 2
    + [{DataType.RGB_IMAGES}] * 2
    + [REQUESTED_TYPES] * 2
)
TOTAL_RECORDINGS = len(EPISODE_DATA_TYPES)
COMPLETE_RECORDINGS = EPISODE_DATA_TYPES.count(REQUESTED_TYPES)

VALIDATION_ERROR_HEADER = (
    "Failed to start training run: some recordings are missing requested datatypes."
)

CNNMLP_CONFIG = {
    "batch_size": 16,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


def _record_episode(data_types: set[DataType]) -> None:
    """Record one episode logging only the given data types."""
    t = time.time()
    nc.start_recording(robot_name=ROBOT_NAME, instance=ROBOT_INSTANCE)
    for _ in range(EPISODE_STEPS):
        t += 1.0 / FREQUENCY
        if DataType.JOINT_POSITIONS in data_types:
            nc.log_joint_positions(
                positions=JOINT_POSITION_SAMPLE,
                robot_name=ROBOT_NAME,
                instance=ROBOT_INSTANCE,
                timestamp=t,
            )
        if DataType.JOINT_VELOCITIES in data_types:
            nc.log_joint_velocities(
                velocities=JOINT_VELOCITY_SAMPLE,
                robot_name=ROBOT_NAME,
                instance=ROBOT_INSTANCE,
                timestamp=t,
            )
        if DataType.RGB_IMAGES in data_types:
            nc.log_rgb(
                name=NC_CAM_NAME,
                rgb=RGB_FRAME,
                robot_name=ROBOT_NAME,
                instance=ROBOT_INSTANCE,
                timestamp=t,
            )
    nc.stop_recording(wait=True, robot_name=ROBOT_NAME, instance=ROBOT_INSTANCE)


def _missing_recordings_from_error(error_msg: str) -> dict[str, set[str]]:
    """Parse 'Missing <type>:' sections into {datatype name: recording names}."""
    sections: dict[str, set[str]] = {}
    current: set[str] | None = None
    for line in error_msg.splitlines():
        line = line.strip()
        if line.startswith("Missing ") and line.endswith(":"):
            current = sections.setdefault(line[len("Missing ") : -1], set())
        elif line.startswith("- "):
            assert (
                current is not None
            ), f"Recording listed before any 'Missing <type>:' header:\n{error_msg}"
            current.add(line[len("- ") :])
    return sections


def _delete_training_job_by_name(job_name: str) -> None:
    """Delete the training job with the given name, if one exists."""
    for job in get_training_jobs():
        if job.get("name") == job_name and job.get("id"):
            try:
                nc.delete_training_job(job["id"])
                logger.warning(f"Deleted training job {job_name!r}")
            except Exception:
                logger.warning(
                    f"Failed to delete training job {job_name!r}", exc_info=True
                )


class TestDatasetDatatypeValidation:
    """Mixed-datatype dataset must fail training validation with a grouped error."""

    track_step_teardown = True
    all_steps_passed: bool = True
    dataset_name: str
    dataset: Dataset | None = None
    # recording id -> data types logged for that episode in step 1
    recording_data_types: dict[str, set[DataType]]

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="datatype_validation")
        cls.dataset = None
        cls.recording_data_types = {}
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestDatasetDatatypeValidation teardown: "
                "one or more steps failed"
            )
            return
        if cls.dataset is not None:
            try:
                cls.dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete dataset {cls.dataset_name}", exc_info=True
                )

    def test_step1_collect_mixed_recordings(self) -> None:
        """Collect 8 recordings with four distinct datatype profiles."""
        nc.connect_robot(
            robot_name=ROBOT_NAME,
            instance=ROBOT_INSTANCE,
            overwrite=False,
        )
        nc.create_dataset(name=self.dataset_name)

        with online_daemon_running():
            assert_exactly_one_daemon_pid()
            for ep_idx, data_types in enumerate(EPISODE_DATA_TYPES):
                logger.info(
                    f"Recording episode {ep_idx + 1}/{TOTAL_RECORDINGS}: "
                    f"{sorted(dt.value for dt in data_types)}"
                )
                _record_episode(data_types)
                wait_for_dataset_ready(
                    self.dataset_name,
                    expected_recording_count=ep_idx + 1,
                    timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
                    poll_interval_s=RECORDING_POLL_SECONDS,
                )
                current_ids = {
                    str(r.id) for r in nc.get_dataset(name=self.dataset_name)
                }
                new_ids = current_ids - set(self.recording_data_types)
                assert len(new_ids) == 1, (
                    f"Expected exactly one new recording after episode "
                    f"{ep_idx + 1}, got {sorted(new_ids)}"
                )
                self.recording_data_types[new_ids.pop()] = data_types
                logger.info(f"Episode {ep_idx + 1} ready.")

        self.__class__.dataset = nc.get_dataset(name=self.dataset_name)
        assert (
            len(self.dataset) == TOTAL_RECORDINGS
        ), f"Expected {TOTAL_RECORDINGS} recordings, got {len(self.dataset)}"
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.dataset)} recordings "
            f"in '{self.dataset_name}'"
        )

    def test_step2_training_fails_with_grouped_error(self) -> None:
        """start_training_run must raise the grouped ValueError before submitting."""
        assert self.dataset is not None, "[STEP 1] Did Not Complete"

        robot_ids = self.dataset.robot_ids
        assert len(robot_ids) == 1, f"Expected 1 robot, got {robot_ids}"
        robot_id = robot_ids[0]

        # get_full_embodiment_description only reflects one recording, so it
        # cannot express this deliberately heterogeneous dataset; build the
        # input description from the names the test logged instead.
        indexed_joints = dict(enumerate(JOINT_NAMES))
        input_desc = {
            robot_id: {
                DataType.JOINT_POSITIONS: indexed_joints,
                DataType.JOINT_VELOCITIES: indexed_joints,
                DataType.RGB_IMAGES: {0: NC_CAM_NAME},
            }
        }

        assert len(self.recording_data_types) == TOTAL_RECORDINGS, (
            f"Expected datatype ground truth for {TOTAL_RECORDINGS} recordings, "
            f"got {len(self.recording_data_types)}"
        )
        expected_missing = {
            data_type.value: {
                recording_id
                for recording_id, logged_types in self.recording_data_types.items()
                if data_type not in logged_types
            }
            for data_type in REQUESTED_TYPES
        }
        complete_recordings = {
            recording_id
            for recording_id, logged_types in self.recording_data_types.items()
            if logged_types == REQUESTED_TYPES
        }
        assert len(complete_recordings) == COMPLETE_RECORDINGS, (
            f"Expected {COMPLETE_RECORDINGS} fully-typed recordings, "
            f"got {sorted(complete_recordings)}"
        )

        job_name = unique_name(prefix="validation_test_job")
        try:
            with pytest.raises(
                ValueError, match=rf"^{re.escape(VALIDATION_ERROR_HEADER)}"
            ) as exc_info:
                nc.start_training_run(
                    name=job_name,
                    dataset_name=self.dataset_name,
                    algorithm_name="CNNMLP",
                    algorithm_config=CNNMLP_CONFIG,
                    gpu_type=GPU_TYPE,
                    num_gpus=NUM_GPUS,
                    frequency=FREQUENCY,
                    input_cross_embodiment_description=input_desc,
                    output_cross_embodiment_description={},
                )
        except BaseException:
            # Validation regressed: a real GPU job may exist; delete it
            # before failing.
            _delete_training_job_by_name(job_name)
            raise

        error_msg = str(exc_info.value)
        logger.info(f"Got expected ValueError:\n{error_msg}")

        assert _missing_recordings_from_error(error_msg) == expected_missing, (
            f"Grouped error does not match the per-episode ground truth.\n"
            f"Expected: {expected_missing}\nFull message:\n{error_msg}"
        )
        for recording_id in complete_recordings:
            assert recording_id not in error_msg, (
                f"Fully-typed recording {recording_id} must not appear in the "
                f"error.\nGot: {error_msg}"
            )

        job_names = {job.get("name") for job in get_training_jobs()}
        assert (
            job_name not in job_names
        ), f"Training job {job_name!r} was created despite failing validation"

        logger.info("[STEP 2] [PASSED] Training correctly refused with grouped error")
