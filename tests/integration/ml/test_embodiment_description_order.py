"""Integration test: recording-order dependence of the embodiment description.

TODO: rewrite against the union backend before enabling.
This test was written against the legacy first-recording behaviour of
get_full_embodiment_description, which turned out to pick an effectively
arbitrary recording (no order_by on the join-entry query, so the database
returns entries in document-id order over random recording ids) — the
order-based expectations below can never be deterministic. With
neuracore_backend feat/full-embodiment-description-union the description
is the deterministic union across recordings, so the rewrite should:
  - assert the description equals the union regardless of recording order;
  - assert a training job built from the union FAILS deterministically on
    datasets whose recordings are missing some sensors (the per-recording
    sensor-name validation gap); the "sparse-first order silently trains"
    scenario no longer exists;
  - optionally use /datasets/{id}/embodiment-sensor-names/{robot_id} to
    build a safe request from common_sensor_names and assert it COMPLETES.

The backend's get_full_embodiment_description builds the description from the
FIRST recording it finds for a robot and then stops (documented compromise).
For datasets whose recordings log different joint subsets, the description —
and therefore training validation — depends on recording order:

  Dataset A (decreasing): episodes log {j1,j2,j3}, {j1,j2}, {j1}.
    Description reflects the first episode (all three joints), validation
    accepts a three-joint training request, the job is submitted — and then
    FAILS during training because the later recordings are missing j2/j3.

  Dataset B (increasing): episodes log {j1}, {j1,j2}, {j1,j2,j3}.
    Description shrinks to just j1, the training request built from it only
    asks for j1, validation accepts, and the job COMPLETES — silently
    discarding the j2/j3 data in the richer recordings.

This encodes the CURRENT first-recording behaviour. If
get_full_embodiment_description is ever fixed to return the union across
recordings, dataset B's request will include all three joints, its job will
fail like dataset A's, and this test must be revisited.
"""

import logging
import time
import uuid

import pytest
from neuracore_types import DataType, EmbodimentDescription

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from tests.integration.platform.data_daemon.shared.assertions import (
    assert_exactly_one_daemon_pid,
)
from tests.integration.platform.data_daemon.shared.db_helpers import (
    wait_for_dataset_ready,
)
from tests.integration.platform.data_daemon.shared.runners import online_daemon_running

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINTS_FULL = ["joint_1", "joint_2", "joint_3"]
ROBOT_NAME = "embodiment_order_test_robot"
ROBOT_INSTANCE = 0
FREQUENCY = 20
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
EPISODE_STEPS = 20
RECORDING_STOP_TIMEOUT_SECONDS = 300
RECORDING_POLL_SECONDS = 5
TRAINING_TIMEOUT_MINUTES = 40
TRAINING_POLL_SECONDS = 20
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}

# Episode joint subsets per dataset. Only recording ORDER differs.
DECREASING_JOINT_SETS = [JOINTS_FULL, JOINTS_FULL[:2], JOINTS_FULL[:1]]
INCREASING_JOINT_SETS = [JOINTS_FULL[:1], JOINTS_FULL[:2], JOINTS_FULL]

CNNMLP_CONFIG = {
    "batch_size": 8,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _log_one_episode(joint_names: list[str]) -> None:
    """Record one episode logging positions and target positions for joints."""
    t = time.time()
    nc.start_recording(robot_name=ROBOT_NAME, instance=ROBOT_INSTANCE)
    for step in range(EPISODE_STEPS):
        t += 1.0 / FREQUENCY
        values = {name: 0.01 * step for name in joint_names}
        nc.log_joint_positions(
            positions=values,
            robot_name=ROBOT_NAME,
            instance=ROBOT_INSTANCE,
            timestamp=t,
        )
        nc.log_joint_target_positions(
            target_positions=values,
            robot_name=ROBOT_NAME,
            instance=ROBOT_INSTANCE,
            timestamp=t,
        )
    nc.stop_recording(wait=False, robot_name=ROBOT_NAME, instance=ROBOT_INSTANCE)


def _collect_dataset(dataset_name: str, joint_sets: list[list[str]]) -> Dataset:
    """Record one episode per joint subset, in order, into a new dataset."""
    nc.create_dataset(name=dataset_name)
    with online_daemon_running():
        assert_exactly_one_daemon_pid()
        for ep_idx, joint_names in enumerate(joint_sets):
            logger.info(
                f"Recording episode {ep_idx + 1}/{len(joint_sets)} "
                f"into '{dataset_name}' with joints={joint_names}"
            )
            _log_one_episode(joint_names)
            wait_for_dataset_ready(
                dataset_name,
                expected_recording_count=ep_idx + 1,
                timeout_s=RECORDING_STOP_TIMEOUT_SECONDS,
                poll_interval_s=RECORDING_POLL_SECONDS,
            )
            logger.info(f"Episode {ep_idx + 1} ready.")
    dataset = nc.get_dataset(name=dataset_name)
    assert len(dataset) == len(joint_sets), (
        f"Expected {len(joint_sets)} recordings in '{dataset_name}', "
        f"got {len(dataset)}"
    )
    return dataset


def _description_joints(
    description: EmbodimentDescription, data_type: DataType
) -> set[str]:
    return set(description.get(data_type, {}).values())


def _submit_training_job(dataset: Dataset, dataset_name: str, job_name: str) -> str:
    """Build descriptions via get_full_embodiment_description and submit."""
    robot_ids = dataset.robot_ids
    assert len(robot_ids) == 1, f"Expected 1 robot, got {robot_ids}"
    robot_id = robot_ids[0]
    description = dataset.get_full_embodiment_description(robot_id)

    job_data = nc.start_training_run(
        name=job_name,
        dataset_name=dataset_name,
        algorithm_name="CNNMLP",
        algorithm_config=CNNMLP_CONFIG,
        gpu_type=GPU_TYPE,
        num_gpus=NUM_GPUS,
        frequency=FREQUENCY,
        input_cross_embodiment_description={
            robot_id: {
                DataType.JOINT_POSITIONS: description[DataType.JOINT_POSITIONS],
            }
        },
        output_cross_embodiment_description={
            robot_id: {
                DataType.JOINT_TARGET_POSITIONS: description[
                    DataType.JOINT_TARGET_POSITIONS
                ],
            }
        },
    )
    job_id = job_data["id"]
    assert isinstance(job_id, str) and job_id, f"Bad job id: {job_data}"
    return job_id


def _wait_for_terminal_status(job_id: str) -> str:
    deadline = time.time() + TRAINING_TIMEOUT_MINUTES * 60
    while True:
        status = nc.get_training_job_status(job_id=job_id)
        logger.info(f"Training job {job_id}: {status}")
        if status in TERMINAL_STATES:
            return status
        assert time.time() < deadline, (
            f"Training job {job_id} did not reach a terminal state within "
            f"{TRAINING_TIMEOUT_MINUTES} minutes"
        )
        time.sleep(TRAINING_POLL_SECONDS)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Written against legacy first-recording embodiment description; "
    "rewrite against the union backend — see module TODO."
)
class TestEmbodimentDescriptionOrder:
    """Recording order flips training between failure and silent success."""

    track_step_teardown = True
    all_steps_passed: bool = True
    decreasing_dataset_name: str
    increasing_dataset_name: str
    decreasing_dataset: Dataset | None = None
    increasing_dataset: Dataset | None = None
    decreasing_job_id: str | None = None
    increasing_job_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.decreasing_dataset_name = _unique_name(prefix="embodiment_order_dec")
        cls.increasing_dataset_name = _unique_name(prefix="embodiment_order_inc")
        cls.decreasing_dataset = None
        cls.increasing_dataset = None
        cls.decreasing_job_id = None
        cls.increasing_job_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestEmbodimentDescriptionOrder teardown: "
                "one or more steps failed"
            )
            return
        for job_id in (cls.decreasing_job_id, cls.increasing_job_id):
            if job_id:
                try:
                    nc.delete_training_job(job_id)
                except Exception:
                    logger.warning(f"Failed to delete job {job_id}", exc_info=True)
        for dataset in (cls.decreasing_dataset, cls.increasing_dataset):
            if dataset is not None:
                try:
                    dataset.delete()
                except Exception:
                    logger.warning(
                        f"Failed to delete dataset {dataset.name}", exc_info=True
                    )

    def test_step1_decreasing_order_passes_validation(self) -> None:
        """Full-joints-first dataset: description hides the sparse recordings."""
        nc.connect_robot(
            robot_name=ROBOT_NAME,
            instance=ROBOT_INSTANCE,
            overwrite=False,
        )
        self.__class__.decreasing_dataset = _collect_dataset(
            self.decreasing_dataset_name, DECREASING_JOINT_SETS
        )

        robot_id = self.decreasing_dataset.robot_ids[0]
        description = self.decreasing_dataset.get_full_embodiment_description(robot_id)
        assert _description_joints(description, DataType.JOINT_POSITIONS) == set(
            JOINTS_FULL
        ), (
            "Expected the description to reflect the first (full) recording.\n"
            f"Got: {description}"
        )

        # Validation accepts the three-joint request even though two of the
        # three recordings do not contain joint_2/joint_3 values.
        self.__class__.decreasing_job_id = _submit_training_job(
            dataset=self.decreasing_dataset,
            dataset_name=self.decreasing_dataset_name,
            job_name=_unique_name(prefix="embodiment_order_dec_job"),
        )
        logger.info(
            f"[STEP 1] [PASSED] Validation accepted incomplete dataset; "
            f"job {self.decreasing_job_id} submitted"
        )

    def test_step2_decreasing_order_training_fails(self) -> None:
        """The falsely-accepted job must fail once training touches the data."""
        assert self.decreasing_job_id is not None, "[STEP 1] Did Not Complete"
        final_status = _wait_for_terminal_status(self.decreasing_job_id)
        assert final_status == "FAILED", (
            "Job trained on a dataset whose later recordings lack joint_2/"
            f"joint_3 should fail, got: {final_status}"
        )
        logger.info(
            f"[STEP 2] [PASSED] Job {self.decreasing_job_id} failed as expected"
        )

    def test_step3_increasing_order_passes_validation(self) -> None:
        """Sparse-first dataset: description silently shrinks to joint_1."""
        self.__class__.increasing_dataset = _collect_dataset(
            self.increasing_dataset_name, INCREASING_JOINT_SETS
        )

        robot_id = self.increasing_dataset.robot_ids[0]
        description = self.increasing_dataset.get_full_embodiment_description(robot_id)
        assert _description_joints(description, DataType.JOINT_POSITIONS) == {
            "joint_1"
        }, (
            "Expected the description to shrink to the first (sparse) "
            f"recording.\nGot: {description}"
        )

        self.__class__.increasing_job_id = _submit_training_job(
            dataset=self.increasing_dataset,
            dataset_name=self.increasing_dataset_name,
            job_name=_unique_name(prefix="embodiment_order_inc_job"),
        )
        logger.info(
            f"[STEP 3] [PASSED] Validation accepted joint_1-only request; "
            f"job {self.increasing_job_id} submitted"
        )

    def test_step4_increasing_order_training_completes(self) -> None:
        """The shrunken request trains fine — j2/j3 data silently unused."""
        assert self.increasing_job_id is not None, "[STEP 3] Did Not Complete"
        final_status = _wait_for_terminal_status(self.increasing_job_id)
        assert final_status == "COMPLETED", (
            "Job restricted to joint_1 (present in every recording) should "
            f"complete, got: {final_status}"
        )
        logger.info(
            f"[STEP 4] [PASSED] Job {self.increasing_job_id} completed; "
            "joint_2/joint_3 data was silently dropped"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-svx"])
