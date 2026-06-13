"""End-to-end training lifecycle integration test.

Covers the core training lifecycle: dataset collection, merging, training,
log retrieval while running, and completion. On success this leaves a
COMPLETED training run named under ``INFERENCE_MODEL_TRAIN_RUN_PREFIX`` so
the separate inference and resume-training tests can discover it.

This test intentionally does NOT delete its training job or datasets in
teardown. The resume-training test needs the datasets to resume training,
and the inference test retains the job as a known-good model (pruning only
superseded runs).
"""

import logging
import pprint
import time

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from tests.integration.ml.shared.constants import (
    FREQUENCY,
    GPU_TYPE,
    GRIPPER_NAMES,
    INPUT_DATA_TYPES,
    JOINT_NAMES,
    LANGUAGE_LABEL,
    NC_CAM_NAME,
    NUM_GPUS,
    OUTPUT_DATA_TYPES,
    POSE_SENSOR_NAME,
    ROBOT_NAME,
)
from tests.integration.ml.shared.data_collection import (
    collect_demo_data,
    wait_for_dataset_recording_count,
)
from tests.integration.ml.shared.training import (
    TERMINAL_STATES,
    assert_no_training_log_errors,
    build_cross_embodiment_descriptions,
    cancel_incomplete_training_jobs,
    wait_for_training,
)
from tests.integration.ml.shared.utils import (
    cleanup_training_flow_datasets,
    unique_name,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Predefined name prefix for the training run produced by this test. The
# inference and resume tests scan for the latest COMPLETED run under this prefix
INFERENCE_MODEL_TRAIN_RUN_PREFIX = "ml_integration_flow"

# Dataset name prefixes used by this test. The resume training test cleans up
# datasets matching these prefixes after resume.
FLOW_COLLECTED_DATASET_PREFIX = "collected"
FLOW_MERGED_DATASET_PREFIX = "merged"

JOB_STATE_POLL_SECONDS = 30
RUNNING_STATE_TIMEOUT_MINUTES = 10
LOGS_AVAILABILITY_TIMEOUT_MINUTES = 10
SHARED_DATASET_NAME = "NYU ROT"
COLLECTED_DEMO_EPISODES = 3
CNNMLP_CONFIG = {
    "batch_size": 64,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


class TestTrainingFlow:
    """End-to-end flow: collect → merge → train → logs → complete.

    Leaves a COMPLETED training run (named under
    ``INFERENCE_MODEL_TRAIN_RUN_PREFIX``) for the inference and resume-training
    tests to discover.
    """

    # Shared state across test_step* methods within one pytest session
    step_results: dict[str, bool] = {}
    collected_dataset_name: str
    merged_dataset_name: str
    training_name: str
    collected_dataset: Dataset | None = None
    merged_dataset: Dataset | None = None
    job_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.step_results = {}
        cls.collected_dataset_name = unique_name(prefix=FLOW_COLLECTED_DATASET_PREFIX)
        cls.merged_dataset_name = unique_name(prefix=FLOW_MERGED_DATASET_PREFIX)
        cls.training_name = unique_name(prefix=INFERENCE_MODEL_TRAIN_RUN_PREFIX)
        cls.collected_dataset = None
        cls.merged_dataset = None
        cls.job_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        # Phase A: cancel job if not completed to terminate the GCP VM.
        if cls.job_id and not cls.step_results.get(
            "test_step5_assert_training_completed"
        ):
            cancel_incomplete_training_jobs([cls.job_id])

        # Phase A: always clean up test-created datasets.
        try:
            cleanup_training_flow_datasets(
                collected_prefix=FLOW_COLLECTED_DATASET_PREFIX,
                merged_prefix=FLOW_MERGED_DATASET_PREFIX,
            )
        except Exception:
            logger.warning("Failed to clean up training-flow datasets", exc_info=True)
        # NOTE: the training job is intentionally NOT deleted here.
        # The inference test retains it as a known-good model.

    def test_step1_collect_demo_data(self) -> None:
        self.__class__.collected_dataset = collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.collected_dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=COLLECTED_DEMO_EPISODES,
            frequency=FREQUENCY,
        )
        self.__class__.collected_dataset = wait_for_dataset_recording_count(
            dataset_name=self.collected_dataset_name,
            expected_recordings=COLLECTED_DEMO_EPISODES,
        )
        assert len(self.collected_dataset) == COLLECTED_DEMO_EPISODES, (
            f"Expected {COLLECTED_DEMO_EPISODES} recordings,"
            f" got {len(self.collected_dataset)}"
        )
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.collected_dataset)} recordings"
            f" in '{self.collected_dataset_name}'"
        )

    def test_step2_merge_datasets(self) -> None:
        assert self.collected_dataset is not None, "[STEP 1] Did Not Complete"
        shared_dataset = nc.get_dataset(name=SHARED_DATASET_NAME)
        expected_merged_recordings = len(self.collected_dataset) + len(shared_dataset)
        logger.info(
            f"Shared dataset '{SHARED_DATASET_NAME}' has {len(shared_dataset)}"
            f" recordings — expecting {expected_merged_recordings} merged"
        )

        merged = nc.merge_datasets(
            name=self.merged_dataset_name,
            dataset_names=[self.collected_dataset_name, SHARED_DATASET_NAME],
        )
        assert (
            merged.name == self.merged_dataset_name
        ), f"Merged dataset name mismatch: {merged.name!r}"
        self.__class__.merged_dataset = wait_for_dataset_recording_count(
            dataset_name=self.merged_dataset_name,
            expected_recordings=expected_merged_recordings,
        )
        assert self.merged_dataset is not None
        logger.info(f"[STEP 2] [PASSED] Merged Dataset Id={self.merged_dataset.id}")

    def test_step3_start_training(self) -> None:
        assert self.merged_dataset is not None, "[STEP 2] Did Not Complete"
        dataset = nc.get_dataset(name=self.merged_dataset_name)
        assert (
            len(dataset.robot_ids) == 2
        ), f"Expected 2 robots in merged dataset, got {dataset.robot_ids}"
        logger.info(f"Found {len(dataset.robot_ids)} robots: {dataset.robot_ids}")

        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.merged_dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        for robot_id in dataset.robot_ids:
            logger.info(
                f"Input embodiment for robot {robot_id}:\n"
                f"{pprint.pformat(input_desc[robot_id])}"
            )
            logger.info(
                f"Output embodiment for robot {robot_id}:\n"
                f"{pprint.pformat(output_desc[robot_id])}"
            )

        job_data = nc.start_training_run(
            name=self.training_name,
            dataset_name=self.merged_dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
        )
        self.__class__.job_id = job_data["id"]
        logger.info(
            f"[STEP 3] [PASSED] Training Job Started: {self.job_id}"
            f" (name='{self.training_name}')"
        )

    def test_step4_retrieve_logs_while_running(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        # The backend raises 404 (CloudComputeIDNotFoundError) until the GCP
        # VM is registered, so we poll until logs are available.
        running_deadline = time.time() + RUNNING_STATE_TIMEOUT_MINUTES * 60
        while True:
            job_status = nc.get_training_job_status(job_id=self.job_id)
            logger.info(f"Job {self.job_id} status: {job_status} (waiting for RUNNING)")
            if job_status == "RUNNING":
                break
            assert (
                job_status not in TERMINAL_STATES
            ), f"Job reached {job_status} before entering RUNNING state"
            assert time.time() < running_deadline, (
                f"Job did not reach RUNNING state within"
                f" {RUNNING_STATE_TIMEOUT_MINUTES} minutes"
            )
            time.sleep(JOB_STATE_POLL_SECONDS)

        logs_deadline = time.time() + LOGS_AVAILABILITY_TIMEOUT_MINUTES * 60
        logs = None
        while logs is None:
            job_status = nc.get_training_job_status(job_id=self.job_id)
            try:
                logs = nc.get_training_job_logs(job_id=self.job_id, max_entries=50)
            except ValueError:
                # 404 — compute instance not registered yet
                if job_status in TERMINAL_STATES or time.time() > logs_deadline:
                    logger.warning(
                        "Logs unavailable before job completed; skipping assertions",
                        exc_info=True,
                    )
                    break
                time.sleep(JOB_STATE_POLL_SECONDS)

        if logs is not None:
            for field in ("job_id", "logs", "total_entries", "retrieved_at"):
                assert field in logs, f"Missing '{field}' in CloudComputeLogs response"
            assert isinstance(logs["logs"], list)
            assert isinstance(logs["total_entries"], int)
            for entry in logs["logs"]:
                assert "message" in entry, f"Log entry missing 'message': {entry}"
            assert_no_training_log_errors(
                job_id=self.job_id,
                context="Step 4 (training logs while RUNNING)",
            )
            logger.info(
                f"[STEP 4] [PASSED] Retrieved {logs['total_entries']} Log Entries"
            )

    def test_step5_assert_training_completed(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        final_status = wait_for_training(job_id=self.job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Training ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.job_id,
            context="Step 5 (training completion)",
        )
        logger.info(f"[STEP 5] [PASSED] Job {self.job_id} Completed")
