"""Integration tests for continuing training.

- ``TestResumeTraining``: discovers the model by scanning for the latest
  COMPLETED training run under ``INFERENCE_MODEL_TRAIN_RUN_PREFIX`` (the run
  produced by the end-to-end flow), resumes it for additional epochs, and
  verifies the resume metadata and completion. If no completed run exists it
  errors out. The resumed run remains COMPLETED afterwards, so it stays
  available as a known-good model for the inference test and future sessions.
  Teardown deletes the collected and merged datasets created by the training
  flow test.
- ``TestResumeTrainingAfterDataDeletion``: trains on a freshly collected
  dataset, deletes some of its recordings, then relaunches training to confirm
  a changed dataset can be retrained successfully.
- ``TestResumeTrainingAfterDataAddition``: trains on a freshly collected
  dataset, adds more recordings to it, then relaunches training to confirm a
  grown dataset can be retrained successfully.
"""

import logging

import neuracore as nc
from neuracore.core.auth import get_auth
from neuracore.core.const import API_URL
from neuracore.core.data.dataset import Dataset
from neuracore.core.data.recording import Recording
from neuracore.core.utils.http_session import thread_local_session
from tests.integration.ml.shared.constants import (
    GRIPPER_NAMES,
    INPUT_DATA_TYPES,
    JOINT_NAMES,
    LANGUAGE_LABEL,
    NC_CAM_NAME,
    OUTPUT_DATA_TYPES,
    POSE_SENSOR_NAME,
)
from tests.integration.ml.shared.data_collection import (
    collect_demo_data,
    wait_for_dataset_recording_count,
)
from tests.integration.ml.shared.training import (
    assert_no_training_log_errors,
    build_cross_embodiment_descriptions,
    wait_for_training,
)
from tests.integration.ml.shared.utils import (
    SelectedRun,
    cleanup_training_flow_datasets,
    integration_ml_job_name,
    resolve_latest_completed_run,
    unique_name,
)
from tests.integration.ml.test_training_flow import (
    FLOW_COLLECTED_DATASET_PREFIX,
    FLOW_MERGED_DATASET_PREFIX,
    INFERENCE_MODEL_TRAIN_RUN_PREFIX,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ROBOT_NAME = "integration_test_robot"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 20

DATA_DELETION_NUM_EPISODES = 5
DATA_DELETION_RECORDINGS_TO_REMOVE = 2
DATA_DELETION_INSTANCE_ID = 3
DATA_DELETION_CNNMLP_CONFIG = {
    "batch_size": 16,
    "epochs": 1,
    "output_prediction_horizon": 5,
}
DATA_ADDITION_NUM_EPISODES = 3
DATA_ADDITION_RECORDINGS_TO_ADD = 2
DATA_ADDITION_INSTANCE_ID = 4
DATA_ADDITION_CNNMLP_CONFIG = {
    "batch_size": 16,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


class TestResumeTraining:
    """Resume the latest COMPLETED training run and verify resume behavior."""

    track_step_teardown = True
    all_steps_passed: bool = True
    selected_run: SelectedRun | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.selected_run = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestResumeTraining teardown cleanup: "
                "one or more steps failed"
            )
            return
        try:
            cleanup_training_flow_datasets(
                collected_prefix=FLOW_COLLECTED_DATASET_PREFIX,
                merged_prefix=FLOW_MERGED_DATASET_PREFIX,
            )
        except Exception:
            logger.warning("Failed to clean up training-flow datasets", exc_info=True)

    def test_step1_resolve_trained_model(self) -> None:
        self.__class__.selected_run = resolve_latest_completed_run(
            INFERENCE_MODEL_TRAIN_RUN_PREFIX
        )
        logger.info(
            f"[STEP 1] [PASSED] Resuming training run name='{self.selected_run.name}'"
            f" id={self.selected_run.id}"
        )

    def test_step2_resume_training(self) -> None:
        assert self.selected_run is not None, "[STEP 1] Did Not Complete"
        job_id = self.selected_run.id
        initial_epoch = nc.get_training_job_data(job_id=job_id).get("epoch", 0)
        resumed_job = nc.resume_training_run(job_id=job_id, additional_epochs=1)
        logger.info(f"Resume response: {resumed_job}")

        assert resumed_job["status"] in {
            "PENDING",
            "RUNNING",
        }, f"Expected PENDING/RUNNING after resume, got: {resumed_job['status']!r}"
        assert resumed_job.get(
            "resume_points"
        ), "Expected non-empty resume_points after resume"
        assert (
            resumed_job.get("resumed_at") is not None
        ), "Expected resumed_at to be set after resume"

        final_resumed_status = wait_for_training(job_id=job_id)
        assert (
            final_resumed_status == "COMPLETED"
        ), f"Resumed training ended with non-COMPLETED status: {final_resumed_status}"

        resumed_data = nc.get_training_job_data(job_id=job_id)
        assert (resumed_data.get("epoch") or 0) > initial_epoch, (
            f"Expected epoch to increase after resume, "
            f"was {initial_epoch}, now {resumed_data.get('epoch')}"
        )
        assert (
            resumed_data.get("previous_training_time") is not None
        ), "Expected previous_training_time to be set after resume"
        assert_no_training_log_errors(
            job_id=job_id,
            context="Resume training completion",
        )
        logger.info(
            f"[STEP 2] [PASSED] Resumed Job Completed At Epoch"
            f" {resumed_data.get('epoch')}"
        )


def delete_recording_from_dataset(dataset: Dataset, recording: Recording) -> None:
    """Remove a recording from a dataset via the platform API."""
    session = thread_local_session()
    response = session.delete(
        f"{API_URL}/org/{dataset.org_id}/datasets/{dataset.id}/recording/{recording.id}",
        headers=get_auth().get_headers(),
    )
    response.raise_for_status()


class TestResumeTrainingAfterDataDeletion:
    """Train on a dataset, delete some recordings, then relaunch training."""

    track_step_teardown = True
    all_steps_passed: bool = True
    dataset: Dataset | None = None
    dataset_name: str
    training_name: str
    first_job_id: str | None = None
    relaunched_job_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="train_after_data_deletion")
        cls.training_name = integration_ml_job_name(
            "Resume Training After Data Deletion"
        )
        cls.dataset = None
        cls.first_job_id = None
        cls.relaunched_job_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestResumeTrainingAfterDataDeletion teardown cleanup: "
                "one or more steps failed"
            )
            return
        for job_id in (cls.first_job_id, cls.relaunched_job_id):
            if job_id is None:
                continue
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete training job {job_id}", exc_info=True)
        if cls.dataset:
            try:
                cls.dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete dataset {cls.dataset_name}", exc_info=True
                )

    def test_step1_collect_demo_data(self) -> None:
        collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=DATA_DELETION_NUM_EPISODES,
            instance_id=DATA_DELETION_INSTANCE_ID,
            frequency=FREQUENCY,
        )
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=DATA_DELETION_NUM_EPISODES,
        )
        assert len(self.dataset) == DATA_DELETION_NUM_EPISODES
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.dataset)} Recordings"
            f" Into '{self.dataset_name}'"
        )

    def test_step2_first_training_completes(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=self.training_name,
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=DATA_DELETION_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
        )
        self.__class__.first_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.first_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Initial training ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.first_job_id,
            context="Step 2 (initial training completion)",
        )
        logger.info(
            f"[STEP 2] [PASSED] Initial Training Job {self.first_job_id} Completed"
        )

    def test_step3_delete_some_recordings(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        assert self.first_job_id is not None, "[STEP 2] Did Not Complete"
        dataset = nc.get_dataset(name=self.dataset_name)
        assert len(dataset) == DATA_DELETION_NUM_EPISODES, (
            f"Expected {DATA_DELETION_NUM_EPISODES} recordings before deletion, "
            f"got {len(dataset)}"
        )

        recordings_to_delete = [
            dataset[index] for index in range(DATA_DELETION_RECORDINGS_TO_REMOVE)
        ]
        for recording in recordings_to_delete:
            logger.info(
                f"Deleting recording {recording.id!r} ({recording.name!r})"
                f" from dataset {self.dataset_name!r}"
            )
            delete_recording_from_dataset(dataset=dataset, recording=recording)

        expected_remaining = (
            DATA_DELETION_NUM_EPISODES - DATA_DELETION_RECORDINGS_TO_REMOVE
        )
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=expected_remaining,
        )
        assert len(self.dataset) == expected_remaining
        logger.info(
            f"[STEP 3] [PASSED] Deleted {DATA_DELETION_RECORDINGS_TO_REMOVE} "
            f"Recordings; {len(self.dataset)} Remain In '{self.dataset_name}'"
        )

    def test_step4_relaunch_training_completes(self) -> None:
        assert self.dataset is not None, "[STEP 3] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=self.training_name,
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=DATA_DELETION_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
            name_auto_increment=True,
        )
        self.__class__.relaunched_job_id = job_data["id"]
        assert (
            self.relaunched_job_id != self.first_job_id
        ), "Relaunched training should create a new job id"

        final_status = wait_for_training(job_id=self.relaunched_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Relaunched training ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.relaunched_job_id,
            context="Step 4 (relaunched training completion)",
        )
        logger.info(
            f"[STEP 4] [PASSED] Relaunched Training Job"
            f" {self.relaunched_job_id} Completed"
        )


class TestResumeTrainingAfterDataAddition:
    """Train on a dataset, add more recordings, then relaunch training."""

    track_step_teardown = True
    all_steps_passed: bool = True
    dataset: Dataset | None = None
    dataset_name: str
    training_name: str
    first_job_id: str | None = None
    relaunched_job_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="train_after_data_addition")
        cls.training_name = integration_ml_job_name(
            "Resume Training After Data Addition"
        )
        cls.dataset = None
        cls.first_job_id = None
        cls.relaunched_job_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestResumeTrainingAfterDataAddition teardown cleanup: "
                "one or more steps failed"
            )
            return
        for job_id in (cls.first_job_id, cls.relaunched_job_id):
            if job_id is None:
                continue
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete training job {job_id}", exc_info=True)
        if cls.dataset:
            try:
                cls.dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete dataset {cls.dataset_name}", exc_info=True
                )

    def test_step1_collect_demo_data(self) -> None:
        collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=DATA_ADDITION_NUM_EPISODES,
            instance_id=DATA_ADDITION_INSTANCE_ID,
            frequency=FREQUENCY,
        )
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=DATA_ADDITION_NUM_EPISODES,
        )
        assert len(self.dataset) == DATA_ADDITION_NUM_EPISODES
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.dataset)} Recordings"
            f" Into '{self.dataset_name}'"
        )

    def test_step2_first_training_completes(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=self.training_name,
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=DATA_ADDITION_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
        )
        self.__class__.first_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.first_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Initial training ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.first_job_id,
            context="Step 2 (initial training completion)",
        )
        logger.info(
            f"[STEP 2] [PASSED] Initial Training Job {self.first_job_id} Completed"
        )

    def test_step3_add_more_recordings(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        assert self.first_job_id is not None, "[STEP 2] Did Not Complete"
        dataset = nc.get_dataset(name=self.dataset_name)
        assert len(dataset) == DATA_ADDITION_NUM_EPISODES, (
            f"Expected {DATA_ADDITION_NUM_EPISODES} recordings before addition, "
            f"got {len(dataset)}"
        )

        collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=DATA_ADDITION_RECORDINGS_TO_ADD,
            instance_id=DATA_ADDITION_INSTANCE_ID,
            frequency=FREQUENCY,
        )

        expected_total = DATA_ADDITION_NUM_EPISODES + DATA_ADDITION_RECORDINGS_TO_ADD
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=expected_total,
        )
        assert len(self.dataset) == expected_total
        logger.info(
            f"[STEP 3] [PASSED] Added {DATA_ADDITION_RECORDINGS_TO_ADD} Recordings;"
            f" {len(self.dataset)} Now In '{self.dataset_name}'"
        )

    def test_step4_relaunch_training_completes(self) -> None:
        assert self.dataset is not None, "[STEP 3] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=self.training_name,
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=DATA_ADDITION_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
            name_auto_increment=True,
        )
        self.__class__.relaunched_job_id = job_data["id"]
        assert (
            self.relaunched_job_id != self.first_job_id
        ), "Relaunched training should create a new job id"

        final_status = wait_for_training(job_id=self.relaunched_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Relaunched training ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.relaunched_job_id,
            context="Step 4 (relaunched training completion)",
        )
        logger.info(
            f"[STEP 4] [PASSED] Relaunched Training Job"
            f" {self.relaunched_job_id} Completed"
        )
