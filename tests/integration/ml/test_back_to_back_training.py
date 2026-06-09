"""Integration test for launching multiple training jobs back-to-back."""

import logging

from neuracore_types import DataType

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from tests.integration.ml.shared.data_collection import (
    collect_demo_data,
    wait_for_dataset_recording_count,
)
from tests.integration.ml.shared.training import (
    build_cross_embodiment_descriptions,
    wait_for_all_training,
)
from tests.integration.ml.shared.utils import unique_name
from tests.integration.ml.test_training_flow import (
    GRIPPER_NAMES,
    JOINT_NAMES,
    LANGUAGE_LABEL,
    NC_CAM_NAME,
    POSE_SENSOR_NAME,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ROBOT_NAME = "integration_test_robot"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
BACK_TO_BACK_NUM_EPISODES = 25
BACK_TO_BACK_EPISODE_LENGTH_MULTIPLIER = 5
BACK_TO_BACK_FREQUENCY = 100
BACK_TO_BACK_NUM_CAMERAS = 3
BACK_TO_BACK_NUM_JOBS = 2
BACK_TO_BACK_CNNMLP_CONFIG = {
    "batch_size": 16,
    "epochs": 1,
    "output_prediction_horizon": 5,
}


class TestBackToBackTraining:
    """Launch multiple training jobs back-to-back against the same dataset."""

    track_step_teardown = True
    all_steps_passed: bool = True
    dataset: Dataset | None = None
    dataset_name: str
    job_ids: list[str]
    training_names: list[str]

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="back_to_back_training")
        cls.training_names = [
            unique_name(prefix="ml_integration_back_to_back")
            for _ in range(BACK_TO_BACK_NUM_JOBS)
        ]
        cls.job_ids = []
        cls.dataset = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestBackToBackTraining teardown cleanup: "
                "one or more steps failed"
            )
            return
        for job_id in cls.job_ids:
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
            num_episodes=BACK_TO_BACK_NUM_EPISODES,
            instance_id=2,
            episode_length_multiplier=BACK_TO_BACK_EPISODE_LENGTH_MULTIPLIER,
            num_cameras=BACK_TO_BACK_NUM_CAMERAS,
            frequency=BACK_TO_BACK_FREQUENCY,
        )
        self.__class__.dataset = wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=BACK_TO_BACK_NUM_EPISODES,
        )
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.dataset)} Recordings"
            f" Into '{self.dataset_name}'"
        )

    def test_step2_submit_back_to_back_jobs(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=[
                DataType.JOINT_POSITIONS,
                DataType.RGB_IMAGES,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ],
            output_types=[
                DataType.JOINT_TARGET_POSITIONS,
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
            ],
        )
        for train_run_name in self.training_names:
            job_data = nc.start_training_run(
                name=train_run_name,
                dataset_name=self.dataset_name,
                algorithm_name="CNNMLP",
                algorithm_config=BACK_TO_BACK_CNNMLP_CONFIG,
                gpu_type=GPU_TYPE,
                num_gpus=NUM_GPUS,
                frequency=BACK_TO_BACK_FREQUENCY,
                input_cross_embodiment_description=input_desc,
                output_cross_embodiment_description=output_desc,
            )
            self.__class__.job_ids.append(job_data["id"])

        assert (
            len(set(self.job_ids)) == BACK_TO_BACK_NUM_JOBS
        ), f"Expected distinct training jobs, got duplicate job ids: {self.job_ids}"
        logger.info(
            f"[STEP 2] [PASSED] Submitted {len(self.job_ids)} Back-To-Back Jobs:"
            f" {self.job_ids}"
        )

    def test_step3_all_jobs_complete(self) -> None:
        final_statuses = wait_for_all_training(job_ids=self.job_ids, timeout_minutes=60)
        for job_id, status in final_statuses.items():
            assert status == "COMPLETED", (
                f"Back-to-back training job {job_id} ended with "
                f"non-COMPLETED status: {status}"
            )
        logger.info(
            f"[STEP 3] [PASSED] All {len(self.job_ids)} Back-To-Back Jobs Completed"
        )
