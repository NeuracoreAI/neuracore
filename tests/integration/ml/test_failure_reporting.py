"""Integration test for training-script failure reporting."""

import logging
import os
import sys

from neuracore_types import DataType, EmbodimentDescription

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from tests.integration.ml.shared.dataset import collect_demo_data
from tests.integration.ml.shared.training import (
    build_cross_embodiment_descriptions,
    wait_for_training,
)
from tests.integration.ml.shared.utils import unique_name

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.base_env import BimanualViperXTask

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
)
GRIPPER_NAMES = ["left_gripper", "right_gripper"]
POSE_SENSOR_NAME = "tcp"
LANGUAGE_LABEL = "instruction"


def _indexed_names(names: list[str] | tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


INPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.RGB_IMAGES: {0: NC_CAM_NAME},
    DataType.JOINT_POSITIONS: _indexed_names(names=JOINT_NAMES),
    DataType.LANGUAGE: {0: LANGUAGE_LABEL},
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: _indexed_names(names=GRIPPER_NAMES),
}
OUTPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.JOINT_TARGET_POSITIONS: _indexed_names(names=JOINT_NAMES),
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: _indexed_names(names=GRIPPER_NAMES),
}

INPUT_DATA_TYPES = list(INPUT_EMBODIMENT_DESCRIPTION.keys())
OUTPUT_DATA_TYPES = list(OUTPUT_EMBODIMENT_DESCRIPTION.keys())

ROBOT_NAME = "integration_test_robot"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 20

# A batch_size value that is not "auto" and not parseable as an integer.
# It passes client-side validation (which only checks data types / algorithm
# compatibility, not the batch_size value itself) but causes a ValueError in
# train.py at `batch_size = int(batch_size)`, which happens *after* nc.login()
# so our new top-level error handler can catch and report it.
FAILURE_CNNMLP_CONFIG = {
    "batch_size": "not_a_valid_integer",
    "epochs": 1,
    "output_prediction_horizon": 5,
}


class TestTrainingFailureReporting:
    """Verify that training script failures are correctly reported to the cloud.

    Forces a deliberate runtime failure by submitting a training job whose
    batch_size cannot be parsed as an integer. The error occurs inside
    train.py *after* nc.login() — so the new top-level error handler in
    main() is responsible for catching it and calling
    _try_report_error_to_cloud().

    Assertions:
    1. The job reaches FAILED status (not stuck in RUNNING or PENDING).
    2. The job data returned by the API contains a non-empty 'error' field,
       confirming that the error was propagated back to the server.
    """

    track_step_teardown = True
    all_steps_passed: bool = True
    job_id: str | None = None
    dataset: Dataset | None = None
    dataset_name: str

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="failure_report_test")
        cls.job_id = None
        cls.dataset = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestTrainingFailureReporting teardown cleanup: "
                "one or more steps failed"
            )
            return
        if cls.job_id:
            try:
                nc.delete_training_job(cls.job_id)
            except Exception:
                logger.warning(f"Failed to delete job {cls.job_id}", exc_info=True)
        if cls.dataset:
            try:
                cls.dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete dataset {cls.dataset_name}", exc_info=True
                )

    def test_step1_collect_demo_data(self) -> None:
        self.__class__.dataset = collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=1,
            instance_id=1,
            frequency=FREQUENCY,
        )
        logger.info(
            f"[STEP 1] [PASSED] Collected 1 Recording Into '{self.dataset_name}'"
        )

    def test_step2_submit_failing_job(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=unique_name(prefix="failure_report_job"),
            dataset_name=self.dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=FAILURE_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
        )
        self.__class__.job_id = job_data["id"]
        logger.info(
            f"[STEP 2] [PASSED] Failure-Reporting Test Job Started: {self.job_id}"
        )

    def test_step3_job_reaches_failed_status(self) -> None:
        assert self.job_id is not None, "[STEP 2] Did Not Complete"
        final_status = wait_for_training(job_id=self.job_id, timeout_minutes=30)
        assert final_status == "FAILED", (
            f"Expected FAILED status, got: {final_status!r}.  "
            "The deliberate bad batch_size should have caused a ValueError "
            "in train.py that maps to a FAILED job."
        )
        logger.info(
            f"[STEP 3] [PASSED] Job {self.job_id} Correctly Reached Failed Status"
        )

    def test_step4_error_is_surfaced_in_job_data(self) -> None:
        assert self.job_id is not None, "[STEP 2] Did Not Complete"
        job_detail = nc.get_training_job_data(job_id=self.job_id)
        assert "error" in job_detail, (
            "Job data is missing 'error' field — the server may not have "
            "received the error report from the training script."
        )
        assert job_detail["error"], (
            "The 'error' field in job data is empty — "
            "_try_report_error_to_cloud may not have been called."
        )
        logger.info(
            f"[STEP 4] [PASSED] Error Field Present In Job Data:"
            f" {str(job_detail['error'])[:200]}"
        )
