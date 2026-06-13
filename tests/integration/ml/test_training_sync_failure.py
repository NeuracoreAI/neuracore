"""Integration test: a cloud training job whose synchronization step fails.

A training job synchronizes its dataset (the ``PREPARING_DATA`` phase) before any
training VM runs. A *permanent* synchronization failure during that phase must
move the job to ``FAILED`` and surface the synchronization reason verbatim in the
job's ``error`` field, without a training VM ever starting.

Two distinct permanent failures are exercised against a single collected dataset,
asserting the backend returns the *correct* reason for each:

* **too sparse** — a frequency far above the data rate with duplicates
  disallowed (``SynchronizationDataTypeTooSparseError``), and
* **exceeds max delta** — a max delay so small no data point can match the
  reference timestamps (``SynchronizationDataTypeExceedsMaxDeltaError``).

"""

import logging
from dataclasses import dataclass

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
from tests.integration.ml.shared.data_collection import collect_demo_data
from tests.integration.ml.shared.training import (
    build_cross_embodiment_descriptions,
    wait_for_all_training,
)
from tests.integration.ml.shared.utils import integration_ml_job_name, unique_name

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# A frequency far above the data rate. Reference timestamps then fall between the
# real data points, which both starves the no-duplicates case (too sparse) and
# exposes the tiny-max-delay case (no point within delta). Shared by both cases
# so they differ only in the parameter under test.
SYNC_FAILURE_FREQUENCY = 1000

SYNC_FAILURE_TIMEOUT_MINUTES = 30

# batch_size is irrelevant here — the job never reaches training — but it must be
# a value that passes client-side validation.
SYNC_FAILURE_CNNMLP_CONFIG = {
    "batch_size": "auto",
    "epochs": 1,
    "output_prediction_horizon": 5,
}


@dataclass(frozen=True)
class SyncFailureCase:
    """A permanent synchronization failure and the message it must surface.

    Attributes:
        id: identifier used in job names, logs and the job-id mapping.
        max_delay_s: Max synchronization delay (seconds).
        allow_duplicates: Whether duplicate points are allowed when syncing.
        expected_marker: Substring the backend's failure reason must contain.
            Specific to this failure class, so it also rules out the *other*
            failure (and any unrelated error) reaching FAILED.
    """

    id: str
    max_delay_s: float
    allow_duplicates: bool
    expected_marker: str


SYNC_FAILURE_CASES = (
    # max_delay large enough to pass the max-delta check, so the no-duplicates
    # starvation is what fails: SynchronizationDataTypeTooSparseError.
    SyncFailureCase(
        id="too_sparse",
        max_delay_s=1.0,
        allow_duplicates=False,
        expected_marker="too sparse",
    ),
    # max_delay so small no data point matches a reference timestamp; the
    # max-delta check fires first: SynchronizationDataTypeExceedsMaxDeltaError.
    SyncFailureCase(
        id="exceeds_max_delta",
        max_delay_s=1e-9,
        allow_duplicates=True,
        expected_marker="within max delta",
    ),
)


class TestTrainingSyncFailure:
    """Verify cloud jobs with an impossible data-sync report the right reason.

    One short dataset is collected once and shared by every failure case. Each
    case starts its own training job (with its own derived synced dataset), so
    reusing the source recordings across cases is safe and avoids re-collecting
    per parameter. The requested synchronization is impossible to satisfy, so
    the backend permanently fails each job during PREPARING_DATA, before any
    training VM runs.

    Assertions:
    1. Every case's job reaches FAILED status (not stuck, not COMPLETED).
    2. Each job's surfaced 'error' field carries the failure reason specific
       to its case — and therefore not the other case's.
    """

    track_step_teardown = True
    all_steps_passed: bool = True
    dataset: Dataset | None = None
    dataset_name: str
    # case id -> training job id
    job_ids: dict[str, str]

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.dataset_name = unique_name(prefix="sync_failure_test")
        cls.dataset = None
        cls.job_ids = {}
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestTrainingSyncFailure teardown cleanup: "
                "one or more steps failed"
            )
            return
        for job_id in cls.job_ids.values():
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete job {job_id}", exc_info=True)
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

    def test_step2_submit_failing_jobs(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        for case in SYNC_FAILURE_CASES:
            job_data = nc.start_training_run(
                name=integration_ml_job_name(f"Training Sync Failure {case.id}"),
                dataset_name=self.dataset_name,
                algorithm_name="CNNMLP",
                algorithm_config=SYNC_FAILURE_CNNMLP_CONFIG,
                gpu_type=GPU_TYPE,
                num_gpus=NUM_GPUS,
                frequency=SYNC_FAILURE_FREQUENCY,
                max_delay_s=case.max_delay_s,
                allow_duplicates=case.allow_duplicates,
                input_cross_embodiment_description=input_desc,
                output_cross_embodiment_description=output_desc,
            )
            self.__class__.job_ids[case.id] = job_data["id"]

        assert len(set(self.job_ids.values())) == len(
            SYNC_FAILURE_CASES
        ), f"Expected distinct jobs per case, got duplicate ids: {self.job_ids}"
        logger.info(
            f"[STEP 2] [PASSED] Submitted {len(self.job_ids)} Sync-Failure Jobs:"
            f" {self.job_ids}"
        )

    def test_step3_jobs_reach_failed_status(self) -> None:
        assert self.job_ids, "[STEP 2] Did Not Complete"
        final_statuses = wait_for_all_training(
            job_ids=list(self.job_ids.values()),
            timeout_minutes=SYNC_FAILURE_TIMEOUT_MINUTES,
        )
        for case in SYNC_FAILURE_CASES:
            job_id = self.job_ids[case.id]
            status = final_statuses[job_id]
            assert status == "FAILED", (
                f"[{case.id}] Expected FAILED status, got: {status!r}. "
                "The impossible synchronization request should permanently fail "
                "the job during PREPARING_DATA."
            )
        logger.info(
            f"[STEP 3] [PASSED] All {len(self.job_ids)} Jobs Reached FAILED On Sync"
        )

    def test_step4_correct_reason_surfaced(self) -> None:
        assert self.job_ids, "[STEP 2] Did Not Complete"
        for case in SYNC_FAILURE_CASES:
            job_id = self.job_ids[case.id]
            job_detail = nc.get_training_job_data(job_id=job_id)
            error_message = str(job_detail.get("error") or "")
            assert error_message, (
                f"[{case.id}] The 'error' field in job data is empty — the "
                "synchronization failure reason was not surfaced back to the server."
            )
            assert case.expected_marker in error_message, (
                f"[{case.id}] Expected {case.expected_marker!r} in the surfaced sync "
                f"failure reason, got: {error_message!r}"
            )
            logger.info(
                f"[{case.id}] [STEP 4] [PASSED] Correct Reason Surfaced:"
                f" {error_message[:200]}"
            )
