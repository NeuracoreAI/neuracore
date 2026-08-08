"""End-to-end ML integration lifecycle: collect, train, resume, infer.

A single test class that collects data and trains a model exactly once,
then reuses those artifacts across every later stage (resume, inference,
retraining after dataset mutation) instead of each stage collecting and
training from scratch.

Stages run in this order:
1. Collect demo data.
2. Merge with a shared, pre-existing dataset (cross-embodiment training).
3. Start training on the merged dataset.
4. Retrieve logs while the job is RUNNING.
5. Assert the job COMPLETED.
6-8. Serve the trained model via all three inference paths (direct, local
   server, remote endpoint). These only need the job from steps 3-5, so
   they run before the resume step and never depend on it.
9. Resume the same job for additional epochs. Deliberately placed after
   inference: if resuming is flaky, it must not prevent the already-proven
   dataset/training/inference stages from having run, and nothing later
   depends on it succeeding either.
10-11. Retrain directly on the originally collected dataset after deleting,
   then re-adding, some of its recordings. These only depend on step 1's
   dataset, not on the merged-dataset training or on resume, so they run
   regardless of whether steps 3-9 passed.
"""

import logging
import os
import sys
import time

from neuracore_types import (
    DataType,
    EmbodimentDescription,
    JointData,
    LanguageData,
    ParallelGripperOpenAmountData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from neuracore.core.endpoint import Policy
from tests.integration.ml.shared.dataset import (
    collect_demo_data,
    delete_recording_from_dataset,
    wait_for_dataset_recording_count,
)
from tests.integration.ml.shared.training import (
    TERMINAL_STATES,
    assert_no_training_log_errors,
    build_cross_embodiment_descriptions,
    wait_for_training,
)
from tests.integration.ml.shared.utils import unique_name
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    delete_cloud_robot,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "..", "..", "..", "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.append(_EXAMPLES_DIR)

# ruff: noqa: E402
from common.base_env import BimanualViperXTask
from common.transfer_cube import make_sim_env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embodiment description
# ---------------------------------------------------------------------------

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


# Training/Inference robot (VX300s) embodiment descriptions
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

ROBOT_NAME = "integration_test_robot_lifecycle"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 20

# ---------------------------------------------------------------------------
# Collection / training constants
# ---------------------------------------------------------------------------

COLLECTED_DATASET_PREFIX = "collected"
MERGED_DATASET_PREFIX = "merged"
TRAIN_RUN_PREFIX = "ml_integration_lifecycle"
DELETION_TRAIN_RUN_PREFIX = "ml_integration_lifecycle_after_deletion"
ADDITION_TRAIN_RUN_PREFIX = "ml_integration_lifecycle_after_addition"

SHARED_DATASET_NAME = "NYU ROT"
COLLECT_NUM_EPISODES = 5
DATA_DELETION_RECORDINGS_TO_REMOVE = 2
DATA_ADDITION_RECORDINGS_TO_ADD = 2

JOB_STATE_POLL_SECONDS = 30
RUNNING_STATE_TIMEOUT_MINUTES = 10
LOGS_AVAILABILITY_TIMEOUT_MINUTES = 10

MAIN_CNNMLP_CONFIG = {
    "batch_size": 64,
    "epochs": 1,
    "output_prediction_horizon": 5,
}
MUTATION_CNNMLP_CONFIG = {
    "batch_size": 16,
    "epochs": 1,
    "output_prediction_horizon": 5,
}

# ---------------------------------------------------------------------------
# Inference constants and helpers
# ---------------------------------------------------------------------------

ENDPOINT_POLL_SECONDS = 20
ENDPOINT_TIMEOUT_MINUTES = 30
ENDPOINT_TTL_SECONDS = 60 * 30
LOCAL_SERVER_PORT = 8181
MUJOCO_ROBOT_NAME = "Mujoco VX300s"


def wait_for_endpoint(
    endpoint_id: str, timeout_minutes: int = ENDPOINT_TIMEOUT_MINUTES
) -> str:
    deadline = time.time() + timeout_minutes * 60
    while True:
        status = nc.get_endpoint_status(endpoint_id=endpoint_id)
        logger.info(f"Endpoint {endpoint_id}: {status}")
        if status != "creating":
            return status
        assert time.time() < deadline, (
            f"Endpoint {endpoint_id} did not become "
            f"active within {timeout_minutes} minutes"
        )
        time.sleep(ENDPOINT_POLL_SECONDS)


def make_sync_point(
    obs,
    *,
    joint_names: tuple[str, ...] | list[str],
    gripper_names: list[str],
    language_label: str,
    nc_cam_name: str,
    mj_cam_name: str,
) -> SynchronizedPoint:
    return SynchronizedPoint(
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=obs.qpos[name]) for name in joint_names
            },
            DataType.RGB_IMAGES: {
                nc_cam_name: RGBCameraData(frame=obs.cameras[mj_cam_name].rgb)
            },
            DataType.LANGUAGE: {language_label: LanguageData(text="pick and place")},
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: {
                name: ParallelGripperOpenAmountData(
                    open_amount=float(obs.gripper_open_amounts[name])
                )
                for name in gripper_names
            },
        }
    )


def run_policy_inference(
    policy: Policy,
    *,
    joint_names: tuple[str, ...] | list[str],
    gripper_names: list[str],
    language_label: str,
    nc_cam_name: str,
    mj_cam_name: str,
    output_data_types: list[DataType],
) -> None:
    """Run inference via both sync-point and logging-function paths."""
    try:
        env = make_sim_env(seed=42)
        obs = env.reset()

        logger.info("Running sync-point inference (Path 1)")
        predictions = policy.predict(
            sync_point=make_sync_point(
                obs=obs,
                joint_names=joint_names,
                gripper_names=gripper_names,
                language_label=language_label,
                nc_cam_name=nc_cam_name,
                mj_cam_name=mj_cam_name,
            ),
            timeout=30,
        )
        for data_type in output_data_types:
            assert data_type in predictions, (
                f"Expected {data_type.value} in local "
                f"server output, got: {list(predictions.keys())}"
            )
        logger.info(f"Path 1 passed — output keys: {[k.value for k in predictions]}")

        logger.info("Running logging-function inference (Path 2)")
        nc.log_joint_positions(
            positions={name: float(obs.qpos[name]) for name in joint_names}
        )
        nc.log_language(name=language_label, language="pick and place")
        nc.log_parallel_gripper_open_amounts(
            values={
                name: float(obs.gripper_open_amounts[name]) for name in gripper_names
            }
        )
        nc.log_rgb(name=nc_cam_name, rgb=obs.cameras[mj_cam_name].rgb)
        predictions = policy.predict(timeout=30)
        for data_type in output_data_types:
            assert data_type in predictions, (
                f"Expected {data_type.value} in local "
                f"server output, got: {list(predictions.keys())}"
            )
        logger.info(f"Path 2 passed — output keys: {[k.value for k in predictions]}")
    finally:
        policy.disconnect()


class TestMLLifecycle:
    """Collect once, train once, then resume/infer/retrain off that state."""

    track_step_teardown = True
    all_steps_passed: bool = True

    collected_dataset_name: str
    merged_dataset_name: str
    training_name: str
    deletion_training_name: str
    addition_training_name: str

    collected_dataset: Dataset | None = None
    merged_dataset: Dataset | None = None
    job_id: str | None = None
    endpoint_id: str | None = None
    deletion_relaunch_job_id: str | None = None
    addition_relaunch_job_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.all_steps_passed = True
        cls.collected_dataset_name = unique_name(prefix=COLLECTED_DATASET_PREFIX)
        cls.merged_dataset_name = unique_name(prefix=MERGED_DATASET_PREFIX)
        cls.training_name = unique_name(prefix=TRAIN_RUN_PREFIX)
        cls.deletion_training_name = unique_name(prefix=DELETION_TRAIN_RUN_PREFIX)
        cls.addition_training_name = unique_name(prefix=ADDITION_TRAIN_RUN_PREFIX)
        cls.collected_dataset = None
        cls.merged_dataset = None
        cls.job_id = None
        cls.endpoint_id = None
        cls.deletion_relaunch_job_id = None
        cls.addition_relaunch_job_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if not cls.all_steps_passed:
            logger.warning(
                "Skipping TestMLLifecycle teardown cleanup: one or more steps failed"
            )
            return
        if cls.endpoint_id:
            try:
                nc.delete_endpoint(cls.endpoint_id)
            except Exception:
                logger.warning(
                    f"Failed to delete endpoint {cls.endpoint_id}", exc_info=True
                )
        for job_id in (
            cls.job_id,
            cls.deletion_relaunch_job_id,
            cls.addition_relaunch_job_id,
        ):
            if job_id is None:
                continue
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete training job {job_id}", exc_info=True)
        for dataset in (cls.merged_dataset, cls.collected_dataset):
            if dataset is None:
                continue
            try:
                dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete dataset {dataset.name}", exc_info=True
                )
        delete_cloud_robot(ROBOT_NAME)
        delete_cloud_robot(MUJOCO_ROBOT_NAME)

    # -- 1. Collect ----------------------------------------------------

    def test_step01_collect_demo_data(self) -> None:
        self.__class__.collected_dataset = collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.collected_dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=COLLECT_NUM_EPISODES,
            frequency=FREQUENCY,
        )
        self.__class__.collected_dataset = wait_for_dataset_recording_count(
            dataset_name=self.collected_dataset_name,
            expected_recordings=COLLECT_NUM_EPISODES,
        )
        assert len(self.collected_dataset) == COLLECT_NUM_EPISODES, (
            f"Expected {COLLECT_NUM_EPISODES} recordings,"
            f" got {len(self.collected_dataset)}"
        )
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.collected_dataset)} recordings"
            f" in '{self.collected_dataset_name}'"
        )

    # -- 2. Merge --------------------------------------------------------

    def test_step02_merge_datasets(self) -> None:
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
        logger.info(f"[STEP 2] [PASSED] Merged Dataset Id={self.merged_dataset.id}")

    # -- 3. Train ----------------------------------------------------------

    def test_step03_start_training(self) -> None:
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
        job_data = nc.start_training_run(
            name=self.training_name,
            dataset_name=self.merged_dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=MAIN_CNNMLP_CONFIG,
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

    def test_step04_retrieve_logs_while_running(self) -> None:
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

    def test_step05_training_completes(self) -> None:
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

    # -- 6-8. Inference off the same job -----------------------------------

    def test_step06_direct_inference(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        nc.connect_robot(robot_name=MUJOCO_ROBOT_NAME)
        policy = nc.policy(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            train_run_name=self.training_name,
        )
        run_policy_inference(
            policy=policy,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            mj_cam_name=MJ_CAM_NAME,
            output_data_types=OUTPUT_DATA_TYPES,
        )
        logger.info(
            f"[STEP 6] [PASSED] Direct In-Process Inference Succeeded"
            f" (run name='{self.training_name}' id={self.job_id})"
        )

    def test_step07_local_server_inference(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        policy = nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            train_run_name=self.training_name,
            port=LOCAL_SERVER_PORT,
        )
        run_policy_inference(
            policy=policy,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            mj_cam_name=MJ_CAM_NAME,
            output_data_types=OUTPUT_DATA_TYPES,
        )
        logger.info(
            f"[STEP 7] [PASSED] Local Server Inference Succeeded"
            f" (run name='{self.training_name}' id={self.job_id})"
        )

    def test_step08_deploy_remote_endpoint(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        endpoint_name = unique_name(prefix="lifecycle_endpoint")
        endpoint_data = nc.deploy_model(
            job_id=self.job_id,
            name=endpoint_name,
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            ttl=ENDPOINT_TTL_SECONDS,
        )
        self.__class__.endpoint_id = endpoint_data["id"]
        assert self.endpoint_id is not None
        final_status = wait_for_endpoint(endpoint_id=self.endpoint_id)
        assert (
            final_status == "active"
        ), f"Endpoint did not become active, status: {final_status!r}"
        logger.info(
            f"[STEP 8] [PASSED] Endpoint {self.endpoint_id} Is Active"
            f" (run name='{self.training_name}' id={self.job_id})"
        )

    # -- 9. Resume -----------------------------------------------------

    def test_step09_resume_training(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        job_id = self.job_id
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
            f"[STEP 9] [PASSED] Resumed Job Completed At Epoch"
            f" {resumed_data.get('epoch')}"
        )

    # -- 10-11. Retrain after mutating the original collected dataset -----

    def test_step10_retrain_after_data_deletion(self) -> None:
        assert self.collected_dataset is not None, "[STEP 1] Did Not Complete"
        dataset = nc.get_dataset(name=self.collected_dataset_name)
        assert len(dataset) == COLLECT_NUM_EPISODES, (
            f"Expected {COLLECT_NUM_EPISODES} recordings before deletion, "
            f"got {len(dataset)}"
        )

        recordings_to_delete = [
            dataset[index] for index in range(DATA_DELETION_RECORDINGS_TO_REMOVE)
        ]
        for recording in recordings_to_delete:
            logger.info(
                f"Deleting recording {recording.id!r} ({recording.name!r})"
                f" from dataset {self.collected_dataset_name!r}"
            )
            delete_recording_from_dataset(dataset=dataset, recording=recording)

        expected_after_deletion = (
            COLLECT_NUM_EPISODES - DATA_DELETION_RECORDINGS_TO_REMOVE
        )
        self.__class__.collected_dataset = wait_for_dataset_recording_count(
            dataset_name=self.collected_dataset_name,
            expected_recordings=expected_after_deletion,
        )
        logger.info(
            f"Deleted {DATA_DELETION_RECORDINGS_TO_REMOVE} recordings; "
            f"{len(self.collected_dataset)} remain in '{self.collected_dataset_name}'"
        )

        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.collected_dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=self.deletion_training_name,
            dataset_name=self.collected_dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=MUTATION_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
        )
        self.__class__.deletion_relaunch_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.deletion_relaunch_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Training after deletion ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.deletion_relaunch_job_id,
            context="Step 10 (training after data deletion)",
        )
        logger.info(
            f"[STEP 10] [PASSED] Training Job {self.deletion_relaunch_job_id}"
            f" Completed On {len(self.collected_dataset)}-Recording Dataset"
        )

    def test_step11_retrain_after_data_addition(self) -> None:
        assert self.collected_dataset is not None, "[STEP 1] Did Not Complete"
        dataset = nc.get_dataset(name=self.collected_dataset_name)
        recordings_before_addition = len(dataset)

        collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.collected_dataset_name,
            joint_names=JOINT_NAMES,
            gripper_names=GRIPPER_NAMES,
            language_label=LANGUAGE_LABEL,
            nc_cam_name=NC_CAM_NAME,
            pose_sensor_name=POSE_SENSOR_NAME,
            num_episodes=DATA_ADDITION_RECORDINGS_TO_ADD,
            frequency=FREQUENCY,
        )

        expected_after_addition = (
            recordings_before_addition + DATA_ADDITION_RECORDINGS_TO_ADD
        )
        self.__class__.collected_dataset = wait_for_dataset_recording_count(
            dataset_name=self.collected_dataset_name,
            expected_recordings=expected_after_addition,
        )
        logger.info(
            f"Added {DATA_ADDITION_RECORDINGS_TO_ADD} recordings; "
            f"{len(self.collected_dataset)} now in '{self.collected_dataset_name}'"
        )

        input_desc, output_desc = build_cross_embodiment_descriptions(
            dataset=self.collected_dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=self.addition_training_name,
            dataset_name=self.collected_dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=MUTATION_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=input_desc,
            output_cross_embodiment_description=output_desc,
        )
        self.__class__.addition_relaunch_job_id = job_data["id"]
        final_status = wait_for_training(job_id=self.addition_relaunch_job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Training after addition ended with non-COMPLETED status: {final_status}"
        assert_no_training_log_errors(
            job_id=self.addition_relaunch_job_id,
            context="Step 11 (training after data addition)",
        )
        logger.info(
            f"[STEP 11] [PASSED] Training Job {self.addition_relaunch_job_id}"
            f" Completed On {len(self.collected_dataset)}-Recording Dataset"
        )
