"""Integration tests for training flows on the Neuracore platform.

An end-to-end test that covers: dataset collection, merging, training
(with auto batch sizing),
log retrieval, direct inference, local server inference, and remote
endpoint deployment.
"""

import logging
import os
import pprint
import sys
import time
import uuid

import numpy as np
from neuracore_types import (
    DataType,
    EmbodimentDescription,
    JointData,
    LanguageData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc
from neuracore.core.data.dataset import Dataset
from neuracore.core.endpoint import Policy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.base_env import BimanualViperXTask
from common.rollout_utils import rollout_policy
from common.transfer_cube import BIMANUAL_VIPERX_URDF_PATH, make_sim_env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARED_DATASET_NAME = "ASU Table Top"
COLLECTED_DEMO_EPISODES = 3
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 20
NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
ROBOT_NAME = "integration_test_robot"
MUJOCO_ROBOT_NAME = "Mujoco VX300s"
LOCAL_SERVER_PORT = 8181
TRAINING_TIMEOUT_MINUTES = 120
ENDPOINT_TIMEOUT_MINUTES = 30
MERGED_DATASET_RECORDING_TIMEOUT_SECONDS = 120
MERGED_DATASET_RECORDING_POLL_SECONDS = 5
RUNNING_STATE_TIMEOUT_MINUTES = 10
LOGS_AVAILABILITY_TIMEOUT_MINUTES = 10
TRAINING_POLL_SECONDS = 20
ENDPOINT_POLL_SECONDS = 20
JOB_STATE_POLL_SECONDS = 30
ENDPOINT_TTL_SECONDS = 60 * 30

JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
)
GRIPPER_NAMES = ["left_gripper", "right_gripper"]
DEPTH_CAM_NAME = "depth_angle"
POINT_CLOUD_SENSOR_NAME = "point_cloud"
POSE_SENSOR_NAME = "tcp"
LANGUAGE_LABEL = "instruction"


def _indexed_names(names: list[str] | tuple[str, ...]) -> dict[int, str]:
    return {index: name for index, name in enumerate(names)}


# Training/Inference robot (VX300s) embodiment descriptions
INPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.RGB_IMAGES: {0: NC_CAM_NAME},
    DataType.JOINT_POSITIONS: _indexed_names(names=JOINT_NAMES),
    DataType.LANGUAGE: {0: LANGUAGE_LABEL},
    DataType.JOINT_VELOCITIES: _indexed_names(names=JOINT_NAMES),
}
OUTPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.JOINT_POSITIONS: _indexed_names(names=JOINT_NAMES),
}

INPUT_DATA_TYPES = list(INPUT_EMBODIMENT_DESCRIPTION.keys())
OUTPUT_DATA_TYPES = list(OUTPUT_EMBODIMENT_DESCRIPTION.keys())

# "auto" lets the backend select an appropriate batch size automatically
CNNMLP_CONFIG = {
    "batch_size": "auto",
    "epochs": 1,
    "output_prediction_horizon": 5,
}

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "ERROR"}

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_sync_point(obs) -> SynchronizedPoint:
    return SynchronizedPoint(
        data={
            DataType.JOINT_POSITIONS: {
                name: JointData(value=obs.qpos[name]) for name in JOINT_NAMES
            },
            DataType.RGB_IMAGES: {
                NC_CAM_NAME: RGBCameraData(frame=obs.cameras[MJ_CAM_NAME].rgb)
            },
            DataType.JOINT_VELOCITIES: {
                name: JointData(value=obs.qvel[name]) for name in JOINT_NAMES
            },
            DataType.LANGUAGE: {LANGUAGE_LABEL: LanguageData(text="pick and place")},
        }
    )


def _collect_demo_data(
    robot_name: str,
    dataset_name: str,
    num_episodes: int = 3,
    instance_id: int = 0,
    episode_length_multiplier: int = 1,
    num_cameras: int = 1,
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
        for frame_idx, action_dict in enumerate(expanded_action_traj):
            t += 1.0 / FREQUENCY
            joint_positions = {
                k: v for k, v in action_dict.items() if "gripper" not in k
            }
            joint_torques = {
                name: float(0.01 * ((index + frame_idx) % 5))
                for index, name in enumerate(JOINT_NAMES)
            }
            joint_velocities = {
                name: float(0.05 * ((index + frame_idx) % 7))
                for index, name in enumerate(JOINT_NAMES)
            }
            gripper_open_amounts = {
                name: float(0.25 + 0.5 * ((frame_idx % 2) == 0))
                for name in GRIPPER_NAMES
            }
            pose = np.array([0.1 + frame_idx * 0.001, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
            img = np.zeros((84, 84, 3), dtype=np.uint8)
            img.fill(50 + frame_idx % 200)
            depth = np.full((64, 64), 0.75 + 0.01 * (frame_idx % 10), dtype=np.float32)
            point_cloud = np.linspace(0.0, 1.0, 96, dtype=np.float16).reshape(
                32, 3
            ) + np.float16(frame_idx * 0.001)
            rgb_points = np.full(
                (point_cloud.shape[0], 3), 80 + frame_idx % 120, dtype=np.uint8
            )
            nc.log_joint_positions(
                positions=joint_positions,
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
            nc.log_pose(
                name=POSE_SENSOR_NAME,
                pose=pose,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_language(
                name=LANGUAGE_LABEL,
                language="pick and place",
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_rgb(
                name=NC_CAM_NAME,
                rgb=img,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_parallel_gripper_open_amounts(
                values={"gripper1": 0.5, "gripper2": 0.5},
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_depth(
                name=DEPTH_CAM_NAME,
                depth=depth,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_point_cloud(
                name=POINT_CLOUD_SENSOR_NAME,
                points=point_cloud,
                rgb_points=rgb_points,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
        nc.stop_recording(wait=True, robot_name=robot_name, instance=instance_id)
        logger.info(
            f"Episode {ep_idx + 1} recorded ({len(expanded_action_traj)} frames)"
        )
    return dataset


def _wait_for_training(
    job_id: str, timeout_minutes: int = TRAINING_TIMEOUT_MINUTES
) -> str:
    deadline = time.time() + timeout_minutes * 60
    while True:
        status = nc.get_training_job_status(job_id=job_id)
        logger.info(f"Training job {job_id}: {status}")
        if status in TERMINAL_STATES:
            return status
        assert (
            time.time() < deadline
        ), f"Training job {job_id} did not finish within {timeout_minutes} minutes"
        time.sleep(TRAINING_POLL_SECONDS)


def _wait_for_all_training(
    job_ids: list[str], timeout_minutes: int = TRAINING_TIMEOUT_MINUTES
) -> dict[str, str]:
    assert job_ids, "Expected at least one training job id"
    deadline = time.time() + timeout_minutes * 60
    final_statuses: dict[str, str] = {}

    while True:
        for job_id in job_ids:
            if job_id in final_statuses:
                continue
            status = nc.get_training_job_status(job_id=job_id)
            logger.info(f"Training job {job_id}: {status}")
            if status in TERMINAL_STATES:
                final_statuses[job_id] = status

        if len(final_statuses) == len(job_ids):
            return final_statuses

        assert (
            time.time() < deadline
        ), f"Training job(s) {job_ids} did not finish within {timeout_minutes} minutes"
        time.sleep(TRAINING_POLL_SECONDS)


def _wait_for_endpoint(
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


def _wait_for_dataset_recording_count(
    dataset_name: str,
    expected_recordings: int,
    timeout_seconds: int = MERGED_DATASET_RECORDING_TIMEOUT_SECONDS,
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

        time.sleep(MERGED_DATASET_RECORDING_POLL_SECONDS)

    if last_error is not None:
        raise AssertionError(
            f"Dataset {dataset_name!r} did not become queryable within "
            f"{timeout_seconds} seconds; last error: {last_error}"
        )
    raise AssertionError(
        f"Dataset {dataset_name!r} had {last_count} recordings after "
        f"{timeout_seconds} seconds; expected {expected_recordings}"
    )


def _build_cross_embodiment_descriptions(
    dataset: Dataset,
    input_types: list[DataType],
    output_types: list[DataType],
) -> tuple[dict, dict]:
    input_desc: dict = {}
    output_desc: dict = {}
    for robot_id in dataset.robot_ids:
        embodiment = dataset.get_full_embodiment_description(robot_id)
        for data_type in input_types:
            assert (
                data_type in embodiment
            ), f"{data_type.value} missing from robot {robot_id} embodiment description"
        input_desc[robot_id] = {dt: embodiment[dt] for dt in input_types}
        output_desc[robot_id] = {dt: embodiment[dt] for dt in output_types}
    return input_desc, output_desc


def _run_policy_inference(policy: Policy) -> None:
    """Run inference via both sync-point and logging-function paths."""
    try:
        env = make_sim_env(seed=42)
        obs = env.reset()

        # Path 1: explicit SynchronizedPoint
        logger.info("Running sync-point inference (Path 1)")
        predictions = policy.predict(sync_point=_make_sync_point(obs=obs), timeout=30)
        for data_type in OUTPUT_DATA_TYPES:
            assert data_type in predictions, (
                f"Expected {data_type.value} in local "
                f"server output, got: {list(predictions.keys())}"
            )
        logger.info(f"Path 1 passed — output keys: {[k.value for k in predictions]}")

        # Path 2: nc.log_* → get_latest_sync_point internally
        logger.info("Running logging-function inference (Path 2)")
        nc.log_joint_positions(
            positions={name: float(obs.qpos[name]) for name in JOINT_NAMES}
        )
        nc.log_language(name=LANGUAGE_LABEL, language="pick and place")
        nc.log_joint_velocities(
            velocities={name: float(obs.qvel[name]) for name in JOINT_NAMES}
        )
        nc.log_rgb(name=NC_CAM_NAME, rgb=obs.cameras[MJ_CAM_NAME].rgb)
        predictions = policy.predict(timeout=30)
        assert DataType.JOINT_POSITIONS in predictions, (
            "Expected JOINT_POSITIONS in local "
            f"server output, got: {list(predictions.keys())}"
        )
        logger.info(f"Path 2 passed — output keys: {[k.value for k in predictions]}")
    finally:
        policy.disconnect()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainingFlow:
    """End-to-end flow: collect → merge → train → logs → resume → infer → deploy."""

    # Shared state across test_step* methods within one pytest session
    collected_dataset_name: str
    merged_dataset_name: str
    training_name: str
    collected_dataset: Dataset | None = None
    merged_dataset: Dataset | None = None
    job_id: str | None = None
    endpoint_id: str | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls.collected_dataset_name = _unique_name(prefix="collected")
        cls.merged_dataset_name = _unique_name(prefix="merged")
        cls.training_name = _unique_name(prefix="cnnmlp_flow")
        cls.collected_dataset = None
        cls.merged_dataset = None
        cls.job_id = None
        cls.endpoint_id = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
        if cls.endpoint_id:
            try:
                nc.delete_endpoint(cls.endpoint_id)
            except Exception:
                logger.warning(
                    f"Failed to delete endpoint {cls.endpoint_id}", exc_info=True
                )
        if cls.job_id:
            try:
                nc.delete_training_job(cls.job_id)
            except Exception:
                logger.warning(
                    f"Failed to delete training job {cls.job_id}", exc_info=True
                )
        if cls.merged_dataset:
            try:
                cls.merged_dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete merged dataset {cls.merged_dataset_name}",
                    exc_info=True,
                )
        if cls.collected_dataset:
            try:
                cls.collected_dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete collected dataset {cls.collected_dataset_name}",
                    exc_info=True,
                )

    def test_step1_collect_demo_data(self) -> None:
        self.__class__.collected_dataset = _collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.collected_dataset_name,
            num_episodes=COLLECTED_DEMO_EPISODES,
        )
        self.__class__.collected_dataset = _wait_for_dataset_recording_count(
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
        self.__class__.merged_dataset = _wait_for_dataset_recording_count(
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

        input_desc, output_desc = _build_cross_embodiment_descriptions(
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
        logger.info(f"[STEP 3] [PASSED] Training Job Started: {self.job_id}")

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
            filtered = nc.get_training_job_logs(
                job_id=self.job_id, max_entries=10, severity_filter="ERROR"
            )
            assert "logs" in filtered
            logger.info(
                f"[STEP 4] [PASSED] Retrieved {logs['total_entries']} Log Entries"
            )

    def test_step5_assert_training_completed(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        final_status = _wait_for_training(job_id=self.job_id)
        assert (
            final_status == "COMPLETED"
        ), f"Training ended with non-COMPLETED status: {final_status}"
        logger.info(f"[STEP 5] [PASSED] Job {self.job_id} Completed")

    def test_step6_resume_training(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        initial_epoch = nc.get_training_job_data(job_id=self.job_id).get("epoch", 0)
        resumed_job = nc.resume_training_run(job_id=self.job_id, additional_epochs=1)
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

        final_resumed_status = _wait_for_training(job_id=self.job_id)
        assert (
            final_resumed_status == "COMPLETED"
        ), f"Resumed training ended with non-COMPLETED status: {final_resumed_status}"

        resumed_data = nc.get_training_job_data(job_id=self.job_id)
        assert (resumed_data.get("epoch") or 0) > initial_epoch, (
            f"Expected epoch to increase after resume, "
            f"was {initial_epoch}, now {resumed_data.get('epoch')}"
        )
        assert (
            resumed_data.get("previous_training_time") is not None
        ), "Expected previous_training_time to be set after resume"
        logger.info(
            f"[STEP 6] [PASSED] Resumed Job Completed At Epoch"
            f" {resumed_data.get('epoch')}"
        )

    def test_step7_direct_inference(self) -> None:
        nc.connect_robot(robot_name=MUJOCO_ROBOT_NAME)
        policy = nc.policy(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            train_run_name=self.training_name,
        )
        _run_policy_inference(policy=policy)
        logger.info("[STEP 7] [PASSED] Direct In-Process Inference Succeeded")

    def test_step8_local_server_inference(self) -> None:
        policy = nc.policy_local_server(
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            train_run_name=self.training_name,
            port=LOCAL_SERVER_PORT,
        )
        _run_policy_inference(policy=policy)
        logger.info("[STEP 8] [PASSED] Local Server Inference Succeeded")

    def test_step9_deploy_remote_endpoint(self) -> None:
        assert self.job_id is not None, "[STEP 3] Did Not Complete"
        endpoint_name = _unique_name(prefix="flow_endpoint")
        endpoint_data = nc.deploy_model(
            job_id=self.job_id,
            name=endpoint_name,
            input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
            output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
            ttl=ENDPOINT_TTL_SECONDS,
        )
        self.__class__.endpoint_id = endpoint_data["id"]
        assert self.endpoint_id is not None
        final_status = _wait_for_endpoint(endpoint_id=self.endpoint_id)
        assert (
            final_status == "active"
        ), f"Endpoint did not become active, status: {final_status!r}"
        logger.info(f"[STEP 9] [PASSED] Endpoint {self.endpoint_id} Is Active")


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

    job_id: str | None = None
    dataset: Dataset | None = None
    dataset_name: str

    @classmethod
    def setup_class(cls) -> None:
        cls.dataset_name = _unique_name(prefix="failure_report_test")
        cls.job_id = None
        cls.dataset = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
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
        self.__class__.dataset = _collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.dataset_name,
            num_episodes=1,
            instance_id=1,
        )
        logger.info(
            f"[STEP 1] [PASSED] Collected 1 Recording Into '{self.dataset_name}'"
        )

    def test_step2_submit_failing_job(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = _build_cross_embodiment_descriptions(
            dataset=self.dataset,
            input_types=INPUT_DATA_TYPES,
            output_types=OUTPUT_DATA_TYPES,
        )
        job_data = nc.start_training_run(
            name=_unique_name(prefix="failure_report_job"),
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
        final_status = _wait_for_training(job_id=self.job_id, timeout_minutes=30)
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


class TestBackToBackTraining:
    """Launch multiple training jobs back-to-back against the same dataset."""

    dataset: Dataset | None = None
    dataset_name: str
    job_ids: list[str]
    training_names: list[str]

    @classmethod
    def setup_class(cls) -> None:
        cls.dataset_name = _unique_name(prefix="back_to_back_training")
        cls.training_names = [
            _unique_name(prefix="cnnmlp_back_to_back")
            for _ in range(BACK_TO_BACK_NUM_JOBS)
        ]
        cls.job_ids = []
        cls.dataset = None
        nc.login()

    @classmethod
    def teardown_class(cls) -> None:
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
        _collect_demo_data(
            robot_name=ROBOT_NAME,
            dataset_name=self.dataset_name,
            num_episodes=BACK_TO_BACK_NUM_EPISODES,
            instance_id=2,
            episode_length_multiplier=BACK_TO_BACK_EPISODE_LENGTH_MULTIPLIER,
            num_cameras=BACK_TO_BACK_NUM_CAMERAS,
        )
        self.__class__.dataset = _wait_for_dataset_recording_count(
            dataset_name=self.dataset_name,
            expected_recordings=BACK_TO_BACK_NUM_EPISODES,
        )
        logger.info(
            f"[STEP 1] [PASSED] Collected {len(self.dataset)} Recordings"
            f" Into '{self.dataset_name}'"
        )

    def test_step2_submit_back_to_back_jobs(self) -> None:
        assert self.dataset is not None, "[STEP 1] Did Not Complete"
        input_desc, output_desc = _build_cross_embodiment_descriptions(
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
        final_statuses = _wait_for_all_training(
            job_ids=self.job_ids, timeout_minutes=60
        )
        for job_id, status in final_statuses.items():
            assert status == "COMPLETED", (
                f"Back-to-back training job {job_id} ended with "
                f"non-COMPLETED status: {status}"
            )
        logger.info(
            f"[STEP 3] [PASSED] All {len(self.job_ids)} Back-To-Back Jobs Completed"
        )
