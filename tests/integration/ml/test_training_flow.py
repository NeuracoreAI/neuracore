"""Integration tests for training flows on the Neuracore platform.

A single end-to-end test that covers: dataset collection, merging, training
(with auto batch sizing), log retrieval, direct inference, local server
inference, and remote endpoint deployment — all sharing one training job.
"""

import logging
import os
import pprint
import sys
import time
import uuid

import numpy as np
import pytest
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
TRAINING_TIMEOUT_MINUTES = 180
ENDPOINT_TIMEOUT_MINUTES = 30
MERGED_DATASET_RECORDING_TIMEOUT_SECONDS = 120
MERGED_DATASET_RECORDING_POLL_SECONDS = 5

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
    DataType.JOINT_POSITIONS: _indexed_names(JOINT_NAMES),
    DataType.LANGUAGE: {0: LANGUAGE_LABEL},
    DataType.JOINT_VELOCITIES: _indexed_names(JOINT_NAMES),
}
OUTPUT_EMBODIMENT_DESCRIPTION: EmbodimentDescription = {
    DataType.JOINT_POSITIONS: _indexed_names(JOINT_NAMES),
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
    assert episode_length_multiplier >= 1, (
        "episode_length_multiplier must be >= 1, " f"got {episode_length_multiplier}"
    )
    assert num_cameras >= 1, f"num_cameras must be >= 1, got {num_cameras}"

    nc.connect_robot(
        robot_name,
        instance=instance_id,
        urdf_path=str(BIMANUAL_VIPERX_URDF_PATH),
        overwrite=False,
    )
    dataset = nc.create_dataset(dataset_name)

    for ep_idx in range(num_episodes):
        logger.info(f"Collecting episode {ep_idx + 1}/{num_episodes}")
        action_traj = rollout_policy()
        expanded_action_traj = [
            action_dict
            for action_dict in action_traj
            for _ in range(episode_length_multiplier)
        ]
        [f"{NC_CAM_NAME}_{camera_index}" for camera_index in range(1, num_cameras + 1)]
        nc.start_recording(robot_name=robot_name, instance=instance_id)
        t = time.time()
        for frame_idx, action_dict in enumerate(expanded_action_traj):
            t += 0.02
            joint_positions = {
                k: v for k, v in action_dict.items() if "gripper" not in k
            }
            joint_torques = {
                name: float(0.01 * ((index + frame_idx) % 5))
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
                joint_positions,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_joint_velocities(
                joint_positions,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_joint_torques(
                joint_torques,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_parallel_gripper_open_amounts(
                gripper_open_amounts,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_pose(
                POSE_SENSOR_NAME,
                pose,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_language(
                LANGUAGE_LABEL,
                "pick and place",
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_rgb(
                NC_CAM_NAME,
                img,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_parallel_gripper_open_amounts(
                {"gripper1": 0.5, "gripper2": 0.5},
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_depth(
                DEPTH_CAM_NAME,
                depth,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
            nc.log_point_cloud(
                POINT_CLOUD_SENSOR_NAME,
                point_cloud,
                rgb_points=rgb_points,
                timestamp=t,
                robot_name=robot_name,
                instance=instance_id,
            )
        nc.stop_recording(wait=True, robot_name=robot_name, instance=instance_id)
        logger.info(
            "Episode %s recorded (%s frames)",
            ep_idx + 1,
            len(expanded_action_traj),
            episode_length_multiplier,
            num_cameras,
        )
    return dataset


def _wait_for_training(
    job_id: str | list[str], timeout_minutes: int = TRAINING_TIMEOUT_MINUTES
) -> str | dict[str, str]:
    job_ids = [job_id] if isinstance(job_id, str) else list(job_id)
    assert job_ids, "Expected at least one training job id"

    deadline = time.time() + timeout_minutes * 60
    final_statuses: dict[str, str] = {}

    while True:
        for current_job_id in job_ids:
            if current_job_id in final_statuses:
                continue
            status = nc.get_training_job_status(current_job_id)
            logger.info(f"Training job {current_job_id}: {status}")
            if status in TERMINAL_STATES:
                final_statuses[current_job_id] = status

        if len(final_statuses) == len(job_ids):
            if isinstance(job_id, str):
                return final_statuses[job_id]
            return final_statuses

        assert (
            time.time() < deadline
        ), f"Training job(s) {job_ids} did not finish within {timeout_minutes} minutes"
        time.sleep(20)


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
        time.sleep(20)


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
            dataset = nc.get_dataset(dataset_name)
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


def _run_policy_inference(policy: Policy) -> None:
    """Run inference via both sync-point and logging-function paths."""
    try:
        env = make_sim_env(seed=42)
        obs = env.reset()

        # Path 1: explicit SynchronizedPoint
        logger.info("Running sync-point inference (Path 1)")
        predictions = policy.predict(sync_point=_make_sync_point(obs), timeout=30)
        for data_type in OUTPUT_DATA_TYPES:
            assert data_type in predictions, (
                f"Expected {data_type.value} in local "
                f"server output, got: {list(predictions.keys())}"
            )
        logger.info(f"Path 1 passed — output keys: {[k.value for k in predictions]}")

        # Path 2: nc.log_* → get_latest_sync_point internally
        logger.info("Running logging-function inference (Path 2)")
        nc.log_joint_positions({name: float(obs.qpos[name]) for name in JOINT_NAMES})
        nc.log_language(name=LANGUAGE_LABEL, language="pick and place")
        nc.log_joint_velocities({name: float(obs.qvel[name]) for name in JOINT_NAMES})
        nc.log_rgb(NC_CAM_NAME, obs.cameras[MJ_CAM_NAME].rgb)
        predictions = policy.predict(timeout=30)
        assert DataType.JOINT_POSITIONS in predictions, (
            "Expected JOINT_POSITIONS in local "
            f"server output, got: {list(predictions.keys())}"
        )
        logger.info(f"Path 2 passed — output keys: {[k.value for k in predictions]}")
    finally:
        policy.disconnect()


def test_training_flow():
    """End-to-end flow: collect → merge → train → logs → resume → infer → deploy.

    Steps (each reports a clear failure message):
      1. Collect demonstration data
      2. Merge collected dataset with shared dataset
      3. Train CNNMLP with auto batch sizing
      4. Retrieve and validate training logs while RUNNING
      5. Assert training COMPLETED
      6. Resume training with additional epochs and verify completion
      7. Direct in-process policy inference
      8. Local server policy inference
      9. Deploy remote endpoint and verify active
    """
    nc.login()

    collected_dataset_name = _unique_name("collected")
    merged_dataset_name = _unique_name("merged")
    training_name = _unique_name("cnnmlp_flow")
    collected_dataset = None
    merged_dataset = None
    job_id = None
    endpoint_id = None

    try:
        # ------------------------------------------------------------------
        # Step 1: Collect demo data
        # ------------------------------------------------------------------
        logger.info(f"Collecting {COLLECTED_DEMO_EPISODES} demo episodes")
        try:
            collected_dataset = _collect_demo_data(
                ROBOT_NAME,
                collected_dataset_name,
                num_episodes=COLLECTED_DEMO_EPISODES,
            )
            # stop_recording(wait=True) ensures upload completes, but the
            # backend indexes recordings asynchronously so len(dataset) may
            # lag. Poll until the count matches before proceeding.
            collected_dataset = _wait_for_dataset_recording_count(
                collected_dataset_name,
                expected_recordings=COLLECTED_DEMO_EPISODES,
            )
            collected_recordings = len(collected_dataset)
            logger.info(
                f"Step 1 passed — {collected_recordings} recordings"
                f" in '{collected_dataset_name}'"
            )
        except Exception as e:
            pytest.fail(f"Step 1 (collect demo data) failed: {e}")

        # ------------------------------------------------------------------
        # Step 2: Merge collected dataset with shared dataset
        # ------------------------------------------------------------------
        logger.info(f"Merging '{collected_dataset_name}' with '{SHARED_DATASET_NAME}'")
        try:
            shared_dataset = nc.get_dataset(SHARED_DATASET_NAME)
            shared_recordings = len(shared_dataset)
            expected_merged_recordings = collected_recordings + shared_recordings
            logger.info(
                f"Shared dataset '{SHARED_DATASET_NAME}' has"
                f" {shared_recordings} recordings"
                f" — expecting {expected_merged_recordings} merged"
            )

            merged_dataset = nc.merge_datasets(
                merged_dataset_name,
                [collected_dataset_name, SHARED_DATASET_NAME],
            )
            assert (
                merged_dataset.name == merged_dataset_name
            ), f"Merged dataset name mismatch: {merged_dataset.name!r}"
            merged_dataset = _wait_for_dataset_recording_count(
                merged_dataset_name,
                expected_recordings=expected_merged_recordings,
            )
            logger.info(f"Merged dataset id={merged_dataset.id}")
        except Exception as e:
            pytest.fail(f"Step 2 (merge datasets) failed: {e}")

        # ------------------------------------------------------------------
        # Step 3: Train CNNMLP with auto batch sizing
        # ------------------------------------------------------------------
        logger.info(f"Preparing training inputs for '{training_name}'")
        try:
            dataset = nc.get_dataset(merged_dataset_name)
            robot_ids = dataset.robot_ids

            assert (
                len(robot_ids) == 2
            ), f"Expected 2 robots in merged dataset, got {robot_ids}"
            logger.info(f"Found {len(robot_ids)} robots: {robot_ids}")
            input_cross_embodiment_description = {}
            output_cross_embodiment_description = {}
            for robot_id in robot_ids:
                embodiment_description = merged_dataset.get_full_embodiment_description(
                    robot_id
                )
                for data_type in INPUT_DATA_TYPES:
                    assert data_type in embodiment_description, (
                        f"{data_type.value} missing from robot {robot_id} "
                        "embodiment description"
                    )
                input_cross_embodiment_description[robot_id] = {
                    data_type: embodiment_description[data_type]
                    for data_type in INPUT_DATA_TYPES
                }
                logger.info(
                    f"Input embodiment for robot {robot_id}:\n"
                    f"{pprint.pformat(input_cross_embodiment_description[robot_id])}"
                )
                output_cross_embodiment_description[robot_id] = {
                    data_type: embodiment_description[data_type]
                    for data_type in OUTPUT_DATA_TYPES
                }
                logger.info(
                    f"Output embodiment for robot {robot_id}:\n"
                    f"{pprint.pformat(output_cross_embodiment_description[robot_id])}"
                )

            job_data = nc.start_training_run(
                name=training_name,
                dataset_name=merged_dataset_name,
                algorithm_name="CNNMLP",
                algorithm_config=CNNMLP_CONFIG,
                gpu_type=GPU_TYPE,
                num_gpus=NUM_GPUS,
                frequency=FREQUENCY,
                input_cross_embodiment_description=input_cross_embodiment_description,
                output_cross_embodiment_description=output_cross_embodiment_description,
            )
            job_id = job_data["id"]
            logger.info(f"Training job started: {job_id}")
        except Exception as e:
            pytest.fail(f"Step 3 (start training) failed: {e}")

        # ------------------------------------------------------------------
        # Step 4: Retrieve training logs while RUNNING
        # The backend raises 404 (CloudComputeIDNotFoundError) until the GCP
        # VM is registered, so we poll until logs are available.
        # ------------------------------------------------------------------
        logger.info("Waiting for RUNNING state then retrieving logs")
        try:
            # Wait for RUNNING state
            running_deadline = time.time() + 10 * 60
            while True:
                job_status = nc.get_training_job_status(job_id)
                logger.info(f"Job {job_id} status: {job_status} (waiting for RUNNING)")
                if job_status == "RUNNING":
                    break
                assert (
                    job_status not in TERMINAL_STATES
                ), f"Job reached {job_status} before entering RUNNING state"
                assert (
                    time.time() < running_deadline
                ), "Job did not reach RUNNING state within 10 minutes"
                time.sleep(30)

            # Poll for logs — 404 is expected until the compute instance registers
            logs_deadline = time.time() + 10 * 60
            logs = None
            while logs is None:
                job_status = nc.get_training_job_status(job_id)
                try:
                    logs = nc.get_training_job_logs(job_id, max_entries=50)
                except ValueError:
                    # 404 — compute instance not registered yet
                    if job_status in TERMINAL_STATES or time.time() > logs_deadline:
                        logger.warning(
                            "Logs unavailable before job completed; "
                            "Skipping log assertions"
                        )
                        break
                    time.sleep(30)

            if logs is not None:
                for field in ("job_id", "logs", "total_entries", "retrieved_at"):
                    assert (
                        field in logs
                    ), f"Missing '{field}' in CloudComputeLogs response"
                assert isinstance(logs["logs"], list)
                assert isinstance(logs["total_entries"], int)
                for entry in logs["logs"]:
                    assert "message" in entry, f"Log entry missing 'message': {entry}"

                filtered = nc.get_training_job_logs(
                    job_id, max_entries=10, severity_filter="ERROR"
                )
                assert "logs" in filtered
                logger.info(f"Retrieved {logs['total_entries']} log entries")
        except Exception as e:
            pytest.fail(f"Step 4 (training logs) failed: {e}")

        # ------------------------------------------------------------------
        # Step 5: Wait for training to complete
        # ------------------------------------------------------------------
        logger.info(f"Waiting for training job {job_id} to complete")
        try:
            final_status = _wait_for_training(job_id)
            assert (
                final_status == "COMPLETED"
            ), f"Training ended with non-COMPLETED status: {final_status}"
            logger.info(f"Step 5 passed — job {job_id} COMPLETED")
        except Exception as e:
            pytest.fail(f"Step 5 (training completion) failed: {e}")

        # ------------------------------------------------------------------
        # Step 6: Resume training with additional epochs
        # ------------------------------------------------------------------
        logger.info(f"Resuming training job {job_id} for 1 more epoch")
        try:
            initial_epoch = nc.get_training_job_data(job_id).get("epoch", 0)
            resumed_job = nc.resume_training_run(job_id, additional_epochs=1)
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

            final_resumed_status = _wait_for_training(job_id)
            assert final_resumed_status == "COMPLETED", (
                "Resumed training ended with non-COMPLETED status: "
                f"{final_resumed_status}"
            )
            resumed_data = nc.get_training_job_data(job_id)
            assert (resumed_data.get("epoch") or 0) > initial_epoch, (
                f"Expected epoch to increase after resume, "
                f"was {initial_epoch}, now {resumed_data.get('epoch')}"
            )
            assert (
                resumed_data.get("previous_training_time") is not None
            ), "Expected previous_training_time to be set after resume"
            logger.info(
                "Step 6 passed — resumed job completed at epoch"
                f" {resumed_data.get('epoch')}"
            )
        except Exception as e:
            pytest.fail(f"Step 6 (resume training) failed: {e}")

        # ------------------------------------------------------------------
        # Step 7: Direct in-process policy inference
        # ------------------------------------------------------------------
        logger.info("Direct in-process policy inference")
        try:
            nc.connect_robot(MUJOCO_ROBOT_NAME)
            policy = nc.policy(
                input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
                output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
                train_run_name=training_name,
            )
            _run_policy_inference(policy)
            logger.info("Step 7 passed — direct in-process inference succeeded")
        except Exception as e:
            pytest.fail(f"Step 7 (direct policy inference) failed: {e}")

        # ------------------------------------------------------------------
        # Step 8: Local server policy inference
        # ------------------------------------------------------------------
        logger.info("Local server policy inference on port 8181")
        try:
            policy = nc.policy_local_server(
                input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
                output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
                train_run_name=training_name,
                port=8181,
            )
            _run_policy_inference(policy)
            logger.info("Step 8 passed — local server inference succeeded")
        except Exception as e:
            pytest.fail(f"Step 8 (local server inference) failed: {e}")

        # ------------------------------------------------------------------
        # Step 9: Deploy remote endpoint and verify active
        # ------------------------------------------------------------------
        logger.info("Deploying remote endpoint")
        try:
            endpoint_name = _unique_name("flow_endpoint")
            endpoint_data = nc.deploy_model(
                job_id=job_id,
                name=endpoint_name,
                input_embodiment_description=INPUT_EMBODIMENT_DESCRIPTION,
                output_embodiment_description=OUTPUT_EMBODIMENT_DESCRIPTION,
                ttl=60 * 30,
            )
            endpoint_id = endpoint_data["id"]
            final_endpoint_status = _wait_for_endpoint(endpoint_id)
            assert (
                final_endpoint_status == "active"
            ), f"Endpoint did not become active, status: {final_endpoint_status!r}"
            logger.info(f"Endpoint {endpoint_id} is active")
        except Exception as e:
            pytest.fail(f"Step 9 (remote endpoint deployment) failed: {e}")

    finally:
        if endpoint_id:
            try:
                nc.delete_endpoint(endpoint_id)
            except Exception:
                logger.warning(f"Failed to delete endpoint {endpoint_id}")
        if job_id:
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete training job {job_id}")
        if merged_dataset:
            try:
                merged_dataset.delete()
            except Exception:
                logger.warning(f"Failed to delete merged dataset {merged_dataset_name}")
        if collected_dataset:
            try:
                collected_dataset.delete()
            except Exception:
                logger.warning(
                    f"Failed to delete collected dataset {collected_dataset_name}"
                )


def test_training_failure_error_reporting():
    """Verify that training script failures are correctly reported to the cloud.

    Forces a deliberate runtime failure by submitting a training job whose
    batch_size cannot be parsed as an integer.  The error occurs inside
    train.py *after* nc.login() — so the new top-level error handler in
    main() is responsible for catching it and calling
    _try_report_error_to_cloud().

    Assertions:
    1. The job reaches FAILED status (not stuck in RUNNING or PENDING).
    2. The job data returned by the API contains a non-empty 'error' field,
       confirming that the error was propagated back to the server.
    """
    nc.login()

    job_id = None
    dataset = None
    dataset_name = _unique_name("failure_report_test")

    logger.info(f"Collecting 1 demo episode into '{dataset_name}'")
    try:
        dataset = _collect_demo_data(
            ROBOT_NAME, dataset_name, num_episodes=1, instance_id=1
        )
        logger.info(f"Collected 1 recording into '{dataset_name}'")
    except Exception as e:
        pytest.fail(f"Data collection failed: {e}")

    # ------------------------------------------------------------------
    # Build a per-robot embodiment description from the collected dataset
    # ------------------------------------------------------------------
    logger.info("Building embodiment descriptions from collected dataset")
    try:
        robot_ids = dataset.robot_ids
        logger.info(f"Found {len(robot_ids)} robots: {robot_ids}")
        input_cross_embodiment_description: dict = {}
        output_cross_embodiment_description: dict = {}
        for robot_id in robot_ids:
            embodiment_description = dataset.get_full_embodiment_description(robot_id)
            for data_type in INPUT_DATA_TYPES:
                assert data_type in embodiment_description, (
                    f"{data_type.value} missing from robot {robot_id} "
                    "embodiment description"
                )
            input_cross_embodiment_description[robot_id] = {
                data_type: embodiment_description[data_type]
                for data_type in INPUT_DATA_TYPES
            }
            output_cross_embodiment_description[robot_id] = {
                data_type: embodiment_description[data_type]
                for data_type in OUTPUT_DATA_TYPES
            }
    except Exception as e:
        pytest.fail(f"Building embodiment description failed: {e}")

    # ------------------------------------------------------------------
    # Submit a training job that will fail at runtime
    # ------------------------------------------------------------------
    try:
        job_data = nc.start_training_run(
            name=_unique_name("failure_report_job"),
            dataset_name=dataset_name,
            algorithm_name="CNNMLP",
            algorithm_config=FAILURE_CNNMLP_CONFIG,
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            input_cross_embodiment_description=(input_cross_embodiment_description),
            output_cross_embodiment_description=(output_cross_embodiment_description),
        )
        job_id = job_data["id"]
        logger.info(f"Failure-reporting test job started: {job_id}")
    except Exception as e:
        pytest.fail(f"Failed to submit training job: {e}")

    # ------------------------------------------------------------------
    # Wait for the job to reach a terminal state (expect FAILED)
    # ------------------------------------------------------------------
    try:
        final_status = _wait_for_training(job_id, timeout_minutes=30)
        assert final_status == "FAILED", (
            f"Expected FAILED status, got: {final_status!r}.  "
            "The deliberate bad batch_size should have caused a ValueError "
            "in train.py that maps to a FAILED job."
        )
        logger.info(f"Job {job_id} correctly reached FAILED status")
    except Exception as e:
        pytest.fail(f"Unexpected error waiting for job failure: {e}")

    # ------------------------------------------------------------------
    # Verify error info is surfaced in the job data
    # ------------------------------------------------------------------
    try:
        job_detail = nc.get_training_job_data(job_id)
        assert "error" in job_detail, (
            "Job data is missing 'error' field — the server may not have "
            "received the error report from the training script."
        )
        assert job_detail["error"], (
            "The 'error' field in job data is empty — "
            "_try_report_error_to_cloud may not have been called."
        )
        logger.info(
            f"Error field present in job data: {str(job_detail['error'])[:200]}",
        )
    except Exception as e:
        pytest.fail(f"Failed to verify error info in job data: {e}")

    finally:
        if job_id:
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete job {job_id}")
        if dataset:
            try:
                dataset.delete()
            except Exception:
                logger.warning(f"Failed to delete dataset {dataset_name}")


def test_back_to_back_training_jobs_same_dataset():
    """Launch multiple training jobs back-to-back against the same dataset.

    Steps:
      1. Collect demonstration data for a single dataset
      2. Build cross-embodiment data specs
      3. Submit multiple training jobs back-to-back
      4. Wait for both jobs to complete successfully
    """
    nc.login()

    dataset = None
    dataset_name = _unique_name("back_to_back_training")
    training_names = [
        _unique_name("cnnmlp_back_to_back") for _ in range(BACK_TO_BACK_NUM_JOBS)
    ]
    job_ids: list[str] = []

    input_cross_embodiment_description = {}
    output_cross_embodiment_description = {}

    try:
        # ------------------------------------------------------------------
        # Step 1: Collect demo data (requires measurable synchronization time)
        # ------------------------------------------------------------------
        try:
            dataset = _collect_demo_data(
                ROBOT_NAME,
                dataset_name,
                num_episodes=BACK_TO_BACK_NUM_EPISODES,
                instance_id=2,
                episode_length_multiplier=BACK_TO_BACK_EPISODE_LENGTH_MULTIPLIER,
                num_cameras=BACK_TO_BACK_NUM_CAMERAS,
            )
            assert len(dataset) == BACK_TO_BACK_NUM_EPISODES, (
                f"Expected {BACK_TO_BACK_NUM_EPISODES} recordings in dataset "
                f"{dataset_name!r}, got {len(dataset)}"
            )
        except Exception as e:
            pytest.fail(f"Step 1 (collect demo data) failed: {e}")

        # ------------------------------------------------------------------
        # Step 2: Build cross-embodiment data specs
        # ------------------------------------------------------------------
        try:
            robot_ids = dataset.robot_ids
            assert robot_ids, "Expected at least one robot in collected dataset"

            for robot_id in robot_ids:
                data_spec = dataset.get_full_embodiment_description(robot_id)
                input_cross_embodiment_description[robot_id] = {
                    DataType.JOINT_POSITIONS: data_spec[DataType.JOINT_POSITIONS],
                    DataType.RGB_IMAGES: data_spec[DataType.RGB_IMAGES],
                    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: data_spec[
                        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS
                    ],
                }
                output_cross_embodiment_description[robot_id] = {
                    DataType.JOINT_TARGET_POSITIONS: data_spec[
                        DataType.JOINT_TARGET_POSITIONS
                    ],
                    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: data_spec[
                        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS
                    ],
                }
        except Exception as e:
            pytest.fail(f"Step 2 (build data specs) failed: {e}")

        # ------------------------------------------------------------------
        # Step 3: Submit multiple training jobs back-to-back
        # ------------------------------------------------------------------
        try:
            for train_run_name in training_names:
                job_data = nc.start_training_run(
                    name=train_run_name,
                    dataset_name=dataset_name,
                    algorithm_name="CNNMLP",
                    algorithm_config=BACK_TO_BACK_CNNMLP_CONFIG,
                    gpu_type=GPU_TYPE,
                    num_gpus=NUM_GPUS,
                    frequency=BACK_TO_BACK_FREQUENCY,
                    input_cross_embodiment_description=(
                        input_cross_embodiment_description
                    ),
                    output_cross_embodiment_description=(
                        output_cross_embodiment_description
                    ),
                )
                job_ids.append(job_data["id"])
            assert len(set(job_ids)) == BACK_TO_BACK_NUM_JOBS, (
                "Expected distinct training jobs, got duplicate job ids: " f"{job_ids}"
            )
            assert (
                len(job_ids) == BACK_TO_BACK_NUM_JOBS
            ), f"Expected {BACK_TO_BACK_NUM_JOBS} submitted jobs, got {len(job_ids)}"
        except Exception as e:
            pytest.fail(f"Step 3 (submit back-to-back trainings) failed: {e}")

        # ------------------------------------------------------------------
        # Step 4: Wait for both jobs to complete
        # ------------------------------------------------------------------
        try:
            final_statuses = _wait_for_training(job_ids, timeout_minutes=60)
            for job_id in job_ids:
                final_status = final_statuses[job_id]
                assert final_status == "COMPLETED", (
                    f"Back-to-back training job {job_id} ended with "
                    f"non-COMPLETED status: {final_status}"
                )
        except Exception as e:
            pytest.fail(f"Step 4 (wait for back-to-back trainings) failed: {e}")
    finally:
        for job_id in job_ids:
            try:
                nc.delete_training_job(job_id)
            except Exception:
                logger.warning(f"Failed to delete training job {job_id}")
        if dataset:
            try:
                dataset.delete()
            except Exception:
                logger.warning(f"Failed to delete dataset {dataset_name}")
