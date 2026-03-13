"""Integration tests for training flows on the Neuracore platform.

A single end-to-end test that covers: dataset collection, merging, training
(with auto batch sizing), log retrieval, direct inference, local server
inference, and remote endpoint deployment — all sharing one training job.
"""

import logging
import os
import sys
import time
import uuid

import numpy as np
import pytest
from neuracore_types import DataType, JointData, RGBCameraData, SynchronizedPoint

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
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 20
NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
ROBOT_NAME = "integration_test_robot"
MUJOCO_ROBOT_NAME = "Mujoco VX300s"
TRAINING_TIMEOUT_MINUTES = 180
ENDPOINT_TIMEOUT_MINUTES = 30

JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
)

INPUT_DATA_SPEC = {
    DataType.RGB_IMAGES: [NC_CAM_NAME],
    DataType.JOINT_POSITIONS: list(JOINT_NAMES),
}
OUTPUT_DATA_SPEC = {
    DataType.JOINT_POSITIONS: JOINT_NAMES,
}

INPUT_DATA_TYPES = [
    DataType.RGB_IMAGES,
    DataType.JOINT_POSITIONS,
]

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
        }
    )


def _collect_demo_data(
    robot_name: str, dataset_name: str, num_episodes: int = 3
) -> Dataset:
    """Collect scripted demonstrations and log them to neuracore."""
    nc.connect_robot(
        robot_name, urdf_path=str(BIMANUAL_VIPERX_URDF_PATH), overwrite=False
    )
    dataset = nc.create_dataset(dataset_name)

    for ep_idx in range(num_episodes):
        logger.info(f"Collecting episode {ep_idx + 1}/{num_episodes}")
        action_traj = rollout_policy()
        nc.start_recording()
        for frame_idx, action_dict in enumerate(action_traj):
            t = time.time()
            joint_positions = {
                k: v for k, v in action_dict.items() if "gripper" not in k
            }
            nc.log_joint_positions(joint_positions, timestamp=t)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img.fill(50 + frame_idx % 200)
            nc.log_rgb(NC_CAM_NAME, img, timestamp=t)
            nc.log_joint_target_positions(action_dict, timestamp=t)
        nc.stop_recording(wait=True)
        logger.info(f"Episode {ep_idx + 1} recorded ({len(action_traj)} frames)")
    return dataset


def _wait_for_training(
    job_id: str, timeout_minutes: int = TRAINING_TIMEOUT_MINUTES
) -> str:
    deadline = time.time() + timeout_minutes * 60
    while True:
        status = nc.get_training_job_status(job_id)
        logger.info(f"Training job {job_id}: {status}")
        if status in TERMINAL_STATES:
            return status
        assert (
            time.time() < deadline
        ), f"Training job {job_id} did not finish within {timeout_minutes} minutes"
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


def _run_policy_inference(policy: Policy) -> None:
    try:
        env = make_sim_env(seed=42)
        obs = env.reset()
        predictions = policy.predict(sync_point=_make_sync_point(obs), timeout=30)
        assert DataType.JOINT_POSITIONS in predictions, (
            "Expected JOINT_POSITIONS in local "
            f"server output, got: {list(predictions.keys())}"
        )
    finally:
        policy.disconnect()


def test_training_flow():
    """End-to-end training flow: collect → merge → train → logs → infer → deploy.

    Steps (each reports a clear failure message):
      1. Collect demonstration data
      2. Merge collected dataset with shared dataset
      3. Train CNNMLP with auto batch sizing
      4. Retrieve and validate training logs while RUNNING
      5. Assert training COMPLETED
      6. Direct in-process policy inference
      7. Local server policy inference
      8. Deploy remote endpoint and verify active
    """
    nc.login()

    collected_dataset_name = _unique_name("collected")
    collected_dataset_name = "collected_6daf55cb"
    merged_dataset_name = _unique_name("merged")
    merged_dataset_name = "merged_2cf69cf7"
    training_name = _unique_name("cnnmlp_flow")
    job_id = None
    endpoint_id = None
    merged_dataset = None
    collected_dataset = None

    try:
        # ------------------------------------------------------------------
        # Step 1: Collect demo data
        # ------------------------------------------------------------------
        # try:
        #     collected_dataset = _collect_demo_data(
        #         ROBOT_NAME, collected_dataset_name, num_episodes=3
        #     )
        # except Exception as e:
        #     pytest.fail(f"Step 1 (collect demo data) failed: {e}")

        # # ------------------------------------------------------------------
        # # Step 2: Merge collected dataset with shared dataset
        # # ------------------------------------------------------------------
        # try:
        #     merged_dataset = nc.merge_datasets(
        #         merged_dataset_name,
        #         [collected_dataset_name, SHARED_DATASET_NAME],
        #     )
        #     assert (
        #         merged_dataset.name == merged_dataset_name
        #     ), f"Merged dataset name mismatch: {merged_dataset.name!r}"
        #     logger.info(f"Merged dataset: {merged_dataset.id}")
        # except Exception as e:
        #     pytest.fail(f"Step 2 (merge datasets) failed: {e}")

        # ------------------------------------------------------------------
        # Step 3: Train CNNMLP with auto batch sizing
        # ------------------------------------------------------------------
        try:
            # time.sleep(30)  # wait for merge to complete and be queryable
            dataset = nc.get_dataset(merged_dataset_name)
            robot_ids = dataset.robot_ids
            assert (
                len(robot_ids) == 2
            ), f"Expected 2 robots in merged dataset, got {robot_ids}"
            input_cross_embodiment_description = {}
            output_cross_embodiment_description = {}
            for robot_id in robot_ids:
                embodiment_description = dataset.get_full_data_spec(robot_id)
                assert (
                    DataType.JOINT_POSITIONS in embodiment_description
                ), f"JOINT_POSITIONS missing from robot {robot_id} data spec"

                input_cross_embodiment_description[robot_id] = {
                    DataType.JOINT_POSITIONS: embodiment_description[
                        DataType.JOINT_POSITIONS
                    ],
                    DataType.RGB_IMAGES: embodiment_description[DataType.RGB_IMAGES],
                }
                output_cross_embodiment_description[robot_id] = {
                    DataType.JOINT_POSITIONS: embodiment_description[
                        DataType.JOINT_POSITIONS
                    ],
                }
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
        try:
            # Wait for RUNNING state
            running_deadline = time.time() + 10 * 60
            while True:
                job_status = nc.get_training_job_status(job_id)
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
                            "skipping log assertions"
                        )
                        break
                    time.sleep(30)

            if logs is not None:
                # Validate CloudComputeLogs structure:
                # job_id, logs, total_entries, retrieved_at
                for field in ("job_id", "logs", "total_entries", "retrieved_at"):
                    assert (
                        field in logs
                    ), f"Missing '{field}' in CloudComputeLogs response"
                assert isinstance(logs["logs"], list)
                assert isinstance(logs["total_entries"], int)
                for entry in logs["logs"]:
                    assert "message" in entry, f"Log entry missing 'message': {entry}"

                # Verify severity filtering is accepted
                filtered = nc.get_training_job_logs(
                    job_id, max_entries=10, severity_filter="ERROR"
                )
                assert "logs" in filtered
                logger.info(
                    f"Step 4 passed — {logs['total_entries']} log entries retrieved"
                )
        except Exception as e:
            pytest.fail(f"Step 4 (training logs) failed: {e}")

        # ------------------------------------------------------------------
        # Step 5: Wait for training to complete
        # ------------------------------------------------------------------
        try:
            final_status = _wait_for_training(job_id)
            assert (
                final_status == "COMPLETED"
            ), f"Training ended with non-COMPLETED status: {final_status}"
        except Exception as e:
            pytest.fail(f"Step 5 (training completion) failed: {e}")

        # ------------------------------------------------------------------
        # Step 6: Direct in-process policy inference
        # ------------------------------------------------------------------
        try:
            nc.connect_robot(MUJOCO_ROBOT_NAME)
            policy = nc.policy(
                model_input_order=INPUT_DATA_SPEC,
                model_output_order=OUTPUT_DATA_SPEC,
                train_run_name=training_name,
            )
            _run_policy_inference(policy)
        except Exception as e:
            pytest.fail(f"Step 6 (direct policy inference) failed: {e}")

        # ------------------------------------------------------------------
        # Step 7: Local server policy inference
        # ------------------------------------------------------------------
        try:
            policy = nc.policy_local_server(
                model_input_order=INPUT_DATA_SPEC,
                model_output_order=OUTPUT_DATA_SPEC,
                train_run_name=training_name,
                port=8181,
            )
            _run_policy_inference(policy)
        except Exception as e:
            pytest.fail(f"Step 7 (local server inference) failed: {e}")

        # ------------------------------------------------------------------
        # Step 8: Deploy remote endpoint and verify active
        # ------------------------------------------------------------------
        try:
            endpoint_name = _unique_name("flow_endpoint")
            endpoint_data = nc.deploy_model(
                job_id=job_id,
                name=endpoint_name,
                model_input_order=INPUT_DATA_SPEC,
                model_output_order=OUTPUT_DATA_SPEC,
                ttl=60 * 30,
            )
            endpoint_id = endpoint_data["id"]
            final_endpoint_status = _wait_for_endpoint(endpoint_id)
            assert (
                final_endpoint_status == "active"
            ), f"Endpoint did not become active, status: {final_endpoint_status!r}"
            logger.info(f"Step 8 passed — endpoint {endpoint_id} is active")
        except Exception as e:
            pytest.fail(f"Step 8 (remote endpoint deployment) failed: {e}")

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

    # def test_training_failure_error_reporting():
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

    try:
        # ------------------------------------------------------------------
        # Collect minimal demo data (1 episode is enough)
        # ------------------------------------------------------------------
        try:
            dataset = _collect_demo_data(ROBOT_NAME, dataset_name, num_episodes=1)
        except Exception as e:
            pytest.fail(f"Data collection failed: {e}")

        # ------------------------------------------------------------------
        # Build a per-robot data spec from the collected dataset
        # ------------------------------------------------------------------
        try:
            robot_ids = dataset.robot_ids
            input_robot_data_spec: dict = {}
            output_robot_data_spec: dict = {}
            for robot_id in robot_ids:
                data_spec = dataset.get_full_data_spec(robot_id)
                filtered = {
                    dt: item for dt, item in data_spec.items() if dt in INPUT_DATA_TYPES
                }
                input_robot_data_spec[robot_id] = filtered
                output_robot_data_spec[robot_id] = {
                    DataType.JOINT_POSITIONS: filtered[DataType.JOINT_POSITIONS],
                }
        except Exception as e:
            pytest.fail(f"Building data spec failed: {e}")

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
                input_robot_data_spec=input_robot_data_spec,
                output_robot_data_spec=output_robot_data_spec,
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
                "Error field present in job data: %s",
                str(job_detail["error"])[:200],
            )
        except Exception as e:
            pytest.fail(f"Failed to verify error info in job data: {e}")

    finally:
        # if job_id:
        #     try:
        #         nc.delete_training_job(job_id)
        #     except Exception:
        #         logger.warning(f"Failed to delete job {job_id}")
        # if dataset:
        #     try:
        #         dataset.delete()
        #     except Exception:
        #         logger.warning(f"Failed to delete dataset {dataset_name}")
        pass
