"""Integration tests verifying per-algorithm success rates on the Transfer Cube task.

The suite is split into two phases so CI runners are not held idle during training:

  test_start_training  — submits a training job and records the job ID.
  test_evaluate        — deploys an endpoint for the completed job, runs
                         NUM_ROLLOUTS in MuJoCo, and checks the success rate
                         against the threshold in algorithm_configs.yaml.

Algorithm names, hyperparameters, and thresholds all live in algorithm_configs.yaml.
To add or remove an algorithm, edit only that file.

Running locally
---------------
    # Phase 1 — kick off training (job ID is printed in the log output)
    ALGORITHM_NAME=ACT pytest -k test_start_training -v <this file>

    # Phase 2 — evaluate once training is complete
    ALGORITHM_NAME=ACT TRAINING_JOB_ID=<id> pytest -k test_evaluate -v <this file>

    # Running the full file without TRAINING_JOB_ID: test_start_training runs for
    # all algorithms, test_evaluate skips cleanly for each.
"""

import logging
import os
import sys
import time
from typing import cast

import matplotlib.pyplot as plt
import pytest
import torch
from neuracore_types import (
    BatchedJointData,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc
from neuracore.core.endpoint import Policy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "..", "examples"))
# ruff: noqa: E402
from common.transfer_cube import (
    BOX_POSE,
    BimanualViperXTask,
    TransferCubeTask,
    make_sim_env,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPISODE_LENGTH: int = 400
NC_CAM_NAME = "rgb_angle"
MJ_CAM_NAME = "angle"
MAX_REWARD = 4.0
ENDPOINT_NAME = "Integration Test Endpoint"
TRAINING_NAME = "Integration Test"
DATASET_NAME = "Transfer Cube VX300s Dataset"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 50
NUM_ROLLOUTS = 20
ONSCREEN_RENDER = False
# Training should be complete by the time Phase 2 runs; this is a safety buffer
# for jobs that finish slightly after the Phase 2 workflow starts.
TRAINING_WAIT_TIMEOUT_MINUTES = 60

JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES
    + BimanualViperXTask.LEFT_GRIPPER_JOINT_NAMES
    + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    + BimanualViperXTask.RIGHT_GRIPPER_JOINT_NAMES
)

INPUT_DATA_SPEC = {
    DataType.RGB_IMAGES: [NC_CAM_NAME],
    DataType.JOINT_POSITIONS: JOINT_NAMES,
}
OUTPUT_DATA_SPEC = {
    DataType.JOINT_TARGET_POSITIONS: BimanualViperXTask.ACTION_KEYS,
}


def eval_model(
    policy: Policy,
    env: TransferCubeTask,
    num_rollouts: int,
    onscreen_render: bool = False,
) -> float:
    plt_img = None
    success = 0
    for episode_idx in range(num_rollouts):
        logger.info(f"Starting rollout {episode_idx + 1} / {num_rollouts}")
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[MJ_CAM_NAME].rgb)
            plt.ion()

        episode_max = 0
        horizon = 1
        actions = []

        for i in range(EPISODE_LENGTH):
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                sync_point = SynchronizedPoint(
                    data={
                        DataType.JOINT_POSITIONS: {
                            name: JointData(value=obs.qpos[name])
                            for name in JOINT_NAMES
                        },
                        DataType.RGB_IMAGES: {
                            NC_CAM_NAME: RGBCameraData(
                                frame=obs.cameras[MJ_CAM_NAME].rgb
                            ),
                        },
                    },
                )
                predictions = policy.predict(sync_point=sync_point, timeout=10)
                joint_target_positions = cast(
                    dict[str, BatchedJointData],
                    predictions[DataType.JOINT_TARGET_POSITIONS],
                )

                left_arm = torch.cat(
                    [
                        joint_target_positions[n].value
                        for n in BimanualViperXTask.LEFT_ARM_JOINT_NAMES
                    ],
                    dim=2,
                )
                right_arm = torch.cat(
                    [
                        joint_target_positions[n].value
                        for n in BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
                    ],
                    dim=2,
                )
                left_gripper = joint_target_positions[
                    BimanualViperXTask.LEFT_GRIPPER_OPEN
                ].value
                right_gripper = joint_target_positions[
                    BimanualViperXTask.RIGHT_GRIPPER_OPEN
                ].value

                batched_actions = (
                    torch.cat([left_arm, left_gripper, right_arm, right_gripper], dim=2)
                    .cpu()
                    .numpy()
                )
                actions = batched_actions[0]
                horizon = len(actions)

            a = actions[idx_in_horizon]
            obs, reward, done = env.step(a)
            episode_max = max(episode_max, reward)

            if onscreen_render:
                assert plt_img is not None
                plt_img.set_data(obs.cameras[MJ_CAM_NAME].rgb)
                plt.pause(0.002)

        if onscreen_render:
            plt.close()

        if episode_max >= MAX_REWARD:
            success += 1

    return success / num_rollouts


class TestAlgorithmPerformance:
    def test_start_training(self, algorithm_config_entry: dict) -> None:
        """Phase 1: start a training job and record its ID.

        In CI this writes the job ID to $GITHUB_OUTPUT so the Phase 2 workflow
        can cache and forward it. Locally the ID is logged at INFO level.
        """
        algorithm_name = algorithm_config_entry["name"]

        nc.login()

        dataset = nc.get_dataset(DATASET_NAME)
        robot_ids = dataset.robot_ids
        assert len(robot_ids) == 1, f"Expected one robot in dataset, got {robot_ids}"
        robot_id = robot_ids[0]

        timestamp = int(time.time())
        logger.info(f"[{algorithm_name}] Starting training job...")
        job_data = nc.start_training_run(
            name=f"{TRAINING_NAME} - {algorithm_name} - {timestamp}",
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            algorithm_name=algorithm_name,
            dataset_name=DATASET_NAME,
            algorithm_config=algorithm_config_entry["algorithm_config"],
            input_robot_data_spec={robot_id: INPUT_DATA_SPEC},
            output_robot_data_spec={robot_id: OUTPUT_DATA_SPEC},
        )
        training_job_id = job_data["id"]
        logger.info(f"[{algorithm_name}] Training job started: {training_job_id}")

        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"training_job_id={training_job_id}\n")

    def test_evaluate(self, algorithm_config_entry: dict) -> None:
        """Phase 2: deploy an endpoint for a completed training job and evaluate it.

        Expects TRAINING_JOB_ID to be set in the environment (written by
        test_start_training in Phase 1 and passed through the CI cache).
        Skips gracefully when TRAINING_JOB_ID is absent so that running the
        full test file locally does not fail on this test.
        """
        training_job_id = os.environ.get("TRAINING_JOB_ID")
        if not training_job_id:
            pytest.skip(
                "TRAINING_JOB_ID not set — run test_start_training first, "
                "then re-run with TRAINING_JOB_ID=<id>"
            )

        algorithm_name = algorithm_config_entry["name"]
        min_success_rate = algorithm_config_entry["min_success_rate"]

        nc.login()

        # Training should already be complete; poll briefly as a safety buffer.
        training_job_status = nc.get_training_job_status(training_job_id)
        training_wait_start = time.time()
        while training_job_status in ["PREPARING_DATA", "PENDING", "RUNNING"]:
            logger.info(
                f"[{algorithm_name}] Waiting for training: "
                f"status={training_job_status}"
            )
            time.sleep(60)
            training_job_status = nc.get_training_job_status(training_job_id)
            elapsed_minutes = (time.time() - training_wait_start) / 60
            if elapsed_minutes > TRAINING_WAIT_TIMEOUT_MINUTES:
                raise TimeoutError(
                    f"[{algorithm_name}] Training still not complete after "
                    f"{TRAINING_WAIT_TIMEOUT_MINUTES} minutes in Phase 2 — "
                    "training may have overrun its scheduled window"
                )

        if training_job_status != "COMPLETED":
            raise ValueError(
                f"[{algorithm_name}] Training job did not complete, "
                f"status: {training_job_status}"
            )

        timestamp = int(time.time())
        endpoint_name = f"{ENDPOINT_NAME} - {algorithm_name} - {timestamp}"
        endpoint_id = None
        try:
            endpoint_data = nc.deploy_model(
                job_id=training_job_id,
                name=endpoint_name,
                model_input_order=INPUT_DATA_SPEC,
                model_output_order=OUTPUT_DATA_SPEC,
                ttl=60 * 30,
            )
            endpoint_id = endpoint_data["id"]

            endpoint_status = nc.get_endpoint_status(endpoint_id=endpoint_id)
            while endpoint_status == "creating":
                logger.info(
                    f"[{algorithm_name}] Waiting for endpoint: "
                    f"status={endpoint_status}"
                )
                time.sleep(60)
                endpoint_status = nc.get_endpoint_status(endpoint_id=endpoint_id)

            if endpoint_status != "active":
                raise ValueError(
                    f"[{algorithm_name}] Endpoint did not become active: "
                    f"{endpoint_status}"
                )

            nc.connect_robot("Mujoco VX300s")
            policy = nc.policy_remote_server(endpoint_name)
            env = make_sim_env(seed=42)
            success_rate = eval_model(
                policy=policy,
                env=env,
                num_rollouts=NUM_ROLLOUTS,
                onscreen_render=ONSCREEN_RENDER,
            )
            policy.disconnect()
        except Exception:
            if endpoint_id is not None:
                nc.delete_endpoint(endpoint_id)
            raise

        logger.info(f"[{algorithm_name}] success_rate={success_rate:.2%}")
        nc.delete_endpoint(endpoint_id)
        nc.delete_training_job(training_job_id)

        if success_rate < min_success_rate:
            raise ValueError(
                f"[{algorithm_name}] success rate {success_rate:.2%} is below "
                f"the minimum threshold of {min_success_rate:.2%}"
            )
