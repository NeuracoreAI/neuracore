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
    DataSpec,
    DataType,
    JointData,
    RGBCameraData,
    SynchronizedPoint,
)

import neuracore as nc
from neuracore.core.endpoint import Policy

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "examples"))
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
CAM_NAME = "rgb_angle"
MAX_REWARD = 4.0
ENDPOINT_NAME = "Integration Test Endpoint"
TRAINING_NAME = "Integration Test"
DATASET_NAME = "Transfer Cube VX300s Dataset"
GPU_TYPE = "NVIDIA_TESLA_V100"
NUM_GPUS = 1
FREQUENCY = 50
BATCH_SIZE = 32
OUTPUT_PREDICTION_HORIZON = 100
NUM_ROLLOUTS = 10
ONSCREEN_RENDER = False
TRAINING_TIMEOUT_MINUTES = 360

# Input includes finger joints (16 joints total)
JOINT_NAMES = (
    BimanualViperXTask.LEFT_ARM_JOINT_NAMES
    + BimanualViperXTask.LEFT_GRIPPER_JOINT_NAMES
    + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    + BimanualViperXTask.RIGHT_GRIPPER_JOINT_NAMES
)


def eval_model(
    policy: Policy,
    env: TransferCubeTask,
    num_rollouts: int,
    onscreen_render: bool = False,
):
    plt_img = None
    success = 0
    for episode_idx in range(num_rollouts):
        logger.info(f"Starting rollout {episode_idx + 1} / {num_rollouts}")
        # Setup the environment
        BOX_POSE[0] = env.sample_box_pose()
        obs = env.reset()

        # Setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(obs.cameras[CAM_NAME].rgb)
            plt.ion()

        episode_max = 0
        horizon = 1
        actions = []

        # Run episode
        for i in range(EPISODE_LENGTH):
            idx_in_horizon = i % horizon
            if idx_in_horizon == 0:
                # Create SynchronizedPoint with current observations
                sync_point = SynchronizedPoint(
                    data={
                        DataType.JOINT_POSITIONS: {
                            name: JointData(value=obs.qpos[name])
                            for name in JOINT_NAMES
                        },
                        DataType.RGB_IMAGES: {
                            CAM_NAME: RGBCameraData(frame=obs.cameras[CAM_NAME].rgb),
                        },
                    },
                )
                # Get predictions from the model
                predictions = policy.predict(sync_point=sync_point, timeout=10)
                joint_target_positions = cast(
                    dict[str, BatchedJointData],
                    predictions[DataType.JOINT_TARGET_POSITIONS],
                )

                # Build action array in correct order for env.step():
                # [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
                left_arm_names = BimanualViperXTask.LEFT_ARM_JOINT_NAMES
                right_arm_names = BimanualViperXTask.RIGHT_ARM_JOINT_NAMES

                left_arm = torch.cat(
                    [joint_target_positions[name].value for name in left_arm_names],
                    dim=2,
                )
                right_arm = torch.cat(
                    [joint_target_positions[name].value for name in right_arm_names],
                    dim=2,
                )

                # Get gripper from JOINT_TARGET_POSITIONS
                left_gripper = joint_target_positions[
                    BimanualViperXTask.LEFT_GRIPPER_OPEN
                ].value
                right_gripper = joint_target_positions[
                    BimanualViperXTask.RIGHT_GRIPPER_OPEN
                ].value

                # Concatenate: left_arm, left_gripper, right_arm, right_gripper
                batched_actions = (
                    torch.cat(
                        [left_arm, left_gripper, right_arm, right_gripper],
                        dim=2,
                    )
                    .cpu()
                    .numpy()
                )

                # Get first batch: (horizon, num_joints)
                actions = batched_actions[0]
                horizon = len(actions)
            a = actions[idx_in_horizon]
            obs, reward, done = env.step(a)
            episode_max = max(episode_max, reward)

            if onscreen_render:
                assert plt_img is not None
                plt_img.set_data(obs.cameras[CAM_NAME].rgb)
                plt.pause(0.002)

        if onscreen_render:
            plt.close()

        if episode_max >= MAX_REWARD:
            success += 1

    success_rate = success / num_rollouts
    return success_rate


@pytest.mark.parametrize(
    "algorithm_name, input_data_spec, output_data_spec, epochs, min_success_rate",  # noqa: E501
    [
        (
            "CNNMLP",
            {
                DataType.RGB_IMAGES: [CAM_NAME],
                DataType.JOINT_POSITIONS: JOINT_NAMES,
            },
            {
                DataType.JOINT_TARGET_POSITIONS: BimanualViperXTask.ACTION_KEYS,
            },
            50,
            0.5,
        ),
        (
            "ACT",
            {
                DataType.RGB_IMAGES: [CAM_NAME],
                DataType.JOINT_POSITIONS: JOINT_NAMES,
            },
            {
                DataType.JOINT_TARGET_POSITIONS: BimanualViperXTask.ACTION_KEYS,
            },
            50,
            0.5,
        ),
    ],
)
class TestAlgorithm:
    """A class with common parameters, `param1` and `param2`."""

    def test_start_training(
        self,
        algorithm_name: str,
        input_data_spec: DataSpec,
        output_data_spec: DataSpec,
        epochs: int,
        min_success_rate: float,
    ) -> None:
        nc.login()

        # Construct robot data specs
        dataset = nc.get_dataset(DATASET_NAME)
        robot_ids_dataset = dataset.robot_ids
        assert len(robot_ids_dataset) == 1, "Expected only one robot in the dataset"
        robot_id = robot_ids_dataset[0]
        input_robot_data_spec = {robot_id: input_data_spec}
        output_robot_data_spec = {robot_id: output_data_spec}

        # Timestamp used for unique naming
        timestamp = int(time.time())
        algorithm_config = {
            "batch_size": BATCH_SIZE,
            "epochs": epochs,
            "output_prediction_horizon": OUTPUT_PREDICTION_HORIZON,
        }
        logger.info("Starting training job...")
        job_data = nc.start_training_run(
            name=f"{TRAINING_NAME} - {algorithm_name} - {timestamp}",
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            algorithm_name=algorithm_name,
            dataset_name=DATASET_NAME,
            algorithm_config=algorithm_config,
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
        )
        logger.info("Training job started!")
        training_job_id = job_data["id"]
        training_timeout_time = time.time() + TRAINING_TIMEOUT_MINUTES * 60

        training_job_status = nc.get_training_job_status(training_job_id)
        while training_job_status in ["PREPARING_DATA", "PENDING", "RUNNING"]:
            logger.info(
                f"Waiting for training to finish, status: {training_job_status}"
            )
            time.sleep(60)
            training_job_status = nc.get_training_job_status(training_job_id)
            if time.time() > training_timeout_time:
                raise TimeoutError(
                    f"Training job did not complete within "
                    f"{TRAINING_TIMEOUT_MINUTES} minutes"
                )

        if training_job_status != "COMPLETED":
            raise ValueError(
                f"Training job did not complete and is in status: {training_job_status}"
            )

        endpoint_name = f"{ENDPOINT_NAME} - {algorithm_name} - {timestamp}"
        endpoint_id = None
        try:
            endpoint_data = nc.deploy_model(
                job_id=training_job_id,
                name=endpoint_name,
                model_input_order=input_data_spec,
                model_output_order=output_data_spec,
                ttl=60 * 30,  # 30 minutes
            )
            endpoint_id = endpoint_data["id"]
        except Exception as e:
            if endpoint_id is not None:
                nc.delete_endpoint(endpoint_id)
            raise e

        try:
            endpoint_status = nc.get_endpoint_status(endpoint_id=endpoint_id)
            while endpoint_status == "creating":
                logger.info(
                    f"Waiting for endpoint to finish, status: {endpoint_status}"
                )
                time.sleep(60)
                endpoint_status = nc.get_endpoint_status(endpoint_id=endpoint_id)
        except Exception as e:
            nc.delete_endpoint(endpoint_id)
            raise e

        if endpoint_status != "active":
            raise ValueError(f"Endpoint did not become active: {endpoint_status}")

        nc.connect_robot(
            robot_name="Mujoco VX300s",
        )

        try:
            policy = nc.policy_remote_server(endpoint_name)
            env = make_sim_env()
            success_rate = eval_model(
                policy=policy,
                env=env,
                num_rollouts=NUM_ROLLOUTS,
                onscreen_render=ONSCREEN_RENDER,
            )
            policy.disconnect()
        except Exception as e:
            nc.delete_endpoint(endpoint_id)
            raise e

        if success_rate < min_success_rate:
            raise ValueError(f"Success rate is too low: {success_rate}")

        logger.info(f"Success rate: {success_rate}")
        nc.delete_endpoint(endpoint_id)
        nc.delete_training_job(training_job_id)
