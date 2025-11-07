import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import pytest

import neuracore as nc
from neuracore.core.endpoint import Policy
from neuracore.core.nc_types import CameraData, JointData, SyncPoint

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "..", "..", "examples"))
# ruff: noqa: E402
from common.transfer_cube import BOX_POSE, TransferCubeTask, make_sim_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPISODE_LENGTH: int = 400
CAM_NAME = "angle"
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
                sync_point = SyncPoint(
                    joint_positions=JointData(values=obs.qpos),
                    rgb_images={
                        CAM_NAME: CameraData(frame=obs.cameras[CAM_NAME].rgb),
                    },
                )
                predicted_sync_points = policy.predict(
                    sync_point=sync_point, timeout=10
                )
                joint_target_positions = [
                    sp.joint_target_positions for sp in predicted_sync_points
                ]
                actions = [
                    jtp.numpy(order=env.ACTION_KEYS)
                    for jtp in joint_target_positions
                    if jtp is not None
                ]
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
    "algorithm_name, input_data_types, output_data_types, epochs, min_success_rate",
    [
        (
            "CNNMLP",
            [
                nc.DataType.RGB_IMAGE,
                nc.DataType.JOINT_POSITIONS,
            ],
            [
                nc.DataType.JOINT_TARGET_POSITIONS,
            ],
            200,
            0.1,  # CNNMLP is not that powerful, so low bar
        ),
        (
            "ACT",
            [
                nc.DataType.RGB_IMAGE,
                nc.DataType.JOINT_POSITIONS,
            ],
            [
                nc.DataType.JOINT_TARGET_POSITIONS,
            ],
            50,
            0.5,
        ),
        (
            "DiffusionPolicy",
            [
                nc.DataType.RGB_IMAGE,
                nc.DataType.JOINT_POSITIONS,
            ],
            [
                nc.DataType.JOINT_TARGET_POSITIONS,
            ],
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
        input_data_types: list,
        output_data_types: list,
        epochs: int,
        min_success_rate: float,
    ) -> None:
        nc.login()

        algorithm_config = {
            "batch_size": BATCH_SIZE,
            "epochs": epochs,
            "output_prediction_horizon": OUTPUT_PREDICTION_HORIZON,
        }
        logger.info("Starting training job...")
        job_data = nc.start_training_run(
            name=f"{TRAINING_NAME} - {algorithm_name}",
            gpu_type=GPU_TYPE,
            num_gpus=NUM_GPUS,
            frequency=FREQUENCY,
            algorithm_name=algorithm_name,
            dataset_name=DATASET_NAME,
            algorithm_config=algorithm_config,
            input_data_types=input_data_types,
            output_data_types=output_data_types,
        )
        logger.info("Training job started!")
        training_job_id = job_data["id"]
        training_timeout_time = time.time() + TRAINING_TIMEOUT_MINUTES * 60

        training_job_status = nc.get_training_job_status(training_job_id)
        while training_job_status in ["preparing_data", "pending", "running"]:
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

        if training_job_status != "completed":
            raise ValueError(
                f"Training job did not complete and is in status: {training_job_status}"
            )

        endpoint_name = f"{ENDPOINT_NAME} - {algorithm_name}"
        endpoint_id = None
        try:
            endpoint_data = nc.deploy_model(
                job_id=training_job_id,
                name=endpoint_name,
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
