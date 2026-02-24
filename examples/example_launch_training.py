"""This example demonstrates how you can launch a training job
from the Neuracore platform."""

import argparse

from common.base_env import BimanualViperXTask
from neuracore_types import DataSpec, DataType, RobotDataSpec

import neuracore as nc

MODEL_OUTPUT_ORDER: DataSpec = {
    DataType.JOINT_TARGET_POSITIONS: (
        BimanualViperXTask.LEFT_ARM_JOINT_NAMES
        + BimanualViperXTask.RIGHT_ARM_JOINT_NAMES
    ),
}


def create_parser():
    """Create argument parser with parameters."""
    parser = argparse.ArgumentParser(description="Launching a training job.")

    parser.add_argument(
        "--name",
        type=str,
        default="My Training Job",
        help="Name of the training job.",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="NVIDIA_TESLA_V100",
        help="Type of GPU to use for training.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training.",
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=50,
        help="Frequency of training.",
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="CNNMLP",
        help="Name of the algorithm to use for training.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Transfer Cube VX300s Dataset",
        help="Name of the dataset to use for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--output_prediction_horizon",
        type=int,
        default=50,
        help="Prediction horizon.",
    )
    return parser


if __name__ == "__main__":
    nc.login()

    parser = create_parser()
    args = parser.parse_args()
    training_name = args.name
    gpu_type = args.gpu_type
    num_gpus = args.num_gpus
    frequency = args.frequency
    algorithm_name = args.algorithm_name
    dataset_name = args.dataset_name

    dataset = nc.get_dataset(dataset_name)
    robot_id = dataset.robot_ids[0]

    # Here, algorithm specific configs can be added.
    # Uses default values, if not defined.
    algorithm_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "output_prediction_horizon": args.output_prediction_horizon,
    }

    input_robot_data_spec: RobotDataSpec = {
        robot_id: {
            nc.DataType.RGB_IMAGES: ["angle"],
        }
    }

    output_robot_data_spec: RobotDataSpec = {
        robot_id: MODEL_OUTPUT_ORDER,
    }

    job_data = nc.start_training_run(
        name=training_name,
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        frequency=frequency,
        algorithm_name=algorithm_name,
        dataset_name=dataset_name,
        algorithm_config=algorithm_config,
        input_robot_data_spec=input_robot_data_spec,
        output_robot_data_spec=output_robot_data_spec,
    )

    print("Training job started")
    print(f"Training job data: {job_data}")
