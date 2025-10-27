import argparse

import neuracore as nc
from neuracore.core.nc_types import DataType


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
        default="NVIDIA_L4",
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
        default="Example Dataset",
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
        default=10,
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--output_prediction_horizon",
        type=int,
        default=50,
        help="Prediction horizon.",
    )
    parser.add_argument(
        "--output_data_types",
        type=list[DataType],
        default=[DataType.JOINT_TARGET_POSITIONS],
        help="Output data types to use for training.",
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
    # Here, algorithm specific configs can be added.
    # Uses default values, if not defined.
    algorithm_config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "output_prediction_horizon": args.output_prediction_horizon,
    }

    job_data = nc.start_training_run(
        name=training_name,
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        frequency=frequency,
        algorithm_name=algorithm_name,
        dataset_name=dataset_name,
        algorithm_config=algorithm_config,
    )

    print("Training job started")
    print(f"Training job data: {job_data}")
