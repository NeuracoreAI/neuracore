"""Algorithm validation system for Neuracore model development and deployment.

This module provides comprehensive validation testing for Neuracore algorithms
including model loading, training pipeline verification, export functionality,
and deployment readiness checks. It ensures algorithms are compatible with
the Neuracore training and inference infrastructure.
"""

import logging
import tempfile
import time
import traceback
from pathlib import Path

import torch
from neuracore_types import (
    DATA_TYPE_TO_NC_DATA_CLASS,
    BatchedNCData,
    DataType,
    ModelInitDescription,
    NCData,
    SynchronizedPoint,
)
from pydantic import BaseModel
from torch.utils.data import DataLoader

import neuracore as nc
from neuracore.ml.utils.device_utils import get_default_device

from ..core.ml_types import BatchedTrainingOutputs, BatchedTrainingSamples
from ..datasets.pytorch_dummy_dataset import MAX_LEN_PER_DATA_TYPE, PytorchDummyDataset
from .algorithm_loader import AlgorithmLoader
from .nc_archive import create_nc_archive


class AlgorithmCheck(BaseModel):
    """Validation results tracking the success of each algorithm check.

    This class tracks the status of various validation steps to provide
    detailed feedback on which parts of the algorithm validation passed
    or failed during testing.
    """

    successfully_loaded_file: bool = False
    successfully_initialized_model: bool = False
    successfully_configured_optimizer: bool = False
    successfully_forward_pass: bool = False
    successfully_backward_pass: bool = False
    successfully_optimiser_step: bool = False
    successfully_exported_model: bool = False
    successfully_launched_endpoint: bool = False


def setup_logging(output_dir: Path) -> None:
    """Configure logging for validation process with file and console output.

    Sets up logging to capture validation progress and errors both in the
    console and in a log file for debugging purposes.

    Args:
        output_dir: Directory where the validation log file will be created.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "validate.log"),
        ],
    )


def run_validation(
    output_dir: Path,
    algorithm_dir: Path,
    port: int = 8080,
    skip_endpoint_check: bool = False,
    algorithm_config: dict = {},
    device: torch.device | None = None,
) -> tuple[AlgorithmCheck, str]:
    """Run comprehensive validation tests on a Neuracore algorithm.

    Performs a series of validation checks to ensure the algorithm is
    compatible with Neuracore's training and inference infrastructure.
    Tests include model loading, training pipeline, export functionality,
    and deployment readiness.

    Args:
        output_dir: Directory where validation artifacts and logs will be saved.
        algorithm_dir: Directory containing the algorithm code to validate.
        port: TCP port to use for local endpoint testing.
        skip_endpoint_check: Whether to skip the endpoint deployment test.
            Useful for faster validation when deployment testing isn't needed.
        algorithm_config: Custom configuration arguments for the algorithm.
        device: Torch device to run the validation on (e.g., 'cpu' or 'cuda').

    Returns:
        A tuple containing:
        - AlgorithmCheck object with detailed results of each validation step
        - Error message string if validation failed, empty string if successful

    Raises:
        ValueError: If the algorithm directory contains no Python files or
            if critical validation steps fail.
    """
    nc.stop_live_data()

    device = device or get_default_device()

    # find the first folder that contains Python files
    python_files = list(algorithm_dir.rglob("*.py"))
    if not python_files:
        raise ValueError(
            f"No Python files found in the algorithm directory: {algorithm_dir}"
        )
    # Get parent directories and find the one with minimum number of parts
    algorithm_dir = min([f.parent for f in python_files], key=lambda d: len(d.parts))

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    algo_check = AlgorithmCheck()
    error_msg = ""
    try:
        logger.info("Starting algorithm validation")

        # Load the algorithm model class
        logger.info("Loading algorithm model class")
        algorithm_loader = AlgorithmLoader(algorithm_dir)
        model_class = algorithm_loader.load_model()

        logger.info(f"Loaded model class: {model_class.__name__}")
        algo_check.successfully_loaded_file = True

        supported_input_data_types = model_class.get_supported_input_data_types()
        supported_output_data_types = model_class.get_supported_output_data_types()

        logger.info(f"Supported input data types: {supported_input_data_types}")
        logger.info(f"Supported output data types: {supported_output_data_types}")

        # Create dummy robot data specs
        input_robot_data_spec = {
            "robot_1": {
                dt: [f"{dt.value}_{i}" for i in range(MAX_LEN_PER_DATA_TYPE)]
                for dt in supported_input_data_types
            }
        }
        output_robot_data_spec = {
            "robot_1": {
                dt: [f"{dt.value}_{i}" for i in range(MAX_LEN_PER_DATA_TYPE)]
                for dt in supported_output_data_types
            }
        }

        dataset = PytorchDummyDataset(
            input_robot_data_spec=input_robot_data_spec,
            output_robot_data_spec=output_robot_data_spec,
            num_samples=5,
        )

        # Create a minimal dataloader
        batch_size = 2  # Small batch size for quick testing
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )

        model_init_description = ModelInitDescription(
            input_data_types=supported_input_data_types,
            output_data_types=supported_output_data_types,
            dataset_statistics=dataset._dataset_statistics,
            output_prediction_horizon=dataset.output_prediction_horizon,
        )

        # Check 1: Can initialize the model
        logger.info("Initializing model")
        model = model_class(
            model_init_description=model_init_description,
            **algorithm_config,
        )
        model = model.to(device)
        logger.info(
            "Model initialized with "
            f"{sum(p.numel() for p in model.parameters()):,} parameters"
        )
        algo_check.successfully_initialized_model = True

        # Check 2: Can configure optimizer
        logger.info("Configuring optimizer")
        optimizers = model.configure_optimizers()
        logger.info("Optimizer configured successfully")
        algo_check.successfully_configured_optimizer = True

        # Check 3: Can do a forward and backward pass
        logger.info("Testing forward and backward pass")
        model.train()

        # Get a batch from the dataloader
        batch: BatchedTrainingSamples = next(iter(dataloader))
        batch = batch.to(model.device)

        # Forward pass
        for optimizer in optimizers:
            optimizer.zero_grad()
        outputs: BatchedTrainingOutputs = model.training_step(batch)

        # Ensure loss is calculated
        if len(outputs.losses) == 0:
            raise ValueError(
                "Model output does not contain a loss. "
                "Forward pass must return a BatchOutput object with at least one loss."
            )

        # Sum all losses
        loss = torch.stack(list(outputs.losses.values())).sum(0).mean()
        logger.info(f"Forward pass successful, loss: {loss.item():.4f}")
        algo_check.successfully_forward_pass = True

        # Backward pass
        loss.backward()
        logger.info("Backward pass successful")
        algo_check.successfully_backward_pass = True

        # Check if gradients were calculated
        has_grad = any(
            p.grad is not None and torch.sum(torch.abs(p.grad)) > 0
            for p in model.parameters()
            if p.requires_grad
        )
        if not has_grad:
            raise ValueError("No gradients were calculated during backward pass")

        # Optimizer step
        for optimizer in optimizers:
            optimizer.step()
        logger.info("Optimizer step successful")
        algo_check.successfully_optimiser_step = True

        # Check 4: Can export to NC archive
        logger.info("Testing NC archive export")
        with tempfile.TemporaryDirectory():
            try:
                artifacts_dir = output_dir
                create_nc_archive(model, artifacts_dir, algorithm_config)

                algo_check.successfully_exported_model = True
                logger.info("NC archive export successful")

            except Exception as e:
                raise ValueError(f"Model cannot be exported to NC archive: {str(e)}")

            if skip_endpoint_check:
                algo_check.successfully_launched_endpoint = True
            else:
                policy = None
                try:
                    model_input_order = {
                        dt: [f"{dt.value}_{i}" for i in range(MAX_LEN_PER_DATA_TYPE)]
                        for dt in supported_input_data_types
                    }
                    model_output_order = {
                        dt: [f"{dt.value}_{i}" for i in range(MAX_LEN_PER_DATA_TYPE)]
                        for dt in supported_output_data_types
                    }

                    # Check if the exported model can be loaded
                    policy = nc.policy_local_server(
                        model_input_order=model_input_order,
                        model_output_order=model_output_order,
                        model_file=str(artifacts_dir / "model.nc.zip"),
                        port=port,
                        device=str(device),
                    )

                except Exception:
                    if policy is not None:
                        policy.disconnect()
                    raise ValueError(
                        f"Failed to connect to local endpoint on port {port}."
                    )

                try:
                    t = time.time()
                    sync_data: dict[DataType, dict[str, NCData]] = {}
                    for data_type, list_batched_nc_data in batch.inputs.items():
                        names = model_input_order[data_type]
                        sync_data[data_type] = {}
                        for name in names:
                            sync_data[data_type][name] = DATA_TYPE_TO_NC_DATA_CLASS[
                                data_type
                            ].sample()

                    sync_point = SynchronizedPoint(
                        timestamp=t, robot_id=dataset.robot.id, data=sync_data
                    )

                    # Test the policy prediction
                    prediction: dict[DataType, dict[str, BatchedNCData]] = (
                        policy.predict(sync_point)
                    )
                    logger.info(
                        f"Exported model loaded successfully, prediction: {prediction}"
                    )

                    if len(prediction) == 0:
                        raise ValueError("Policy prediction returned no data.")

                    for data_type, data_dict in prediction.items():
                        if len(data_dict) == 0:
                            raise ValueError(
                                "Policy prediction returned no data for "
                                f"data type {data_type}."
                            )
                        for name, batched_nc_data in data_dict.items():
                            if not isinstance(batched_nc_data, BatchedNCData):
                                raise ValueError(
                                    "Policy prediction did not return a "
                                    "BatchedNCData object"
                                )
                            if not batched_nc_data.model_dump():
                                raise ValueError(
                                    "BatchedNCData object returned by "
                                    "policy prediction is empty"
                                )

                    policy.disconnect()
                    algo_check.successfully_launched_endpoint = True

                except Exception:
                    if policy:
                        policy.disconnect()
                    raise ValueError("Failed to get prediction from local endpoint:")

        # All checks passed!
        logger.info("✓ All validation checks passed successfully")

    except Exception as e:
        error_msg = f"Validation failed: {str(e)}\n"
        error_msg += traceback.format_exc()
        logger.error("Validation failed.", exc_info=True)

    return algo_check, error_msg
