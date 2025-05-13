import base64
import io
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader

import neuracore as nc

from ...core.nc_types import (
    CameraData,
    DataType,
    JointData,
    ModelInitDescription,
    SyncPoint,
)
from ..ml_types import BatchedTrainingOutputs, BatchedTrainingSamples, MaskableData
from .algorithm_loader import AlgorithmLoader
from .dummy_dataset import DummyDataset
from .mar import create_mar


class AlgorthmCheck(BaseModel):
    successfully_loaded_file: bool = False
    successfully_initialized_model: bool = False
    successfully_configured_optimizer: bool = False
    successfully_forward_pass: bool = False
    successfully_backward_pass: bool = False
    successfully_optimiser_step: bool = False
    successfully_exported_model: bool = False
    successfully_launched_endpoint: bool = False


def setup_logging(output_dir: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "validate.log"),
        ],
    )


def _encode_image(image: np.ndarray) -> str:
    """Encode image as base64 string."""
    pil_image = Image.fromarray(image.astype("uint8"))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _create_joint_data(maskable_data: MaskableData) -> JointData:
    """Convert MaskableData to JointData."""
    t = time.time()
    return JointData(
        timestamp=t,
        values={
            f"joint{i}": v
            for i, v in enumerate(maskable_data.data[0].cpu().numpy().tolist())
        },
    )


def run_validation(
    output_dir: Path,
    algorithm_dir: Path,
    port: int = 8080,
    skip_endpoint_check: bool = False,
) -> tuple[AlgorthmCheck, str]:
    """Run the minimal validation process to check if an algorithm works."""

    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    algo_check = AlgorthmCheck()
    error_msg = ""
    try:
        logger.info("Starting algorithm validation")

        # Load the algorithm model class
        logger.info("Loading algorithm model class")
        algorithm_loader = AlgorithmLoader(algorithm_dir)
        model_class = algorithm_loader.load_model()

        logger.info(f"Loaded model class: {model_class.__name__}")
        algo_check.successfully_loaded_file = True

        supported_input_data_types: list[DataType] = (
            model_class.get_supported_input_data_types()
        )
        supported_output_data_types: list[DataType] = (
            model_class.get_supported_output_data_types()
        )

        dataset = DummyDataset(
            num_samples=5,
            input_data_types=supported_input_data_types,
            output_data_types=supported_output_data_types,
            tokenize_text=model_class.tokenize_text,
        )

        # Create a minimal dataloader
        batch_size = 2  # Small batch size for quick testing
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model_init_description = ModelInitDescription(
            dataset_description=dataset.dataset_description,
            input_data_types=supported_input_data_types,
            output_data_types=supported_output_data_types,
            output_prediction_horizon=dataset.output_prediction_horizon,
        )

        # Check 1: Can initialize the model
        logger.info("Initializing model")
        algorithm_config = {}  # Use the dafault configuration
        model = model_class(
            model_init_description=model_init_description, **algorithm_config
        ).to(device)
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
        batch = batch.to(device)

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
        loss = sum(outputs.losses.values()).mean()
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

        # Check 4: Can export to TorchScript
        logger.info("Testing TorchScript export")
        with tempfile.TemporaryDirectory():
            try:
                artifacts_dir = output_dir
                create_mar(model, artifacts_dir)

                algo_check.successfully_exported_model = True
                logger.info("TorchScript export successful")

            except Exception as e:
                logger.error(f"TorchScript export failed: {str(e)}")
                raise ValueError(f"Model cannot be exported to TorchScript: {str(e)}")

            if skip_endpoint_check:
                algo_check.successfully_launched_endpoint = True
            else:
                policy = None
                try:
                    # Check if the exported model can be loaded
                    policy = nc.connect_local_endpoint(
                        path_to_model=str(artifacts_dir / "model.mar"),
                        port=port,
                    )

                    # Log some data to send to the model
                    sync_point = SyncPoint(timestamp=time.time())
                    if batch.inputs.joint_positions:
                        sync_point.joint_positions = _create_joint_data(
                            batch.inputs.joint_positions
                        )
                    if batch.inputs.joint_velocities:
                        sync_point.joint_velocities = _create_joint_data(
                            batch.inputs.joint_velocities
                        )
                    if batch.inputs.joint_torques:
                        sync_point.joint_torques = _create_joint_data(
                            batch.inputs.joint_torques
                        )
                    if batch.inputs.joint_target_positions:
                        sync_point.joint_target_positions = _create_joint_data(
                            batch.inputs.joint_target_positions
                        )
                    if batch.inputs.rgb_images:
                        rgbs = (
                            batch.inputs.rgb_images.data[0]
                            .cpu()
                            .numpy()
                            .transpose(0, 2, 3, 1)
                            * 255
                        ).astype(np.uint8)
                        rgbs = {
                            f"camera{i}": CameraData(
                                timestamp=time.time(), frame=_encode_image(v)
                            )
                            for i, v in enumerate(rgbs)
                        }
                        sync_point.rgb_images = rgbs

                    action = policy.predict(sync_point)
                    logger.info(f"Exported model loaded successfully, action: {action}")
                    policy.disconnect()
                    algo_check.successfully_launched_endpoint = True
                except Exception as e:
                    if policy:
                        policy.disconnect()
                    logger.error(f"Failed to load exported model: {str(e)}")
                    raise ValueError(f"Model cannot be loaded from export: {str(e)}")

        # All checks passed!
        logger.info("All validation checks passed successfully")

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        try:
            error_msg = str(e)
        except Exception as inner_e:
            logger.error(f"Failed to save error details: {str(inner_e)}")

    return algo_check, error_msg
