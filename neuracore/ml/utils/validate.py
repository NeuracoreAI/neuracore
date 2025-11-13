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
from typing import Optional

import numpy as np
import torch
from neuracore_types import (
    CameraData,
    CustomData,
    DataType,
    EndEffectorData,
    EndEffectorPoseData,
    JointData,
    LanguageData,
    ModelInitDescription,
    ParallelGripperOpenAmountData,
    PointCloudData,
    PoseData,
    SyncPoint,
)
from pydantic import BaseModel
from torch.utils.data import DataLoader

import neuracore as nc
from neuracore.ml.utils.device_utils import get_default_device

from ..core.ml_types import BatchedTrainingOutputs, BatchedTrainingSamples, MaskableData
from ..datasets.pytorch_dummy_dataset import PytorchDummyDataset
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


def _create_joint_data(maskable_data: MaskableData) -> JointData:
    """Convert MaskableData to JointData format for testing.

    Transforms batch tensor data back to the individual data format
    used in the Neuracore API for validation testing.

    Args:
        maskable_data: Batched joint data from the training pipeline.

    Returns:
        JointData object with properly formatted joint values.
    """
    t = time.time()
    return JointData(
        timestamp=t,
        values={
            f"joint{i}": v
            for i, v in enumerate(maskable_data.data[0].cpu().numpy().tolist())
        },
    )


def _create_end_effector_data(maskable_data: MaskableData) -> EndEffectorData:
    """Convert MaskableData to EndEffectorData format for testing.

    Args:
        maskable_data: Batched end-effector data from the training pipeline.

    Returns:
        EndEffectorData object with properly formatted values.
    """
    t = time.time()
    return EndEffectorData(
        timestamp=t,
        open_amounts={
            f"gripper{i}": v
            for i, v in enumerate(maskable_data.data[0].cpu().numpy().tolist())
        },
    )


def _create_end_effector_pose_data(maskable_data: MaskableData) -> EndEffectorPoseData:
    """Convert MaskableData to EndEffectorPoseData format for testing.

    Args:
        maskable_data: Batched end-effector pose data from the training pipeline.

    Returns:
        EndEffectorPoseData object with properly formatted values.
    """
    t = time.time()
    return EndEffectorPoseData(
        timestamp=t,
        poses={
            f"end_effector{i}": v
            for i, v in enumerate(maskable_data.data[0].cpu().numpy().tolist())
        },
    )


def _create_parallel_gripper_open_amount_data(
    maskable_data: MaskableData,
) -> ParallelGripperOpenAmountData:
    """Convert MaskableData to ParallelGripperOpenAmountData format for testing.

    Args:
        maskable_data: Batched parallel gripper open amount
        data from the training pipeline.

    Returns:
        ParallelGripperOpenAmountData object with properly formatted values.
    """
    t = time.time()
    return ParallelGripperOpenAmountData(
        timestamp=t,
        open_amounts={
            f"parallel_gripper{i}": v
            for i, v in enumerate(maskable_data.data[0].cpu().numpy().tolist())
        },
    )


def _create_pose_data(maskable_data: MaskableData) -> PoseData:
    """Convert MaskableData to PoseData format for testing.

    Args:
        maskable_data: Batched pose data from the training pipeline.

    Returns:
        Dictionary of PoseData objects with 6DOF poses.
    """
    t = time.time()
    pose_values = maskable_data.data[0].cpu().numpy().tolist()

    # Group values into 6DOF poses (position + orientation)
    poses = {}
    for i in range(0, len(pose_values), 6):
        if i + 5 < len(pose_values):
            pose_name = f"pose{i // 6}"
            poses[pose_name] = pose_values[i : i + 6]

    return PoseData(timestamp=t, pose=poses)


def _create_point_cloud_data(maskable_data: MaskableData) -> dict[str, PointCloudData]:
    """Convert MaskableData to PointCloudData format for testing.

    Args:
        maskable_data: Batched point cloud data from the training pipeline.

    Returns:
        Dictionary of PointCloudData objects.
    """
    t = time.time()
    # Assuming point cloud data is [batch, num_clouds, num_points, 3]
    point_clouds = {}

    for cloud_idx in range(maskable_data.data.shape[1]):
        if maskable_data.mask[0, cloud_idx] > 0:  # Check if this cloud is valid
            points = maskable_data.data[0, cloud_idx].cpu().numpy().astype(np.float16)
            point_clouds[f"cloud{cloud_idx}"] = PointCloudData(
                timestamp=t, points=points
            )

    return point_clouds


def _create_language_data(maskable_data: MaskableData) -> LanguageData:
    """Convert MaskableData to LanguageData format for testing.

    Args:
        maskable_data: Batched language token data from the training pipeline.

    Returns:
        LanguageData object with sample text.
    """
    t = time.time()
    # For validation purposes, create a simple test instruction
    return LanguageData(timestamp=t, text="Move the robot arm to the target position")


def _create_custom_data(
    custom_data_dict: dict[str, MaskableData],
) -> dict[str, CustomData]:
    """Convert custom MaskableData to CustomData format for testing.

    Args:
        custom_data_dict: Dictionary of batched custom data from the training pipeline.

    Returns:
        Dictionary of CustomData objects.
    """
    t = time.time()
    result = {}

    for key, maskable_data in custom_data_dict.items():
        result[key] = CustomData(
            timestamp=t, data=maskable_data.data[0].cpu().numpy().tolist()
        )

    return result


def run_validation(
    output_dir: Path,
    algorithm_dir: Path,
    port: int = 8080,
    skip_endpoint_check: bool = False,
    algorithm_config: dict = {},
    device: Optional[torch.device] = None,
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

        supported_input_data_types: list[DataType] = (
            model_class.get_supported_input_data_types()
        )
        supported_output_data_types: list[DataType] = (
            model_class.get_supported_output_data_types()
        )

        logger.info(f"Supported input data types: {supported_input_data_types}")
        logger.info(f"Supported output data types: {supported_output_data_types}")

        dataset = PytorchDummyDataset(
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

        model_init_description = ModelInitDescription(
            dataset_description=dataset.dataset_description,
            input_data_types=supported_input_data_types,
            output_data_types=supported_output_data_types,
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
                    # Check if the exported model can be loaded
                    policy = nc.policy_local_server(
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
                    sync_point = SyncPoint(
                        timestamp=time.time(), robot_id=dataset.robot.id
                    )

                    # Add joint data
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

                    # Add end-effector pose data
                    if batch.inputs.end_effector_poses:
                        sync_point.end_effector_poses = _create_end_effector_pose_data(
                            batch.inputs.end_effector_poses
                        )

                    # Add parallel gripper open amount data
                    if batch.inputs.parallel_gripper_open_amounts:
                        sync_point.parallel_gripper_open_amounts = (
                            _create_parallel_gripper_open_amount_data(
                                batch.inputs.parallel_gripper_open_amounts
                            )
                        )

                    # Add end-effector data
                    if batch.inputs.end_effectors:
                        sync_point.end_effectors = _create_end_effector_data(
                            batch.inputs.end_effectors
                        )

                    # Add pose data
                    if batch.inputs.poses:
                        sync_point.poses = _create_pose_data(batch.inputs.poses)

                    # Add RGB images
                    if batch.inputs.rgb_images:
                        rgbs = (
                            batch.inputs.rgb_images.data[0]
                            .cpu()
                            .numpy()
                            .transpose(0, 2, 3, 1)
                            * 255
                        ).astype(np.uint8)
                        rgbs = {
                            f"camera{i}": CameraData(timestamp=time.time(), frame=v)
                            for i, v in enumerate(rgbs)
                        }
                        sync_point.rgb_images = rgbs

                    # Add depth images
                    if batch.inputs.depth_images:
                        depths = batch.inputs.depth_images.data[0].cpu().numpy()
                        # Handle depth images (they might be single channel)
                        if depths.ndim == 4 and depths.shape[1] == 1:
                            # Remove channel dimension and convert to uint8
                            depths = depths[:, 0, :, :]
                        elif depths.ndim == 4:
                            # Convert to grayscale if multi-channel
                            depths = np.mean(depths, axis=1)

                        # Normalize to 0-255 range
                        depths_normalized = []
                        for depth in depths:
                            depth_norm = (
                                (depth - depth.min())
                                / (depth.max() - depth.min() + 1e-8)
                                * 255
                            )
                            depths_normalized.append(depth_norm.astype(np.uint8))

                        depth_cameras = {
                            f"depth_camera{i}": CameraData(
                                timestamp=time.time(), frame=v
                            )
                            for i, v in enumerate(depths_normalized)
                        }
                        sync_point.depth_images = depth_cameras

                    # Add point clouds
                    if batch.inputs.point_clouds:
                        sync_point.point_clouds = _create_point_cloud_data(
                            batch.inputs.point_clouds
                        )

                    # Add language data
                    if batch.inputs.language_tokens:
                        sync_point.language_data = _create_language_data(
                            batch.inputs.language_tokens
                        )

                    # Add custom data
                    if batch.inputs.custom_data:
                        sync_point.custom_data = _create_custom_data(
                            batch.inputs.custom_data
                        )

                    # Test the policy prediction
                    action = policy.predict(sync_point)
                    logger.info(f"Exported model loaded successfully, action: {action}")

                    for pred_sync_point in action:
                        if not isinstance(pred_sync_point, SyncPoint):
                            raise ValueError(
                                "Policy prediction did not return a SyncPoint object"
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
