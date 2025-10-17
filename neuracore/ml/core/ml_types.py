"""Machine learning data types for robot learning models.

This module provides data structures for handling batched robot sensor data
with support for masking, device placement, and multi-modal inputs including
joint states, images, point clouds, poses, end-effectors, and language tokens.
"""

from typing import Optional

import torch


class MaskableData:
    """Container for tensor data with associated mask for variable-length sequences.

    Provides a unified interface for handling data that may have variable lengths
    or missing values, commonly used in robot learning for handling sequences
    of different lengths or optional sensor modalities.
    """

    def __init__(self, data: torch.FloatTensor, mask: torch.FloatTensor):
        """Initialize maskable data container.

        Args:
            data: Main data tensor
            mask: Boolean mask tensor indicating valid data positions
        """
        self.data = data
        self.mask = mask

    def to(self, device: torch.device) -> "MaskableData":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            MaskableData: New instance with tensors moved to the specified device
        """
        return MaskableData(
            data=_to_device(self.data, device),
            mask=_to_device(self.mask, device),
        )


def _to_device(
    data: Optional[MaskableData], device: torch.device
) -> Optional[MaskableData]:
    """Utility function to move data to device, handling None values.

    Args:
        data: Data to move (can be None)
        device: Target device

    Returns:
        Data moved to device, or None if input was None
    """
    return data.to(device) if data is not None else None


class BatchedData:
    """Container for batched multi-modal robot sensor data.

    Provides a structured way to handle various types of robot sensor data
    in batched format, including joint states, visual data, poses, end-effectors,
    and custom sensor modalities with support for device placement.
    """

    def __init__(
        self,
        joint_positions: Optional[MaskableData] = None,
        joint_velocities: Optional[MaskableData] = None,
        joint_torques: Optional[MaskableData] = None,
        joint_target_positions: Optional[MaskableData] = None,
        end_effectors: Optional[MaskableData] = None,
        end_effector_poses: Optional[MaskableData] = None,
        parallel_gripper_open_amounts: Optional[MaskableData] = None,
        poses: Optional[MaskableData] = None,
        rgb_images: Optional[MaskableData] = None,
        depth_images: Optional[MaskableData] = None,
        point_clouds: Optional[MaskableData] = None,
        language_tokens: Optional[MaskableData] = None,
        custom_data: Optional[dict[str, MaskableData]] = None,
    ):
        """Initialize batched data container.

        Args:
            joint_positions: Joint position data with mask
            joint_velocities: Joint velocity data with mask
            joint_torques: Joint torque data with mask
            joint_target_positions: Target joint position data with mask
            end_effectors: End-effector state data with mask
            end_effector_poses: 7DOF end-effector pose data with mask
            parallel_gripper_open_amounts: Parallel gripper open amount data with mask
            poses: 7DOF pose data with mask
            rgb_images: RGB image data with mask
            depth_images: Depth image data with mask
            point_clouds: Point cloud data with mask
            language_tokens: Language token data with mask
            custom_data: Dictionary of custom sensor data with masks
        """
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.joint_target_positions = joint_target_positions
        self.end_effectors = end_effectors
        self.end_effector_poses = end_effector_poses
        self.parallel_gripper_open_amounts = parallel_gripper_open_amounts
        self.poses = poses
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.point_clouds = point_clouds
        self.language_tokens = language_tokens
        self.custom_data = custom_data or {}

    def to(self, device: torch.device) -> "BatchedData":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            BatchedData: New instance with all tensors moved to the specified device
        """
        return BatchedData(
            joint_positions=_to_device(self.joint_positions, device),
            joint_velocities=_to_device(self.joint_velocities, device),
            joint_torques=_to_device(self.joint_torques, device),
            joint_target_positions=_to_device(self.joint_target_positions, device),
            end_effectors=_to_device(self.end_effectors, device),
            end_effector_poses=_to_device(self.end_effector_poses, device),
            parallel_gripper_open_amounts=_to_device(
                self.parallel_gripper_open_amounts, device
            ),
            poses=_to_device(self.poses, device),
            rgb_images=_to_device(self.rgb_images, device),
            depth_images=_to_device(self.depth_images, device),
            point_clouds=_to_device(self.point_clouds, device),
            language_tokens=_to_device(self.language_tokens, device),
            custom_data={
                key: moved_value
                for key, value in self.custom_data.items()
                if (moved_value := _to_device(value, device)) is not None
            },
        )

    def __len__(self) -> int:
        """Get the batch size from the first available tensor.

        Returns:
            int: Batch size (first dimension of available tensors)

        Raises:
            ValueError: If no tensors are found in the batch
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, MaskableData) and attr_value.data is not None:
                return attr_value.data.size(0)
            if isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, MaskableData) and value.data is not None:
                        return value.data.size(0)
        raise ValueError("No tensor found in the batch input")


class BatchedTrainingSamples:
    """Container for batched training samples with inputs and target outputs.

    Provides structured access to training data including input features,
    target outputs, and prediction masks for supervised learning scenarios.
    """

    def __init__(
        self,
        output_prediction_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[BatchedData] = None,
        outputs: Optional[BatchedData] = None,
    ):
        """Initialize batched training samples.

        Args:
            inputs: Input data for the model
            outputs: Target output data for supervision
            output_prediction_mask: Mask indicating which outputs to predict
        """
        self.inputs = inputs or BatchedData()
        self.outputs = outputs or BatchedData()
        self.output_prediction_mask = output_prediction_mask

    def to(self, device: torch.device) -> "BatchedTrainingSamples":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            BatchedTrainingSamples: New instance with tensors moved to device
        """
        return BatchedTrainingSamples(
            inputs=self.inputs.to(device),
            outputs=self.outputs.to(device),
            output_prediction_mask=(
                self.output_prediction_mask.to(device)
                if self.output_prediction_mask is not None
                else None
            ),
        )

    def __len__(self) -> int:
        """Get the batch size from the input data.

        Returns:
            int: Batch size
        """
        return len(self.inputs)


class BatchedTrainingOutputs:
    """Container for training step outputs including predictions, losses, and metrics.

    Provides structured access to the results of a training step including
    model predictions, computed losses, and evaluation metrics.
    """

    def __init__(
        self,
        output_predictions: torch.FloatTensor,
        losses: dict[str, torch.FloatTensor],
        metrics: dict[str, torch.FloatTensor],
    ):
        """Initialize batched training outputs.

        Args:
            output_predictions: Model predictions for the batch
            losses: Dictionary of named loss values
            metrics: Dictionary of named evaluation metrics
        """
        self.output_predictions = output_predictions
        self.losses = losses
        self.metrics = metrics


class BatchedInferenceSamples:
    """Container for batched inference samples.

    Provides structured access to input data for model inference,
    supporting all robot sensor modalities with device placement.
    """

    def __init__(
        self,
        joint_positions: Optional[MaskableData] = None,
        joint_velocities: Optional[MaskableData] = None,
        joint_torques: Optional[MaskableData] = None,
        joint_target_positions: Optional[MaskableData] = None,
        end_effectors: Optional[MaskableData] = None,
        end_effector_poses: Optional[MaskableData] = None,
        parallel_gripper_open_amounts: Optional[MaskableData] = None,
        poses: Optional[MaskableData] = None,
        rgb_images: Optional[MaskableData] = None,
        depth_images: Optional[MaskableData] = None,
        point_clouds: Optional[MaskableData] = None,
        language_tokens: Optional[MaskableData] = None,
        custom_data: Optional[dict[str, MaskableData]] = None,
    ):
        """Initialize batched inference samples.

        Args:
            joint_positions: Joint position data with mask
            joint_velocities: Joint velocity data with mask
            joint_torques: Joint torque data with mask
            joint_target_positions: Target joint position data with mask
            end_effectors: End-effector state data with mask
            end_effector_poses: 7DOF end-effector pose data with mask
            parallel_gripper_open_amounts: Parallel gripper open amount data with mask
            poses: 6DOF pose data with mask
            rgb_images: RGB image data with mask
            depth_images: Depth image data with mask
            point_clouds: Point cloud data with mask
            language_tokens: Language token data with mask
            custom_data: Dictionary of custom sensor data with masks
        """
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.joint_target_positions = joint_target_positions
        self.end_effectors = end_effectors
        self.end_effector_poses = end_effector_poses
        self.parallel_gripper_open_amounts = parallel_gripper_open_amounts
        self.poses = poses
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.point_clouds = point_clouds
        self.language_tokens = language_tokens
        self.custom_data = custom_data or {}

    def to(self, device: torch.device) -> "BatchedInferenceSamples":
        """Move all tensors to the specified device.

        Args:
            device: Target device for tensor placement

        Returns:
            BatchedInferenceSamples: New instance with tensors moved to device
        """
        return BatchedInferenceSamples(
            joint_positions=_to_device(self.joint_positions, device),
            joint_velocities=_to_device(self.joint_velocities, device),
            joint_torques=_to_device(self.joint_torques, device),
            joint_target_positions=_to_device(self.joint_target_positions, device),
            end_effectors=_to_device(self.end_effectors, device),
            end_effector_poses=_to_device(self.end_effector_poses, device),
            parallel_gripper_open_amounts=_to_device(
                self.parallel_gripper_open_amounts, device
            ),
            poses=_to_device(self.poses, device),
            rgb_images=_to_device(self.rgb_images, device),
            depth_images=_to_device(self.depth_images, device),
            point_clouds=_to_device(self.point_clouds, device),
            language_tokens=_to_device(self.language_tokens, device),
            custom_data={
                key: moved_value
                for key, value in self.custom_data.items()
                if (moved_value := _to_device(value, device)) is not None
            },
        )

    def __len__(self) -> int:
        """Get the batch size from the first available tensor.

        Returns:
            int: Batch size (first dimension of available tensors)

        Raises:
            ValueError: If no tensors are found in the batch
        """
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, MaskableData) and attr_value.data is not None:
                return attr_value.data.size(0)
        raise ValueError("No tensor found in the batch input")
