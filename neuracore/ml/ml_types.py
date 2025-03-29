from typing import NamedTuple

import torch


class MaskableData:

    def __init__(self, data: torch.FloatTensor, mask: torch.FloatTensor):
        self.data = data
        self.mask = mask

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return MaskableData(
            data=_to_device(self.data, device), mask=_to_device(self.mask, device)
        )


class ActionMaskableData(MaskableData):
    def __init__(
        self,
        data: torch.FloatTensor,
        mask: torch.FloatTensor,
        sequence_mask: torch.FloatTensor,
    ):
        super().__init__(data, mask)
        self.sequence_mask = sequence_mask

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return ActionMaskableData(
            data=_to_device(self.data, device),
            mask=_to_device(self.mask, device),
            sequence_mask=_to_device(self.sequence_mask, device),
        )


def _to_device(data: MaskableData, device: torch.device):
    return data.to(device) if data is not None else None


class BatchedTrainingSamples:

    def __init__(
        self,
        actions: ActionMaskableData = None,
        joint_positions: MaskableData = None,
        joint_velocities: MaskableData = None,
        joint_torques: MaskableData = None,
        gripper_states: MaskableData = None,
        rgb_images: MaskableData = None,
        depth_images: MaskableData = None,
        point_clouds: MaskableData = None,
        custom_data: dict[str, MaskableData] = None,
    ):
        self.actions = actions
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.gripper_states = gripper_states
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.point_clouds = point_clouds
        self.custom_data = custom_data or {}

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedTrainingSamples(
            actions=_to_device(self.actions, device),
            joint_positions=_to_device(self.joint_positions, device),
            joint_velocities=_to_device(self.joint_velocities, device),
            joint_torques=_to_device(self.joint_torques, device),
            gripper_states=_to_device(self.gripper_states, device),
            rgb_images=_to_device(self.rgb_images, device),
            depth_images=_to_device(self.depth_images, device),
            point_clouds=_to_device(self.point_clouds, device),
            custom_data={
                key: _to_device(value, device)
                for key, value in self.custom_data.items()
            },
        )

    def __len__(self):
        if self.joint_positions is not None:
            return self.joint_positions.data.size(0)
        if self.rgb_images is not None:
            return self.rgb_images.data.size(0)
        if self.actions is not None:
            return self.actions.data.size(0)
        raise ValueError("No tensor found in the batch input")


class BatchedTrainingOutputs:
    def __init__(
        self,
        action_predicitons: torch.FloatTensor,
        losses: dict[str, torch.FloatTensor],
        metrics: dict[str, torch.FloatTensor],
    ):
        self.action_predicitons = action_predicitons
        self.losses = losses
        self.metrics = metrics


class BatchedInferenceSamples:

    def __init__(
        self,
        joint_positions: MaskableData = None,
        joint_velocities: MaskableData = None,
        joint_torques: MaskableData = None,
        gripper_states: MaskableData = None,
        rgb_images: MaskableData = None,
        depth_images: MaskableData = None,
        point_clouds: MaskableData = None,
        custom_data: dict[str, MaskableData] = None,
    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.gripper_states = gripper_states
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.point_clouds = point_clouds
        self.custom_data = custom_data or {}

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedTrainingSamples(
            joint_positions=_to_device(self.joint_positions, device),
            joint_velocities=_to_device(self.joint_velocities, device),
            joint_torques=_to_device(self.joint_torques, device),
            gripper_states=_to_device(self.gripper_states, device),
            rgb_images=_to_device(self.rgb_images, device),
            depth_images=_to_device(self.depth_images, device),
            point_clouds=_to_device(self.point_clouds, device),
            custom_data={
                key: _to_device(value, device)
                for key, value in self.custom_data.items()
            },
        )

    def __len__(self):
        if self.joint_positions is not None:
            return self.joint_positions.data.size(0)
        if self.rgb_images is not None:
            return self.rgb_images.data.size(0)
        raise ValueError("No tensor found in the batch input")


# This has to be a NamedTuple because of torchscript
class BatchedInferenceOutputs(NamedTuple):
    action_predicitons: torch.FloatTensor
