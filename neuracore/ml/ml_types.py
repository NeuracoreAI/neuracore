import torch


class MaskableData:

    def __init__(self, data: torch.FloatTensor, mask: torch.FloatTensor):
        self.data = data
        self.mask = mask

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return MaskableData(
            data=_to_device(self.data, device),
            mask=_to_device(self.mask, device),
        )


def _to_device(data, device: torch.device):
    return data.to(device) if data is not None else None


class BatchedData:

    def __init__(
        self,
        joint_positions: MaskableData = None,
        joint_velocities: MaskableData = None,
        joint_torques: MaskableData = None,
        joint_target_positions: MaskableData = None,
        gripper_states: MaskableData = None,
        rgb_images: MaskableData = None,
        depth_images: MaskableData = None,
        point_clouds: MaskableData = None,
        language_tokens: MaskableData = None,
        custom_data: dict[str, MaskableData] = None,
    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.joint_target_positions = joint_target_positions
        self.gripper_states = gripper_states
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.point_clouds = point_clouds
        self.language_tokens = language_tokens
        self.custom_data = custom_data or {}

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedData(
            joint_positions=_to_device(self.joint_positions, device),
            joint_velocities=_to_device(self.joint_velocities, device),
            joint_torques=_to_device(self.joint_torques, device),
            joint_target_positions=_to_device(self.joint_target_positions, device),
            gripper_states=_to_device(self.gripper_states, device),
            rgb_images=_to_device(self.rgb_images, device),
            depth_images=_to_device(self.depth_images, device),
            point_clouds=_to_device(self.point_clouds, device),
            language_tokens=_to_device(self.language_tokens, device),
            custom_data={
                key: _to_device(value, device)
                for key, value in self.custom_data.items()
            },
        )

    def __len__(self):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, MaskableData) and attr_value.data is not None:
                return attr_value.data.size(0)
        raise ValueError("No tensor found in the batch input")


class BatchedTrainingSamples:

    def __init__(
        self,
        inputs: BatchedData = None,
        outputs: BatchedData = None,
        output_predicition_mask: torch.FloatTensor = None,
    ):
        self.inputs = inputs or BatchedData()
        self.outputs = outputs or BatchedData()
        self.output_predicition_mask = output_predicition_mask

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedTrainingSamples(
            inputs=self.inputs.to(device),
            outputs=self.outputs.to(device),
            output_predicition_mask=(
                self.output_predicition_mask.to(device)
                if self.output_predicition_mask is not None
                else None
            ),
        )

    def __len__(self):
        return len(self.inputs)


class BatchedTrainingOutputs:
    def __init__(
        self,
        output_predicitons: torch.FloatTensor,
        losses: dict[str, torch.FloatTensor],
        metrics: dict[str, torch.FloatTensor],
    ):
        self.output_predictions = output_predicitons
        self.losses = losses
        self.metrics = metrics


class BatchedInferenceSamples:

    def __init__(
        self,
        joint_positions: MaskableData = None,
        joint_velocities: MaskableData = None,
        joint_torques: MaskableData = None,
        joint_target_positions: MaskableData = None,
        gripper_states: MaskableData = None,
        rgb_images: MaskableData = None,
        depth_images: MaskableData = None,
        point_clouds: MaskableData = None,
        language_tokens: MaskableData = None,
        custom_data: dict[str, MaskableData] = None,
    ):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities
        self.joint_torques = joint_torques
        self.joint_target_positions = joint_target_positions
        self.gripper_states = gripper_states
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.point_clouds = point_clouds
        self.language_tokens = language_tokens
        self.custom_data = custom_data or {}

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedInferenceSamples(
            joint_positions=_to_device(self.joint_positions, device),
            joint_velocities=_to_device(self.joint_velocities, device),
            joint_torques=_to_device(self.joint_torques, device),
            joint_target_positions=_to_device(self.joint_target_positions, device),
            gripper_states=_to_device(self.gripper_states, device),
            rgb_images=_to_device(self.rgb_images, device),
            depth_images=_to_device(self.depth_images, device),
            point_clouds=_to_device(self.point_clouds, device),
            language_tokens=_to_device(self.language_tokens, device),
            custom_data={
                key: _to_device(value, device)
                for key, value in self.custom_data.items()
            },
        )

    def __len__(self):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, MaskableData) and attr_value.data is not None:
                return attr_value.data.size(0)
        raise ValueError("No tensor found in the batch input")
