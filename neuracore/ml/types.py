from typing import NamedTuple

import torch


class DatasetDescription:

    def __init__(
        self,
        max_num_cameras: int,
        max_state_size: int,
        max_action_size: int,
        action_mean: torch.FloatTensor,
        action_std: torch.FloatTensor,
        state_mean: torch.FloatTensor,
        state_std: torch.FloatTensor,
        action_prediction_horizon: int = 1,
    ):
        self.max_num_cameras = max_num_cameras
        self.max_state_size = max_state_size
        self.max_action_size = max_action_size
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_prediction_horizon = action_prediction_horizon

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return DatasetDescription(
            max_num_cameras=self.max_num_cameras,
            max_state_size=self.max_state_size,
            max_action_size=self.max_action_size,
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
            state_mean=self.state_mean.to(device),
            state_std=self.state_std.to(device),
            action_prediction_horizon=self.action_prediction_horizon,
        )


class BatchedTrainingSamples:

    def __init__(
        self,
        states: torch.FloatTensor,
        states_mask: torch.FloatTensor,
        camera_images: torch.FloatTensor,
        camera_images_mask: torch.FloatTensor,
        actions: torch.FloatTensor,
        actions_mask: torch.FloatTensor,
        actions_sequence_mask: torch.FloatTensor,
    ):
        self.states = states
        self.states_mask = states_mask
        self.camera_images = camera_images
        self.camera_images_mask = camera_images_mask
        self.actions = actions
        self.actions_mask = actions_mask
        self.actions_sequence_mask = actions_sequence_mask

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedTrainingSamples(
            states=self.states.to(device),
            states_mask=self.states_mask.to(device),
            camera_images=self.camera_images.to(device),
            camera_images_mask=self.camera_images_mask.to(device),
            actions=self.actions.to(device),
            actions_mask=self.actions_mask.to(device),
            actions_sequence_mask=self.actions_sequence_mask.to(device),
        )

    def __len__(self):
        if self.states is not None:
            return self.states.size(0)
        if self.camera_images is not None:
            return self.camera_images.size(0)
        if self.actions is not None:
            return self.actions.size(0)
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
        states: torch.FloatTensor,
        states_mask: torch.FloatTensor,
        camera_images: torch.FloatTensor,
        camera_images_mask: torch.FloatTensor,
    ):
        self.states = states
        self.states_mask = states_mask
        self.camera_images = camera_images
        self.camera_images_mask = camera_images_mask

    def to(self, device: torch.device):
        """Move all tensors to the specified device."""
        return BatchedInferenceSamples(
            states=self.states.to(device),
            states_mask=self.states_mask.to(device),
            camera_images=self.camera_images.to(device),
            camera_images_mask=self.camera_images_mask.to(device),
        )

    def __len__(self):
        if self.states is not None:
            return self.states.size(0)
        if self.camera_images is not None:
            return self.camera_images.size(0)
        raise ValueError("No tensor found in the batch input")


# This has to be a NamedTuple because of torchscript
class BatchedInferenceOutputs(NamedTuple):
    action_predicitons: torch.FloatTensor
