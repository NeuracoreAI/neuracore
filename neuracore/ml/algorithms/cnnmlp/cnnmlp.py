"""A simple CNN for each camera using a pretrained resnet18 followed by MLP."""

import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from neuracore.core.nc_types import DataType, ModelInitDescription, ModelPrediction
from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import ImageEncoder


class CNNMLP(NeuracoreModel):
    """CNN+MLP model with single timestep input and sequence output."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 512,
        cnn_output_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_init_description)
        self.hidden_dim = hidden_dim
        self.cnn_output_dim = cnn_output_dim
        self.num_layers = num_layers
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.image_encoders = nn.ModuleList([
            ImageEncoder(output_dim=self.cnn_output_dim)
            for _ in range(self.dataset_description.max_num_rgb_images)
        ])

        state_input_dim = (
            self.dataset_description.joint_positions.max_len
            + self.dataset_description.joint_velocities.max_len
            + self.dataset_description.joint_torques.max_len
        )
        self.state_embed = None
        hidden_state_dim = 0
        if state_input_dim > 0:
            hidden_state_dim = hidden_dim
            self.state_embed = nn.Linear(state_input_dim, hidden_dim)

        mlp_input_dim = (
            self.dataset_description.max_num_rgb_images * cnn_output_dim
            + hidden_state_dim
        )

        self.action_data_type = self.model_init_description.output_data_types[0]
        self.output_prediction_horizon = self.output_prediction_horizon
        if DataType.JOINT_TARGET_POSITIONS == self.action_data_type:
            action_data_item_stats = self.dataset_description.joint_target_positions
        else:
            action_data_item_stats = self.dataset_description.joint_positions
        self.max_output_size = action_data_item_stats.max_len

        # Predict entire sequence at once
        self.output_size = self.max_output_size * self.output_prediction_horizon
        self.mlp = self._build_mlp(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_size,
            num_layers=num_layers,
        )

        self.transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        state_mean = np.concatenate([
            self.dataset_description.joint_positions.mean,
            self.dataset_description.joint_velocities.mean,
            self.dataset_description.joint_torques.mean,
        ])
        state_std = np.concatenate([
            self.dataset_description.joint_positions.std,
            self.dataset_description.joint_velocities.std,
            self.dataset_description.joint_torques.std,
        ])
        self.joint_state_mean = self._to_torch_float_tensor(state_mean)
        self.joint_state_std = self._to_torch_float_tensor(state_std)
        self.action_mean = self._to_torch_float_tensor(action_data_item_stats.mean)
        self.action_std = self._to_torch_float_tensor(action_data_item_stats.std)

    def _to_torch_float_tensor(self, data: list[float]) -> torch.FloatTensor:
        """Convert list of floats to torch tensor."""
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _build_mlp(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> nn.Sequential:
        """Construct MLP."""
        if num_layers == 1:
            return nn.Sequential(nn.Linear(input_dim, output_dim))

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Added normalization
            nn.Dropout(0.1),  # Added dropout
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            ])

        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def _preprocess_joint_state(
        self, joint_state: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Preprocess the states."""
        return (joint_state - self.joint_state_mean) / self.joint_state_std

    def _preprocess_actions(self, actions: torch.FloatTensor) -> torch.FloatTensor:
        """Preprocess the actions."""
        return (actions - self.action_mean) / self.action_std

    def _predict_action(self, batch: BatchedInferenceSamples) -> torch.FloatTensor:
        """Predict action for the given batch."""
        batch_size = batch.joint_positions.data.shape[0]

        # Process images from each camera
        image_features = []
        for cam_id, encoder in enumerate(self.image_encoders):
            features = encoder(self.transform(batch.rgb_images.data[:, cam_id]))
            features *= batch.rgb_images.mask[:, cam_id : cam_id + 1]
            image_features.append(features)

        # Combine image features
        if image_features:
            combined_image_features = torch.cat(image_features, dim=-1)
        else:
            combined_image_features = torch.zeros(
                batch_size, self.cnn_output_dim, device=self.device, dtype=torch.float32
            )

        combined_features = combined_image_features
        if self.state_embed is not None:
            state_inputs = []
            if batch.joint_positions:
                state_inputs.append(
                    batch.joint_positions.data * batch.joint_positions.mask
                )
            if batch.joint_velocities:
                state_inputs.append(
                    batch.joint_velocities.data * batch.joint_velocities.mask
                )
            if batch.joint_torques:
                state_inputs.append(batch.joint_torques.data * batch.joint_torques.mask)
            joint_states = torch.cat(state_inputs, dim=-1)
            joint_states = self._preprocess_joint_state(joint_states)
            state_features = self.state_embed(joint_states)
            combined_features = torch.cat(
                [state_features, combined_image_features], dim=-1
            )

        # Forward through MLP to get entire sequence
        mlp_out = self.mlp(combined_features)

        action_preds = mlp_out.view(
            batch_size, self.output_prediction_horizon, self.max_output_size
        )
        return action_preds

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Forward pass for inference."""
        t = time.time()
        action_preds = self._predict_action(batch)
        prediction_time = time.time() - t
        predictions = (action_preds * self.action_std) + self.action_mean
        predictions = predictions.detach().cpu().numpy()
        return ModelPrediction(
            outputs={self.action_data_type: predictions},
            prediction_time=prediction_time,
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step."""
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
        )
        if self.action_data_type == DataType.JOINT_TARGET_POSITIONS:
            action_data = batch.outputs.joint_target_positions.data
        else:
            action_data = batch.outputs.joint_positions.data

        target_actions = self._preprocess_actions(action_data)
        action_predicitons = self._predict_action(inference_sample)
        losses, metrics = {}, {}
        if self.training:
            losses["mse_loss"] = nn.functional.mse_loss(
                action_predicitons, target_actions
            )
        return BatchedTrainingOutputs(
            output_predicitons=action_predicitons,
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure and return optimizer for the model."""
        backbone_params = []
        other_params = []

        for name, param in self.named_parameters():
            if "image_encoders" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": other_params, "lr": self.lr},
        ]
        return [torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)]

    @staticmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]
