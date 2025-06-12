"""CNN+MLP model for robot manipulation with sequence prediction.

This module implements a simple baseline model that combines convolutional
neural networks for visual feature extraction with multi-layer perceptrons
for action sequence prediction. The model processes single timestep inputs
and outputs entire action sequences.
"""

import time
from typing import Any, Dict

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
    """CNN+MLP model with single timestep input and sequence output.

    A baseline model architecture that uses separate CNN encoders for each
    camera view, combines visual features with proprioceptive state, and
    predicts entire action sequences through a multi-layer perceptron.

    The model processes current observations and outputs a fixed-length
    sequence of future actions, making it suitable for action chunking
    approaches in robot manipulation.
    """

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
        """Initialize the CNN+MLP model.

        Args:
            model_init_description: Model initialization parameters
            hidden_dim: Hidden dimension for MLP layers
            cnn_output_dim: Output dimension for CNN encoders
            num_layers: Number of MLP layers
            lr: Learning rate for main parameters
            lr_backbone: Learning rate for CNN backbone
            weight_decay: Weight decay for optimizer
        """
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
        """Convert list of floats to torch tensor on the correct device.

        Args:
            data: List of float values

        Returns:
            torch.FloatTensor: Tensor on the model's device
        """
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _build_mlp(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ) -> nn.Sequential:
        """Construct multi-layer perceptron with normalization and dropout.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers

        Returns:
            nn.Sequential: Constructed MLP module
        """
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
        """Normalize joint state using dataset statistics.

        Args:
            joint_state: Raw joint state tensor

        Returns:
            torch.FloatTensor: Normalized joint state
        """
        return (joint_state - self.joint_state_mean) / self.joint_state_std

    def _preprocess_actions(self, actions: torch.FloatTensor) -> torch.FloatTensor:
        """Normalize actions using dataset statistics.

        Args:
            actions: Raw action tensor

        Returns:
            torch.FloatTensor: Normalized actions
        """
        return (actions - self.action_mean) / self.action_std

    def _predict_action(self, batch: BatchedInferenceSamples) -> torch.FloatTensor:
        """Predict action sequence for the given batch.

        Processes visual and proprioceptive inputs through separate encoders,
        combines features, and predicts the entire action sequence through MLP.

        Args:
            batch: Input batch with observations

        Returns:
            torch.FloatTensor: Predicted action sequence [B, T, action_dim]
        """
        batch_size = len(batch)

        # Process images from each camera
        image_features = []
        if batch.rgb_images is not None:
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
        """Perform inference to predict action sequence.

        Args:
            batch: Input batch with observations

        Returns:
            ModelPrediction: Model predictions with timing information
        """
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
        """Perform a single training step.

        Predicts action sequences and computes mean squared error loss
        against target actions.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
        )

        if self.action_data_type == DataType.JOINT_TARGET_POSITIONS:
            assert (
                batch.outputs.joint_target_positions is not None
            ), "joint_target_positions required"
            action_data = batch.outputs.joint_target_positions.data
        else:
            assert batch.outputs.joint_positions is not None, "joint_positions required"
            action_data = batch.outputs.joint_positions.data

        target_actions = self._preprocess_actions(action_data)
        action_predicitons = self._predict_action(inference_sample)
        losses: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
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
        """Configure optimizer with different learning rates for different components.

        Uses separate learning rates for image encoder backbones (typically lower)
        and other model parameters.

        Returns:
            list[torch.optim.Optimizer]: List containing the configured optimizer
        """
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
        """Get the input data types supported by this model.

        Returns:
            list[DataType]: List of supported input data types
        """
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Get the output data types supported by this model.

        Returns:
            list[DataType]: List of supported output data types
        """
        return [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]
