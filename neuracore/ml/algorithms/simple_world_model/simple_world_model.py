"""Simple world model for predicting future images given actions.

This module implements a world model that predicts future RGB images based on
current observations, robot state, and actions. The model uses U-Net architectures
conditioned on features to generate future visual states.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from neuracore_types import DataType, ModelInitDescription, ModelPrediction

from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import ImageEncoder, UNet


class SimpleWorldModel(NeuracoreModel):
    """World model that predicts future images based on multimodal inputs.

    This model predicts future RGB images conditioned on current visual
    observations, robot proprioceptive state, and planned actions. It uses
    separate U-Net architectures for each camera view, with conditioning
    features derived from encoded current images, state, and actions.

    The model is designed for world modeling in robot manipulation tasks,
    enabling model-based planning by predicting visual consequences of
    action sequences.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 512,
        cnn_output_dim: int = 128,
        num_layers: int = 3,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        """Initialize the simple world model.

        Args:
            model_init_description: Model initialization parameters
            hidden_dim: Hidden dimension for embedding layers
            cnn_output_dim: Output dimension for CNN encoders
            num_layers: Number of layers (not used in current implementation)
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

        # Image encoders for each camera
        self.image_encoders = nn.ModuleList([
            ImageEncoder(output_dim=self.cnn_output_dim)
            for _ in range(self.dataset_description.rgb_images.max_len)
        ])

        # Determine total state input dimension
        state_input_dim = (
            self.dataset_description.joint_positions.max_len
            + self.dataset_description.joint_velocities.max_len
            + self.dataset_description.joint_torques.max_len
        )

        # State embedding
        self.state_embed = None
        hidden_state_dim = 0
        if state_input_dim > 0:
            hidden_state_dim = hidden_dim
            self.state_embed = nn.Linear(state_input_dim, hidden_dim)

        # Action embedding
        action_input_dim = 0
        if (
            DataType.JOINT_TARGET_POSITIONS
            in self.model_init_description.input_data_types
        ):
            action_input_dim = self.dataset_description.joint_target_positions.max_len

        self.action_embed = None
        hidden_action_dim = 0
        if action_input_dim > 0:
            hidden_action_dim = hidden_dim
            self.action_embed = nn.Linear(action_input_dim, hidden_dim)

        # Create a UNet for each camera to predict future images
        condition_dim = hidden_state_dim + hidden_action_dim + cnn_output_dim
        self.image_predictors = nn.ModuleList([
            UNet(
                input_channels=3,  # RGB channels
                output_channels=3
                * self.model_init_description.output_prediction_horizon,  # RGB channels
                feature_map_sizes=[64, 128, 256, 512],
                condition_dim=condition_dim,  # state + action + image encoding
            )
            for _ in range(self.dataset_description.rgb_images.max_len)
        ])

        self.transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        self.inverse_transform = torch.nn.Sequential(
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        )

        # Normalization statistics
        self._setup_normalization_stats()

    def _setup_normalization_stats(self) -> None:
        """Setup normalization statistics for different data types."""
        # Joint state normalization
        state_means = []
        state_stds = []

        if DataType.JOINT_POSITIONS in self.model_init_description.input_data_types:
            state_means.extend(self.dataset_description.joint_positions.mean)
            state_stds.extend(self.dataset_description.joint_positions.std)
        if DataType.JOINT_VELOCITIES in self.model_init_description.input_data_types:
            state_means.extend(self.dataset_description.joint_velocities.mean)
            state_stds.extend(self.dataset_description.joint_velocities.std)
        if DataType.JOINT_TORQUES in self.model_init_description.input_data_types:
            state_means.extend(self.dataset_description.joint_torques.mean)
            state_stds.extend(self.dataset_description.joint_torques.std)

        if state_means:
            self.register_buffer(
                "joint_state_mean", self._to_torch_float_tensor(state_means)
            )
            self.register_buffer(
                "joint_state_std", self._to_torch_float_tensor(state_stds)
            )
        else:
            self.joint_state_mean = None
            self.joint_state_std = None

        if (
            DataType.JOINT_TARGET_POSITIONS
            in self.model_init_description.input_data_types
        ):
            self.register_buffer(
                "joint_target_mean",
                self._to_torch_float_tensor(
                    self.dataset_description.joint_target_positions.mean
                ),
            )
            self.register_buffer(
                "joint_target_std",
                self._to_torch_float_tensor(
                    self.dataset_description.joint_target_positions.std
                ),
            )
            self.register_buffer(
                "actions_mean",
                self._to_torch_float_tensor(
                    self.dataset_description.joint_target_positions.mean
                ),
            )
            self.register_buffer(
                "actions_std",
                self._to_torch_float_tensor(
                    self.dataset_description.joint_target_positions.std
                ),
            )

    def _to_torch_float_tensor(self, data: list[float]) -> torch.FloatTensor:
        """Convert list of floats to torch tensor on the correct device.

        Args:
            data: List of float values

        Returns:
            torch.FloatTensor: Tensor on the model's device
        """
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _preprocess_joint_state(
        self, joint_state: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Normalize joint state using dataset statistics.

        Args:
            joint_state: Raw joint state tensor

        Returns:
            torch.FloatTensor: Normalized joint state
        """
        if self.joint_state_mean is not None and self.joint_state_std is not None:
            return (joint_state - self.joint_state_mean) / self.joint_state_std
        return joint_state

    def _preprocess_actions(self, actions: torch.FloatTensor) -> torch.FloatTensor:
        """Normalize actions using dataset statistics.

        Args:
            actions: Raw action tensor

        Returns:
            torch.FloatTensor: Normalized actions
        """
        if self.actions_mean is not None and self.actions_std is not None:
            return (actions - self.actions_mean) / self.actions_std
        return actions

    def _predict_future_images(
        self, batch: BatchedInferenceSamples
    ) -> torch.FloatTensor:
        """Predict future images for the given batch.

        Processes current images, state, and actions through their respective
        encoders, then uses U-Net architectures to predict future image sequences
        for each camera view.

        Args:
            batch: Input batch with current observations and actions

        Returns:
            torch.FloatTensor: Predicted future images [B, T, cameras, 3, H, W]
        """
        batch_size = len(batch)
        num_cameras = self.dataset_description.rgb_images.max_len

        # Process current images from each camera
        image_features = []
        if batch.rgb_images is not None:
            for cam_id, encoder in enumerate(self.image_encoders):
                features = encoder(self.transform(batch.rgb_images.data[:, cam_id]))
                masked_features = (
                    features * batch.rgb_images.mask[:, cam_id : cam_id + 1]
                )
                image_features.append(masked_features)

        # Combine state inputs if available
        state_features = None
        if self.state_embed is not None:
            state_inputs = []
            if batch.joint_positions is not None:
                state_inputs.append(
                    batch.joint_positions.data * batch.joint_positions.mask
                )
            if batch.joint_velocities is not None:
                state_inputs.append(
                    batch.joint_velocities.data * batch.joint_velocities.mask
                )
            if batch.joint_torques is not None:
                state_inputs.append(batch.joint_torques.data * batch.joint_torques.mask)

            if state_inputs:
                joint_states = torch.cat(state_inputs, dim=-1)
                joint_states = self._preprocess_joint_state(joint_states)
                state_features = self.state_embed(joint_states)
            else:
                state_features = torch.zeros(
                    batch_size, self.hidden_dim, device=self.device, dtype=torch.float32
                )

        # Process action inputs if available
        action_features = None
        if self.action_embed is not None and batch.joint_target_positions is not None:
            action_data = (
                batch.joint_target_positions.data * batch.joint_target_positions.mask
            )
            action_data = self._preprocess_actions(action_data)
            action_features = self.action_embed(action_data)

        # Predict future images for each camera
        future_images = torch.zeros(
            batch_size,
            num_cameras,
            3 * self.model_init_description.output_prediction_horizon,
            224,
            224,
            device=self.device,
            dtype=torch.float32,
        )

        for cam_id in range(num_cameras):
            # Combine features for conditioning
            conditioning_features = []
            if state_features is not None:
                conditioning_features.append(state_features)
            if action_features is not None:
                conditioning_features.append(action_features)
            conditioning_features.append(image_features[cam_id])

            combined_features = torch.cat(conditioning_features, dim=-1)

            # Use UNet to predict future image
            if batch.rgb_images is not None:
                current_image = batch.rgb_images.data[:, cam_id]
                image_prediction = self.image_predictors[cam_id](
                    current_image, combined_features
                )

                # Apply mask if available
                future_image = image_prediction * batch.rgb_images.mask[:, cam_id].view(
                    batch_size, 1, 1, 1
                )

                future_images[:, cam_id] = future_image

        # [B, CAMS, 3 * T, H, W] -> [B, T, CAMS, 3, H, W]
        b, cams, c, h, w = future_images.shape
        t = self.model_init_description.output_prediction_horizon
        return future_images.view(b, cams, t, -1, h, w).transpose(1, 2)

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Perform inference to predict future image sequences.

        Args:
            batch: Input batch with current observations and actions

        Returns:
            ModelPrediction: Predicted future images with timing information
        """
        t = time.time()
        future_images = self._predict_future_images(batch)
        prediction_time = time.time() - t
        # [B, T, CAMS, 3, H, W] -> [B * T * CAMS, 3, H, W]
        b, t, cams, c, h, w = future_images.shape
        flattened_predictions = future_images.reshape(-1, c, h, w)
        flattened_predictions = self.inverse_transform(flattened_predictions)
        # [B * T * CAMS, 3, H, W] -> [B, T, CAMS, 3, H, W]
        future_images = flattened_predictions.view(b, t, cams, c, h, w)
        predictions_np = future_images.detach().cpu().numpy()
        predictions_np = np.transpose(predictions_np, (0, 1, 2, 4, 5, 3))
        cam_data = np.clip(predictions_np * 255.0, 0, 255).astype(np.uint8)
        return ModelPrediction(
            outputs={
                DataType.RGB_IMAGE: cam_data,
            },
            prediction_time=prediction_time,
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Predicts future images and computes reconstruction loss against target
        future images. Also computes PSNR metric for image quality evaluation.

        Args:
            batch: Training batch with inputs and target future images

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        # Create inference sample from training input
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            joint_target_positions=batch.inputs.joint_target_positions,
            rgb_images=batch.inputs.rgb_images,
        )

        # Preprocess images
        target_future_images = None
        if batch.outputs.rgb_images is not None:
            target_future_images = batch.outputs.rgb_images.data

        # Predict future images
        predicted_future_images = self._predict_future_images(inference_sample)

        losses = {}
        metrics = {}

        if (
            self.training
            and target_future_images is not None
            and batch.outputs.rgb_images is not None
        ):
            # [B, T, CAMS, 3, H, W] -> [B * T * CAMS, 3, H, W]
            _, _, _, c, h, w = target_future_images.shape
            target_future_image = self.transform(
                target_future_images.reshape(-1, c, h, w)
            )
            masked_target_future_image = (
                target_future_image
                * batch.outputs.rgb_images.mask.flatten().reshape(-1, 1, 1, 1)
            )
            reconstruction_loss = nn.functional.mse_loss(
                predicted_future_images.reshape(-1, c, h, w), masked_target_future_image
            )

            # Compute additional metrics like PSNR
            with torch.no_grad():
                mse = (
                    torch.mean(
                        predicted_future_images.reshape(-1, c, h, w)
                        - masked_target_future_image
                    )
                    ** 2
                )
                psnr = 10 * torch.log10(1 / mse) if mse > 0 else torch.tensor(100.0)
                metrics["psnr"] = psnr

            losses["reconstruction_loss"] = reconstruction_loss

        return BatchedTrainingOutputs(
            output_predictions=predicted_future_images,
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
            list[DataType]: List of supported input data types including
                joint states, actions, and RGB images
        """
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.RGB_IMAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Get the output data types supported by this model.

        Returns:
            list[DataType]: List of supported output data types
        """
        return [
            DataType.RGB_IMAGE,
        ]
