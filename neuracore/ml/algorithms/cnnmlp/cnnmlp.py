"""A simple CNN for each camera using a pretrained resnet18 followed by MLP."""

import torch
import torch.nn as nn
import torchvision.transforms as T

from neuracore.core.nc_types import DataType, ModelInitDescription
from neuracore.ml import (
    BatchedInferenceOutputs,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.ml_types import MaskableData

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

        self.state_embed = nn.Linear(
            self.dataset_description.joint_positions.max_len, hidden_dim
        )

        mlp_input_dim = (
            self.dataset_description.max_num_rgb_images * cnn_output_dim + hidden_dim
        )

        # Predict entire sequence at once
        self.action_output_size = (
            self.dataset_description.actions.max_len * self.action_prediction_horizon
        )
        self.mlp = self._build_mlp(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_output_size,
            num_layers=num_layers,
        )

        self.transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        self.action_prediction_horizon = self.action_prediction_horizon
        self.max_action_size = self.dataset_description.actions.max_len

        self.actions_mean = self._to_torch_float_tensor(
            self.dataset_description.actions.mean
        )
        self.actions_std = self._to_torch_float_tensor(
            self.dataset_description.actions.std
        )
        self.joint_positions_mean = self._to_torch_float_tensor(
            self.dataset_description.joint_positions.mean
        )
        self.joint_positions_std = self._to_torch_float_tensor(
            self.dataset_description.joint_positions.std
        )

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

    def _preprocess_joint_pos(self, joint_pos: torch.FloatTensor) -> torch.FloatTensor:
        """Preprocess the states."""
        return (joint_pos - self.joint_positions_mean) / self.joint_positions_std

    def _preprocess_actions(self, actions: torch.FloatTensor) -> torch.FloatTensor:
        """Preprocess the actions."""
        return (actions - self.actions_mean) / self.actions_std

    def _preprocess_camera_images(
        self, camera_images: torch.FloatTensor
    ) -> torch.FloatTensor:
        for cam_id in range(self.dataset_description.max_num_rgb_images):
            camera_images[:, cam_id] = self.transform(camera_images[:, cam_id])
        return camera_images

    def _inference_postprocess(
        self, output: BatchedInferenceOutputs
    ) -> BatchedInferenceOutputs:
        """Postprocess the output of the inference."""
        predictions = (output.action_predicitons * self.actions_std) + self.actions_mean
        return BatchedInferenceOutputs(action_predicitons=predictions)

    def _predict_action(self, batch: BatchedInferenceSamples) -> torch.FloatTensor:
        """Predict action for the given batch."""
        batch_size = batch.joint_positions.data.shape[0]

        # Process images from each camera
        image_features = []
        for cam_id, encoder in enumerate(self.image_encoders):
            features = encoder(batch.rgb_images.data[:, cam_id])
            features *= batch.rgb_images.mask[:, cam_id : cam_id + 1]
            image_features.append(features)

        # Combine image features
        if image_features:
            combined_image_features = torch.cat(image_features, dim=-1)
        else:
            combined_image_features = torch.zeros(
                batch_size, self.cnn_output_dim, device=self.device, dtype=torch.float32
            )

        state_features = self.state_embed(batch.joint_positions.data)

        # Combine all features
        combined_features = torch.cat([state_features, combined_image_features], dim=-1)

        # Forward through MLP to get entire sequence
        mlp_out = self.mlp(combined_features)

        action_preds = mlp_out.view(
            batch_size, self.action_prediction_horizon, self.max_action_size
        )
        return action_preds

    def forward(self, batch: BatchedInferenceSamples) -> BatchedInferenceOutputs:
        """Forward pass for inference."""
        preprocessed_batch = BatchedInferenceSamples(
            joint_positions=MaskableData(
                self._preprocess_joint_pos(batch.joint_positions.data),
                batch.joint_positions.mask,
            ),
            rgb_images=MaskableData(
                self._preprocess_camera_images(batch.rgb_images.data),
                batch.rgb_images.mask,
            ),
        )
        action_preds = self._predict_action(preprocessed_batch)
        return self._inference_postprocess(
            BatchedInferenceOutputs(action_predicitons=action_preds)
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step."""
        preprocessed_batch = BatchedInferenceSamples(
            joint_positions=MaskableData(
                self._preprocess_joint_pos(batch.joint_positions.data),
                batch.joint_positions.mask,
            ),
            rgb_images=MaskableData(
                self._preprocess_camera_images(batch.rgb_images.data),
                batch.rgb_images.mask,
            ),
        )
        target_actions = self._preprocess_actions(batch.actions.data)
        action_predicitons = self._predict_action(preprocessed_batch)
        losses = {}
        metrics = {}
        if self.training:
            loss = nn.functional.mse_loss(action_predicitons, target_actions)
            losses["mse_loss"] = loss

        return BatchedTrainingOutputs(
            action_predicitons=action_predicitons,
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
    def get_supported_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [
            DataType.JOINT_POSITIONS,
            DataType.RGB_IMAGE,
            DataType.ACTIONS,
        ]
