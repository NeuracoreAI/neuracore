"""A vision-language-action model combining image, language and state inputs."""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoTokenizer

from neuracore.core.nc_types import DataType, ModelInitDescription, ModelPrediction
from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import ImageEncoder, LanguageEncoder, MultimodalFusion

LANGUAGE_MODEL_NAME = "distilbert-base-uncased"
_tokenizer = None


class SimpleVLA(NeuracoreModel):
    """
    Simple Vision-Language-Action model.

    This model combines visual features from images, language features from text
    instructions, and robot state information to predict actions.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 512,
        cnn_output_dim: int = 64,
        language_output_dim: int = 64,
        fusion_output_dim: int = 128,
        num_layers: int = 3,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        super().__init__(model_init_description)
        self.hidden_dim = hidden_dim
        self.cnn_output_dim = cnn_output_dim
        self.language_output_dim = language_output_dim
        self.fusion_output_dim = fusion_output_dim
        self.num_layers = num_layers
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # Vision encoders - one for each camera
        self.image_encoders = nn.ModuleList([
            ImageEncoder(output_dim=self.cnn_output_dim)
            for _ in range(self.dataset_description.max_num_rgb_images)
        ])

        # Language encoder
        self.language_encoder = LanguageEncoder(
            output_dim=self.language_output_dim, model_name=LANGUAGE_MODEL_NAME
        )

        # State processing
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

        # Multimodal fusion module
        vision_dim = self.dataset_description.max_num_rgb_images * cnn_output_dim
        self.fusion = MultimodalFusion(
            vision_dim=vision_dim,
            language_dim=self.language_output_dim,
            state_dim=hidden_state_dim,
            hidden_dim=hidden_dim,
            output_dim=fusion_output_dim,
        )

        # Predict entire sequence at once
        self.output_size = (
            self.dataset_description.joint_target_positions.max_len
            * self.output_prediction_horizon
        )

        # Action predictor MLP
        self.action_predictor = self._build_mlp(
            input_dim=fusion_output_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_size,
            num_layers=num_layers,
        )

        # Image transformation
        self.transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        self.max_output_size = self.dataset_description.joint_target_positions.max_len

        # Normalization statistics
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

        self.joint_target_mean = self._to_torch_float_tensor(
            self.dataset_description.joint_target_positions.mean
        )
        self.joint_target_std = self._to_torch_float_tensor(
            self.dataset_description.joint_target_positions.std
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
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
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

    def _preprocess_target_joint_pos(
        self, target_joint_pos: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Preprocess the actions."""
        return (target_joint_pos - self.joint_target_mean) / self.joint_target_std

    def _process_language_tokens(
        self,
        language_tokens: Optional[torch.FloatTensor],
        language_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """Process language tokens through the language encoder."""
        if language_tokens is None:
            # Return zero tensor with appropriate dimensions if no language input
            return torch.zeros(
                language_tokens.shape[0], self.language_output_dim, device=self.device
            )

        # Forward through language encoder
        return self.language_encoder(
            input_ids=language_tokens,
            attention_mask=(
                language_mask
                if language_mask is not None
                else torch.ones_like(language_tokens)
            ),
        )

    def _predict_action(self, batch: BatchedInferenceSamples) -> torch.FloatTensor:
        """Predict action for the given batch."""
        batch_size = len(batch)

        # Process images from each camera
        image_features = []
        if batch.rgb_images is not None:
            for cam_id, encoder in enumerate(self.image_encoders):
                features = encoder(self.transform(batch.rgb_images.data[:, cam_id]))
                # Apply mask if available
                if batch.rgb_images.mask is not None:
                    features *= batch.rgb_images.mask[:, cam_id : cam_id + 1]
                image_features.append(features)

        # Combine image features
        if image_features:
            combined_image_features = torch.cat(image_features, dim=-1)
        else:
            combined_image_features = torch.zeros(
                batch_size,
                self.dataset_description.max_num_rgb_images * self.cnn_output_dim,
                device=self.device,
            )

        # Process language tokens
        language_features = None
        if batch.language_tokens is not None:
            language_features = self._process_language_tokens(
                batch.language_tokens.data, batch.language_tokens.mask
            )
        else:
            language_features = torch.zeros(
                batch_size, self.language_output_dim, device=self.device
            )

        # Process state inputs if available
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
                    batch_size, self.hidden_dim, device=self.device
                )

        # Fuse all features
        fused_features = self.fusion(
            combined_image_features, language_features, state_features
        )

        # Forward through action predictor to get entire sequence
        mlp_out = self.action_predictor(fused_features)

        # Reshape output to (batch, horizon, action_size)
        action_preds = mlp_out.view(
            batch_size, self.output_prediction_horizon, self.max_output_size
        )
        return action_preds

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Forward pass for inference."""
        t = time.time()
        action_preds = self._predict_action(batch)
        prediction_time = time.time() - t

        # Unnormalize predictions
        predictions = (action_preds * self.joint_target_std) + self.joint_target_mean
        predictions = predictions.detach().cpu().numpy()

        return ModelPrediction(
            outputs={DataType.JOINT_TARGET_POSITIONS: predictions},
            prediction_time=prediction_time,
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step."""
        # Convert training batch to inference batch
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
            language_tokens=batch.inputs.language_tokens,
        )

        # Preprocess target actions
        target_actions = self._preprocess_target_joint_pos(
            batch.outputs.joint_target_positions.data
        )

        # Get model predictions
        action_predictions = self._predict_action(inference_sample)

        losses, metrics = {}, {}
        if self.training:
            # Calculate MSE loss
            losses["mse_loss"] = nn.functional.mse_loss(
                action_predictions, target_actions
            )

        return BatchedTrainingOutputs(
            output_predicitons=action_predictions,
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure and return optimizer for the model."""
        # Separate parameters for backbones and other layers
        backbone_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "image_encoders" in name or "language_encoder" in name:
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
            DataType.LANGUAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [DataType.JOINT_TARGET_POSITIONS]

    @staticmethod
    def tokenize_text(text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text."""
        global _tokenizer
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)

        # Tokenize the text
        tokens = _tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract token ids and attention mask
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return input_ids, attention_mask
