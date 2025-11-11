"""CNN+MLP model for robot manipulation with sequence prediction.

This module implements a simple baseline model that combines convolutional
neural networks for visual feature extraction with multi-layer perceptrons
for action sequence prediction. The model processes single timestep inputs
and outputs entire action sequences.
"""

import time
from typing import Any, Dict

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
from neuracore.ml.algorithm_utils.modules import (
    CustomDataEncoder,
    DepthImageEncoder,
    EndEffectorEncoder,
    MultimodalFusionEncoder,
    PointCloudEncoder,
    PoseEncoder,
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
        image_backbone: str = "resnet18",
        hidden_dim: int = 512,
        cnn_output_dim: int = 512,
        num_layers: int = 3,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        """Initialize the CNN+MLP model.

        Args:
            model_init_description: Model initialization parameters
            image_backbone: Backbone architecture for image encoders
            hidden_dim: Hidden dimension for MLP layers
            cnn_output_dim: Output dimension for CNN encoders
            num_layers: Number of MLP layers
            lr: Learning rate for main parameters
            lr_backbone: Learning rate for CNN backbone
            weight_decay: Weight decay for optimizer
        """
        super().__init__(model_init_description)
        self.image_backbone = image_backbone
        self.hidden_dim = hidden_dim
        self.cnn_output_dim = cnn_output_dim
        self.num_layers = num_layers
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        # Initialize encoders for each supported modality
        self.encoders = nn.ModuleDict()
        self.feature_dims = {}

        # Vision encoders
        if DataType.RGB_IMAGE in self.model_init_description.input_data_types:
            self.encoders["rgb"] = nn.ModuleList([
                ImageEncoder(output_dim=cnn_output_dim, backbone=image_backbone)
                for _ in range(self.dataset_description.rgb_images.max_len)
            ])
            self.feature_dims["rgb"] = (
                self.dataset_description.rgb_images.max_len * cnn_output_dim
            )

        if DataType.DEPTH_IMAGE in self.model_init_description.input_data_types:
            self.encoders["depth"] = nn.ModuleList([
                DepthImageEncoder(output_dim=cnn_output_dim)
                for _ in range(self.dataset_description.depth_images.max_len)
            ])
            self.feature_dims["depth"] = (
                self.dataset_description.depth_images.max_len * cnn_output_dim
            )

        if DataType.POINT_CLOUD in self.model_init_description.input_data_types:
            self.encoders["point_cloud"] = nn.ModuleList([
                PointCloudEncoder(output_dim=cnn_output_dim)
                for _ in range(self.dataset_description.point_clouds.max_len)
            ])
            self.feature_dims["point_cloud"] = (
                self.dataset_description.point_clouds.max_len * cnn_output_dim
            )

        # State encoders
        state_input_dim = 0
        if DataType.JOINT_POSITIONS in self.model_init_description.input_data_types:
            state_input_dim += self.dataset_description.joint_positions.max_len
        if DataType.JOINT_VELOCITIES in self.model_init_description.input_data_types:
            state_input_dim += self.dataset_description.joint_velocities.max_len
        if DataType.JOINT_TORQUES in self.model_init_description.input_data_types:
            state_input_dim += self.dataset_description.joint_torques.max_len

        if state_input_dim > 0:
            self.encoders["joints"] = nn.Linear(state_input_dim, cnn_output_dim)
            self.feature_dims["joints"] = cnn_output_dim

        # End-effector encoder
        if DataType.END_EFFECTORS in self.model_init_description.input_data_types:
            self.encoders["end_effectors"] = EndEffectorEncoder(
                output_dim=cnn_output_dim,
                max_effectors=self.dataset_description.end_effector_states.max_len,
            )
            self.feature_dims["end_effectors"] = cnn_output_dim

        # Pose encoder
        if DataType.POSES in self.model_init_description.input_data_types:
            self.encoders["poses"] = PoseEncoder(
                output_dim=cnn_output_dim,
                max_poses=self.dataset_description.poses.max_len // 6,  # 6DOF per pose
            )
            self.feature_dims["poses"] = cnn_output_dim

        # Language encoder (simplified - just use embedding)
        if DataType.LANGUAGE in self.model_init_description.input_data_types:
            self.encoders["language"] = nn.Sequential(
                nn.Embedding(1000, 128),  # Simple embedding
                nn.Linear(128, cnn_output_dim),
            )
            self.feature_dims["language"] = cnn_output_dim

        # Custom data encoders
        self.custom_encoders = nn.ModuleDict()
        for key, data_items_stats in self.dataset_description.custom_data.items():
            self.custom_encoders[key] = CustomDataEncoder(
                input_dim=data_items_stats.max_len, output_dim=cnn_output_dim
            )
            self.feature_dims[key] = cnn_output_dim

        # Use multimodal fusion if multiple modalities
        self.fusion = MultimodalFusionEncoder(
            feature_dims=self.feature_dims, output_dim=hidden_dim
        )
        mlp_input_dim = hidden_dim

        # Determine output configuration
        self.action_data_type = self.model_init_description.output_data_types[0]
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

        # Image transformations
        self.rgb_transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        self.depth_transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.5], std=[0.5]),  # Simple normalization for depth
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

        # Action normalization
        action_data_item_stats = (
            self.dataset_description.joint_target_positions
            if self.action_data_type == DataType.JOINT_TARGET_POSITIONS
            else self.dataset_description.joint_positions
        )
        self.register_buffer(
            "action_mean", self._to_torch_float_tensor(action_data_item_stats.mean)
        )
        self.register_buffer(
            "action_std", self._to_torch_float_tensor(action_data_item_stats.std)
        )

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
        return (actions - self.action_mean) / self.action_std

    def _process_visual_features(
        self, batch: BatchedInferenceSamples
    ) -> Dict[str, torch.Tensor]:
        """Process all visual modalities and return features."""
        features = {}

        # RGB images
        if "rgb" in self.encoders and batch.rgb_images is not None:
            image_features = []
            for cam_id, encoder in enumerate(self.encoders["rgb"]):
                features_cam = encoder(
                    self.rgb_transform(batch.rgb_images.data[:, cam_id])
                )
                # Apply mask
                features_cam *= batch.rgb_images.mask[:, cam_id : cam_id + 1]
                image_features.append(features_cam)
            features["rgb"] = torch.cat(image_features, dim=-1)

        # Depth images
        if "depth" in self.encoders and batch.depth_images is not None:
            depth_features = []
            for cam_id, encoder in enumerate(self.encoders["depth"]):
                features_cam = encoder(
                    self.depth_transform(batch.depth_images.data[:, cam_id])
                )
                # Apply mask
                features_cam *= batch.depth_images.mask[:, cam_id : cam_id + 1]
                depth_features.append(features_cam)
            features["depth"] = torch.cat(depth_features, dim=-1)

        # Point clouds
        if "point_cloud" in self.encoders and batch.point_clouds is not None:
            pc_features = []
            for pc_id, encoder in enumerate(self.encoders["point_cloud"]):
                features_pc = encoder(batch.point_clouds.data[:, pc_id])
                # Apply mask
                features_pc *= batch.point_clouds.mask[:, pc_id : pc_id + 1]
                pc_features.append(features_pc)
            features["point_cloud"] = torch.cat(pc_features, dim=-1)

        return features

    def _process_state_features(
        self, batch: BatchedInferenceSamples
    ) -> Dict[str, torch.Tensor]:
        """Process all state modalities and return features."""
        features = {}

        # Joint states
        if "joints" in self.encoders:
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
                features["joints"] = self.encoders["joints"](joint_states)

        # End-effectors
        if "end_effectors" in self.encoders and batch.end_effectors is not None:
            ee_data = batch.end_effectors.data * batch.end_effectors.mask
            features["end_effectors"] = self.encoders["end_effectors"](ee_data)

        # Poses
        if "poses" in self.encoders and batch.poses is not None:
            pose_data = batch.poses.data * batch.poses.mask
            features["poses"] = self.encoders["poses"](pose_data)

        return features

    def _process_language_features(
        self, batch: BatchedInferenceSamples
    ) -> Dict[str, torch.Tensor]:
        """Process language features."""
        features = {}

        if "language" in self.encoders and batch.language_tokens is not None:
            # Simple approach: use mean of token embeddings
            token_embeddings = self.encoders["language"][0](
                batch.language_tokens.data.long()
            )
            # Apply attention mask and take mean
            masked_embeddings = token_embeddings * batch.language_tokens.mask.unsqueeze(
                -1
            )
            mean_embeddings = masked_embeddings.sum(
                dim=1
            ) / batch.language_tokens.mask.sum(dim=1, keepdim=True)
            features["language"] = self.encoders["language"][1](mean_embeddings)

        return features

    def _process_custom_features(
        self, batch: BatchedInferenceSamples
    ) -> Dict[str, torch.Tensor]:
        """Process custom data features."""
        features = {}

        if batch.custom_data is not None:
            for key, custom_data in batch.custom_data.items():
                if key in self.custom_encoders:
                    custom_input = custom_data.data * custom_data.mask
                    features[key] = self.custom_encoders[key](custom_input)

        return features

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
        all_features = {}

        # Process each modality
        all_features.update(self._process_visual_features(batch))
        all_features.update(self._process_state_features(batch))
        all_features.update(self._process_language_features(batch))
        all_features.update(self._process_custom_features(batch))

        combined_features = self.fusion(all_features)

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
            end_effectors=batch.inputs.end_effectors,
            poses=batch.inputs.poses,
            rgb_images=batch.inputs.rgb_images,
            depth_images=batch.inputs.depth_images,
            point_clouds=batch.inputs.point_clouds,
            language_tokens=batch.inputs.language_tokens,
            custom_data=batch.inputs.custom_data,
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
        action_predictions = self._predict_action(inference_sample)

        losses: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        if self.training:
            losses["l1_loss"] = nn.functional.l1_loss(
                action_predictions, target_actions
            )

        return BatchedTrainingOutputs(
            output_predictions=action_predictions,
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
            if any(backbone in name for backbone in ["rgb", "depth"]):
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
            DataType.END_EFFECTORS,
            DataType.POSES,
            DataType.RGB_IMAGE,
            DataType.DEPTH_IMAGE,
            DataType.POINT_CLOUD,
            DataType.LANGUAGE,
            DataType.CUSTOM,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Get the output data types supported by this model.

        Returns:
            list[DataType]: List of supported output data types
        """
        return [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]

    @staticmethod
    def tokenize_text(text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text using simple word-level tokenization.

        Args:
            text: List of text strings to tokenize

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input IDs and attention masks
        """
        # Simple tokenization - convert to word indices
        max_length = 50
        vocab_size = 1000

        batch_size = len(text)
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.float)

        for i, txt in enumerate(text):
            words = txt.lower().split()[:max_length]
            for j, word in enumerate(words):
                # Simple hash-based vocab mapping
                input_ids[i, j] = hash(word) % vocab_size
                attention_mask[i, j] = 1.0

        return input_ids, attention_mask
