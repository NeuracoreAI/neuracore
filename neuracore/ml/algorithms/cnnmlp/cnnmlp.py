"""CNN+MLP model for robot manipulation with sequence prediction.

This module implements a simple baseline model that combines convolutional
neural networks for visual feature extraction with multi-layer perceptrons
for action sequence prediction. The model processes single timestep inputs
and outputs entire action sequences.
"""

import time
from typing import Any, Dict, cast

import torch
import torch.nn as nn
import torchvision.transforms as T
from neuracore_types import DataType, ModelInitDescription, ModelPrediction

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.modules import (
    ENCODER_TYPE_TO_DATA_TYPE,
    PROPRIOCEPTIVE_DATA_TYPES,
    DepthImageEncoder,
    EncoderType,
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

    # Registered buffer type hints
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor

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
        self.encoders: Dict[EncoderType, nn.Module] = cast(
            Dict[EncoderType, nn.Module], nn.ModuleDict()
        )
        self.encoder_output_dims: Dict[EncoderType, int] = {}

        # Process proprioceptive inputs
        if PROPRIOCEPTIVE_DATA_TYPES.intersection(
            self.model_init_description.input_data_types
        ):
            input_size = 0
            proprio_means, proprio_stds = [], []
            for dt in PROPRIOCEPTIVE_DATA_TYPES:
                if dt in self.model_init_description.input_data_types:
                    input_size += sum(
                        item_stats.max_len
                        for item_stats in self.dataset_statistics.data[dt].values()
                    )
                    stats = self.dataset_statistics.combine_for_data_type(dt)
                    proprio_means.extend(stats.mean)
                    proprio_stds.extend(stats.std)
            self.register_buffer(
                "proprio_mean", torch.tensor(proprio_means, dtype=torch.float32)
            )
            self.register_buffer(
                "proprio_std", torch.tensor(proprio_stds, dtype=torch.float32)
            )

            # Use same cnn_output_dim for proprioceptive encoders
            self.encoder_output_dims[EncoderType.PROPRIOCEPTIVE] = cnn_output_dim
            self.encoders[EncoderType.PROPRIOCEPTIVE] = nn.Linear(
                input_size, cnn_output_dim
            )

        if DataType.RGB_IMAGES in self.model_init_description.input_data_types:
            camera_names = list(
                self.dataset_statistics.data[DataType.RGB_IMAGES].keys()
            )
            self.encoder_output_dims[EncoderType.RGB_IMAGES] = (
                len(camera_names) * cnn_output_dim
            )
            self.encoders[EncoderType.RGB_IMAGES] = nn.ModuleDict({
                name: ImageEncoder(output_dim=cnn_output_dim, backbone=image_backbone)
                for name in camera_names
            })
            self.rgb_transform = torch.nn.Sequential(
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )

        if DataType.DEPTH_IMAGES in self.model_init_description.input_data_types:
            camera_names = list(
                self.dataset_statistics.data[DataType.DEPTH_IMAGES].keys()
            )
            self.encoder_output_dims[EncoderType.DEPTH_IMAGES] = (
                len(camera_names) * cnn_output_dim
            )
            self.encoders[EncoderType.DEPTH_IMAGES] = nn.ModuleDict({
                name: DepthImageEncoder(output_dim=cnn_output_dim)
                for name in camera_names
            })
            self.depth_transform = torch.nn.Sequential(
                T.Resize((224, 224)),
                T.Normalize(mean=[0.5], std=[0.5]),  # Simple normalization for depth
            )

        if DataType.POINT_CLOUDS in self.model_init_description.input_data_types:
            camera_names = list(
                self.dataset_statistics.data[DataType.POINT_CLOUDS].keys()
            )
            self.encoder_output_dims[EncoderType.POINT_CLOUDS] = (
                len(camera_names) * cnn_output_dim
            )
            self.encoders[EncoderType.POINT_CLOUDS] = nn.ModuleDict({
                name: PointCloudEncoder(output_dim=cnn_output_dim)
                for name in camera_names
            })

        # All poses will share the same encoder
        if DataType.POSES in self.model_init_description.input_data_types:
            num_poses = len(self.dataset_statistics.data[DataType.POSES])
            self.encoder_output_dims[EncoderType.POSES] = cnn_output_dim
            self.encoders[EncoderType.POSES] = PoseEncoder(
                output_dim=cnn_output_dim,
                max_poses=num_poses,
            )

        # Language encoder (simplified - just use embedding)
        if DataType.LANGUAGE in self.model_init_description.input_data_types:
            self.encoder_output_dims[EncoderType.LANGUAGE] = cnn_output_dim
            self.encoders[EncoderType.LANGUAGE] = nn.Sequential(
                nn.Embedding(1000, 128),  # Simple embedding
                nn.Linear(128, cnn_output_dim),
            )

        # Use multimodal fusion if multiple modalities
        self.fusion = MultimodalFusionEncoder(
            feature_dims=self.encoder_output_dims, output_dim=hidden_dim
        )
        mlp_input_dim = hidden_dim

        # Determine output configuration
        self.action_data_type = self.model_init_description.output_data_types[0]
        if DataType.JOINT_TARGET_POSITIONS == self.action_data_type:
            action_stats = self.dataset_statistics.combine_for_data_type(
                DataType.JOINT_TARGET_POSITIONS
            )
        else:
            action_stats = self.dataset_statistics.combine_for_data_type(
                DataType.JOINT_POSITIONS
            )

        self.register_buffer(
            "action_mean", torch.tensor(action_stats.mean, dtype=torch.float32)
        )
        self.register_buffer(
            "action_std", torch.tensor(action_stats.std, dtype=torch.float32)
        )

        self.max_output_size = action_stats.max_len
        # Predict entire sequence at once
        self.output_size = self.max_output_size * self.output_prediction_horizon
        self.mlp = self._build_mlp(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_size,
            num_layers=num_layers,
        )

        assert (self.action_mean is not None) and (
            self.action_std is not None
        ), "Action statistics must be set."

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

    def _encode_inputs(
        self, batch: BatchedInferenceInputs
    ) -> Dict[EncoderType, torch.Tensor]:
        """Process all visual modalities and return features."""
        features: Dict[EncoderType, torch.Tensor] = {}

        # Visual data
        for encoder_type, transform in zip(
            [
                EncoderType.RGB_IMAGES,
                EncoderType.DEPTH_IMAGES,
                EncoderType.POINT_CLOUDS,
            ],
            [self.rgb_transform, self.depth_transform, None],
        ):
            if encoder_type not in self.encoders:
                continue
            feats = []
            for camera_name, encoder in self.encoders[encoder_type].items():
                data_type = ENCODER_TYPE_TO_DATA_TYPE[encoder_type]
                camera_data = batch.inputs[data_type][camera_name]
                feats.append(
                    encoder(
                        transform(camera_data.data) if transform else camera_data.data
                    )
                    * camera_data.mask
                )
            features[encoder_type] = torch.cat(feats, dim=-1)

        # Proprioceptive data
        if EncoderType.PROPRIOCEPTIVE in self.encoders:
            proprio_inputs = []
            for proprio_type in PROPRIOCEPTIVE_DATA_TYPES:
                if proprio_type not in batch.inputs:
                    continue
                maskable_data = batch.combine_for_data_type(proprio_type)
                proprio_inputs.append(maskable_data.data * maskable_data.mask)
            proprio_inputs = torch.cat(proprio_inputs, dim=-1)
            proprio_inputs = (proprio_inputs - self.proprio_mean) / self.proprio_std
            features[EncoderType.PROPRIOCEPTIVE] = self.encoders[
                EncoderType.PROPRIOCEPTIVE
            ](proprio_inputs)

        # Pose Data
        if EncoderType.POSES in self.encoders:
            maskable_data = batch.combine_for_data_type(DataType.POSES)
            # TODO: previously required [B, num_poses, pose_dim]
            #   but I believe now handles [B, pose_dim] only
            features[EncoderType.POSES] = self.encoders[EncoderType.POSES](
                maskable_data.data * maskable_data.mask
            )

        # Language data
        if EncoderType.LANGUAGE in self.encoders and DataType.LANGUAGE in batch.inputs:
            # Assume there's only one language input (or use the first one)
            language_data = next(iter(batch.inputs[DataType.LANGUAGE].values()))

            # Simple approach: use mean of token embeddings
            token_embeddings = self.encoders[EncoderType.LANGUAGE][0](
                language_data.data.long()
            )
            # Apply attention mask and take mean
            masked_embeddings = token_embeddings * language_data.mask.unsqueeze(-1)
            mean_embeddings = masked_embeddings.sum(dim=1) / language_data.mask.sum(
                dim=1, keepdim=True
            )
            features[EncoderType.LANGUAGE] = self.encoders[EncoderType.LANGUAGE][1](
                mean_embeddings
            )

        return features

    def _predict_action(self, batch: BatchedInferenceInputs) -> torch.FloatTensor:
        """Predict action sequence for the given batch.

        Processes visual and proprioceptive inputs through separate encoders,
        combines features, and predicts the entire action sequence through MLP.

        Args:
            batch: Input batch with observations

        Returns:
            torch.FloatTensor: Predicted action sequence [B, T, action_dim]
        """
        batch_size = len(batch)
        combined_features = self.fusion(self._encode_inputs(batch))

        # Forward through MLP to get entire sequence
        mlp_out = self.mlp(combined_features)
        action_preds = mlp_out.view(
            batch_size, self.output_prediction_horizon, self.max_output_size
        )
        return action_preds

    def forward(self, batch: BatchedInferenceInputs) -> ModelPrediction:
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
        inference_sample = BatchedInferenceInputs(inputs=batch.inputs)

        # Get target actions based on action data type
        if self.action_data_type == DataType.JOINT_TARGET_POSITIONS:
            assert (
                DataType.JOINT_TARGET_POSITIONS in batch.outputs
            ), "joint_target_positions required"
            # Concatenate all joint target positions
            action_data_list = [
                data.data
                for data in batch.outputs[DataType.JOINT_TARGET_POSITIONS].values()
            ]
            action_data = (
                torch.cat(action_data_list, dim=-1)
                if len(action_data_list) > 1
                else action_data_list[0]
            )
        else:
            assert DataType.JOINT_POSITIONS in batch.outputs, "joint_positions required"
            # Concatenate all joint positions
            action_data_list = [
                data.data for data in batch.outputs[DataType.JOINT_POSITIONS].values()
            ]
            action_data = (
                torch.cat(action_data_list, dim=-1)
                if len(action_data_list) > 1
                else action_data_list[0]
            )

        target_actions = (action_data - self.action_mean) / self.action_std
        action_predictions = self._predict_action(inference_sample)

        losses: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}

        if self.training:
            losses["l1_loss"] = nn.functional.l1_loss(
                action_predictions, target_actions
            )

        return BatchedTrainingOutputs(
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
            if any(
                encoder_type in name
                for encoder_type in [EncoderType.RGB_IMAGES, EncoderType.DEPTH_IMAGES]
            ):
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
            DataType.POSES,
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
            DataType.POINT_CLOUDS,
            DataType.LANGUAGE,
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
