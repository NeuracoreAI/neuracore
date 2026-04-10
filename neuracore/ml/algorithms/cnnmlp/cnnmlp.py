"""CNN+MLP model for robot manipulation with sequence prediction.

This module implements a simple baseline model that combines convolutional
neural networks for visual feature extraction with multi-layer perceptrons
for action sequence prediction. The model processes single timestep inputs
and outputs entire action sequences.
"""

import os
from typing import Any, cast

import torch
import torch.nn as nn
import torchvision.transforms as T
from neuracore_types import (
    BatchedDepthData,
    BatchedJointData,
    BatchedLanguageData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    BatchedPointCloudData,
    BatchedPoseData,
    BatchedRGBData,
    CameraDataStats,
    DataItemStats,
    DataType,
    JointDataStats,
    ModelInitDescription,
    ParallelGripperOpenAmountDataStats,
    PointCloudDataStats,
)

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.modules import (
    DepthImageEncoder,
    MultimodalFusionEncoder,
    PointCloudEncoder,
    PoseEncoder,
)
from neuracore.ml.algorithm_utils.normalizer import MeanStdNormalizer

from .modules import ImageEncoder

PROPRIO_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer
ACTION_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

LANGUAGE_MODEL_NAME = os.getenv("LANGUAGE_MODEL_NAME", "distilbert-base-uncased")
VISION_BACKBONE_DATA_TYPES = [
    DataType.RGB_IMAGES,
    DataType.DEPTH_IMAGES,
    DataType.POINT_CLOUDS,
]
PROPRIO_DATA_TYPES = [
    DataType.JOINT_POSITIONS,
    DataType.JOINT_VELOCITIES,
    DataType.JOINT_TORQUES,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
]


class CNNMLP(NeuracoreModel):
    """CNN+MLP model with single timestep input and sequence output.

    A baseline model architecture that uses separate CNN encoders for each
    camera view, combines visual features with proprioceptive state, and
    predicts entire action sequences through a multi-layer perceptron.

    The model processes current observations and outputs a fixed-length
    sequence of future actions, making it suitable for action chunking
    approaches in robot manipulation.
    """

    @staticmethod
    def _combine_joint_stats(stats: list[JointDataStats]) -> DataItemStats:
        combined_stats = DataItemStats()
        for stat in stats:
            combined_stats = combined_stats.concatenate(stat.value)
        return combined_stats

    @staticmethod
    def _combine_gripper_stats(
        stats: list[ParallelGripperOpenAmountDataStats],
    ) -> DataItemStats:
        combined_stats = DataItemStats()
        for stat in stats:
            combined_stats = combined_stats.concatenate(stat.open_amount)
        return combined_stats

    def _get_combined_input_stats(self, data_type: DataType) -> DataItemStats:
        if data_type not in self.input_dataset_statistics:
            raise ValueError(
                f"Missing input dataset statistics for data type: {data_type}"
            )
        raw_stats = self.input_dataset_statistics[data_type]

        if data_type in {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
        }:
            return self._combine_joint_stats(cast(list[JointDataStats], raw_stats))
        if data_type in {
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        }:
            return self._combine_gripper_stats(
                cast(list[ParallelGripperOpenAmountDataStats], raw_stats)
            )
        raise ValueError(
            f"Unsupported input stats data type for combination: {data_type}"
        )

    def _get_combined_output_stats(self, data_type: DataType) -> DataItemStats:
        if data_type not in self.output_dataset_statistics:
            raise ValueError(
                f"Missing output dataset statistics for data type: {data_type}"
            )
        raw_stats = self.output_dataset_statistics[data_type]

        if data_type in {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
        }:
            return self._combine_joint_stats(cast(list[JointDataStats], raw_stats))
        if data_type in {
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        }:
            return self._combine_gripper_stats(
                cast(list[ParallelGripperOpenAmountDataStats], raw_stats)
            )
        raise ValueError(
            f"Unsupported output stats data type for combination: {data_type}"
        )

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        image_backbone: str = "resnet18",
        use_resnet_stats: bool = True,
        hidden_dim: int = 512,
        cnn_output_dim: int = 512,
        num_layers: int = 3,
        lr: float = 1e-4,
        freeze_backbone: bool = False,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        """Initialize the CNN+MLP model.

        Args:
            model_init_description: Model initialization parameters
            image_backbone: Backbone architecture for image encoders
            use_resnet_stats: Whether to use ResNet normalization statistics
            hidden_dim: Hidden dimension for MLP layers
            cnn_output_dim: Output dimension for CNN encoders
            num_layers: Number of MLP layers
            lr: Learning rate for main parameters
            freeze_backbone: Whether to freeze image encoder backbone
            lr_backbone: Learning rate for CNN backbone
            weight_decay: Weight decay for optimizer
        """
        super().__init__(model_init_description)
        self.image_backbone = image_backbone
        self.use_resnet_stats = use_resnet_stats
        self.hidden_dim = hidden_dim
        self.cnn_output_dim = cnn_output_dim
        self.num_layers = num_layers
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.encoders = nn.ModuleDict()
        self.encoder_output_dims: dict[DataType, int] = {}
        self.input_combined_stats: dict[DataType, DataItemStats] = {}
        self.output_combined_stats: dict[DataType, DataItemStats] = {}
        self.output_layout: list[tuple[DataType, int]] = []

        # Register proprio inputs: combine their stats for normalization and add a
        # linear encoder so each proprio modality maps into the fusion feature space.
        for data_type in PROPRIO_DATA_TYPES:
            if data_type not in self.input_data_types:
                continue
            combined_stats = self._get_combined_input_stats(data_type)
            self.input_combined_stats[data_type] = combined_stats
            stats = self.input_dataset_statistics[data_type]
            self.encoder_output_dims[data_type] = cnn_output_dim
            self.encoders[data_type] = nn.Linear(len(stats), cnn_output_dim)

        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats],
                self.input_dataset_statistics[DataType.RGB_IMAGES],
            )
            max_cameras = len(stats)
            self.encoder_output_dims[DataType.RGB_IMAGES] = max_cameras * cnn_output_dim
            self.encoders[DataType.RGB_IMAGES] = nn.ModuleList()
            for i in range(max_cameras):
                if use_resnet_stats:
                    mean, std = RESNET_MEAN, RESNET_STD
                else:
                    # Will be (3, H, W)
                    mean_c_h_w, std_c_h_w = stats[i].frame.mean, stats[i].frame.std
                    mean = mean_c_h_w.mean(axis=(1, 2)).tolist()
                    std = std_c_h_w.mean(axis=(1, 2)).tolist()
                encoder = torch.nn.Sequential(
                    T.Resize((224, 224)),
                    T.Normalize(mean=mean, std=std),
                    ImageEncoder(output_dim=cnn_output_dim, backbone=image_backbone),
                )
                self.encoders[DataType.RGB_IMAGES].append(encoder)

        if DataType.DEPTH_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats],
                self.input_dataset_statistics[DataType.DEPTH_IMAGES],
            )
            max_cameras = len(stats)
            self.encoder_output_dims[DataType.DEPTH_IMAGES] = (
                max_cameras * cnn_output_dim
            )
            self.encoders[DataType.DEPTH_IMAGES] = nn.ModuleList()
            for i in range(max_cameras):
                encoder = torch.nn.Sequential(
                    T.Resize((224, 224)),
                    DepthImageEncoder(output_dim=cnn_output_dim),
                )
                self.encoders[DataType.DEPTH_IMAGES].append(encoder)

        if DataType.POINT_CLOUDS in self.input_data_types:
            stats = cast(
                list[PointCloudDataStats],
                self.input_dataset_statistics[DataType.POINT_CLOUDS],
            )
            max_pcs = len(stats)
            self.encoder_output_dims[DataType.POINT_CLOUDS] = max_pcs * cnn_output_dim
            self.encoders[DataType.POINT_CLOUDS] = nn.ModuleList()
            for i in range(max_pcs):
                encoder = PointCloudEncoder(output_dim=cnn_output_dim)
                self.encoders[DataType.POINT_CLOUDS].append(encoder)

        # All poses will share the same encoder
        if DataType.POSES in self.input_data_types:
            stats = self.input_dataset_statistics[DataType.POSES]
            stats = cast(list[DataItemStats], stats)
            max_poses = len(stats)
            self.encoder_output_dims[DataType.POSES] = cnn_output_dim
            self.encoders[DataType.POSES] = PoseEncoder(
                output_dim=cnn_output_dim,
                max_poses=max_poses,
            )

        # Language encoder (simplified - just use embedding)
        if DataType.LANGUAGE in self.input_data_types:
            from transformers import AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)
            self.embedding_encoder = nn.Embedding(_tokenizer.vocab_size, 128)
            self.encoder_output_dims[DataType.LANGUAGE] = cnn_output_dim
            self.encoders[DataType.LANGUAGE] = nn.Sequential(
                nn.Linear(128, cnn_output_dim),
            )

        # Use multimodal fusion if multiple modalities
        self.fusion = MultimodalFusionEncoder(
            feature_dims=self.encoder_output_dims, output_dim=hidden_dim
        )
        mlp_input_dim = hidden_dim

        output_stats = []
        self.output_dims: dict[DataType, tuple[int, int]] = {}
        current_output_dim = 0
        for data_type in self.ordered_output_data_types:
            combined_stats = self._get_combined_output_stats(data_type)
            self.output_combined_stats[data_type] = combined_stats
            output_stats.append(combined_stats)
            output_width = len(combined_stats.mean)
            self.output_layout.append((data_type, output_width))
            self.output_dims[data_type] = (
                current_output_dim,
                current_output_dim + output_width,
            )
            current_output_dim += output_width
        self.max_output_size = current_output_dim

        input_stats = []
        self.proprio_dims = {}
        current_dim = 0
        for data_type in PROPRIO_DATA_TYPES:
            if data_type not in self.input_data_types:
                continue
            if data_type not in self.input_combined_stats:
                continue
            combined_stats = self.input_combined_stats[data_type]
            input_stats.append(combined_stats)
            dim = len(combined_stats.mean)
            self.proprio_dims[data_type] = (current_dim, current_dim + dim)
            current_dim += dim

        self.action_normalizer = ACTION_NORMALIZER(
            name="actions", statistics=output_stats
        )
        self.proprio_normalizer = (
            PROPRIO_NORMALIZER(name="proprioception", statistics=input_stats)
            if input_stats
            else None
        )

        # Predict entire sequence at once
        self.output_size = self.max_output_size * self.output_prediction_horizon
        self.mlp = self._build_mlp(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_size,
            num_layers=num_layers,
        )

        # Setup parameter groups
        self._setup_optimizer_param_groups()

    def _setup_optimizer_param_groups(self) -> None:
        """Setup parameter groups for optimizer."""
        backbone_params = []
        backbone_param_names = []
        for data_type in VISION_BACKBONE_DATA_TYPES:
            if data_type in self.encoders:
                for name, param in self.encoders[data_type].named_parameters():
                    backbone_params.append(param)
                    # Example full name: "encoders.RGB_IMAGES.0.2.backbone.0.weight"
                    backbone_param_names.append(f"encoders.{data_type.value}.{name}")
        other_params = [
            param
            for name, param in self.named_parameters()
            if name not in backbone_param_names
        ]

        if self.freeze_backbone:
            for param in backbone_params:
                param.requires_grad = False
            self.param_groups = [{"params": other_params, "lr": self.lr}]
        else:
            self.param_groups = [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": other_params, "lr": self.lr},
            ]

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

    def _encode_rgb_data(
        self,
        batched_nc_data: list[BatchedNCData],
        mask: torch.Tensor,
        encoders: nn.ModuleList,
    ) -> torch.Tensor:
        batched_rgb_data = cast(list[BatchedRGBData], batched_nc_data)
        assert len(batched_rgb_data) == len(
            encoders
        ), "Number of camera inputs does not match number of encoders."
        feats = []
        for encoder, input in zip(encoders, batched_rgb_data):
            last_frame = input.frame[:, -1, :, :, :]  # (B, 3, H, W)
            feats.append(encoder(last_frame))
        combined_feats = torch.stack(feats, dim=1)  # (B, num_cams, feat_dim)
        combined_feats *= mask.unsqueeze(-1)
        # (B, num_cams, feat_dim) -> (B, num_cams * feat_dim)
        return combined_feats.view(combined_feats.shape[0], -1)

    def _encode_depth_data(
        self,
        batched_nc_data: list[BatchedNCData],
        mask: torch.Tensor,
        encoders: nn.ModuleList,
    ) -> torch.Tensor:
        batched_depth_data = cast(list[BatchedDepthData], batched_nc_data)
        assert len(batched_depth_data) == len(
            encoders
        ), "Number of camera inputs does not match number of encoders."
        feats = []
        for encoder, input in zip(encoders, batched_depth_data):
            last_frame = input.frame[:, -1, :, :, :]  # (B, 1, H, W)
            feats.append(encoder(last_frame))
        combined_feats = torch.stack(feats, dim=1)  # (B, num_cams, feat_dim)
        combined_feats *= mask.unsqueeze(-1)
        # (B, num_cams, feat_dim) -> (B, num_cams * feat_dim)
        return combined_feats.view(combined_feats.shape[0], -1)

    def _encode_proprio(
        self,
        batch: BatchedInferenceInputs,
    ) -> dict[DataType, torch.Tensor]:
        """Encode all proprioceptive data with joint normalization."""
        # Concatenate all proprio data
        proprio_list = []
        for data_type in PROPRIO_DATA_TYPES:
            if data_type not in self.input_data_types:
                continue
            if data_type not in batch.inputs:
                continue

            batched_nc_data = batch.inputs[data_type]
            mask = batch.inputs_mask[data_type]

            if data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched_gripper_data = cast(
                    list[BatchedParallelGripperOpenAmountData], batched_nc_data
                )
                proprio_data = torch.cat(
                    [bgd.open_amount for bgd in batched_gripper_data], dim=-1
                )
            else:
                batched_joint_data = cast(list[BatchedJointData], batched_nc_data)
                proprio_data = torch.cat(
                    [bjd.value for bjd in batched_joint_data], dim=-1
                )

            last_proprio = proprio_data[:, -1, :]  # (B, num_features)
            masked_proprio = last_proprio * mask
            proprio_list.append(masked_proprio)

        # Concatenate all proprio together: (B, total_proprio_dim)
        if not proprio_list:
            return {}

        # Shape is (B, total_proprio_dim)
        all_proprio = torch.cat(proprio_list, dim=-1)

        # Normalize once on all proprio
        # Check if normalizer exists (it should if we have proprio data)
        if self.proprio_normalizer is None:
            raise ValueError(
                "Proprioception inputs were provided but no normalizer was available."
            )
        normalized_proprio = self.proprio_normalizer.normalize(all_proprio)

        # Split and encode each part
        features = {}
        for data_type, (start_idx, end_idx) in self.proprio_dims.items():
            if data_type in batch.inputs:
                proprio_slice = normalized_proprio[:, start_idx:end_idx]
                features[data_type] = self.encoders[data_type](proprio_slice)

        return features

    def _encode_pose_data(
        self,
        batched_nc_data: list[BatchedNCData],
        mask: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        batched_pose_data = cast(list[BatchedPoseData], batched_nc_data)
        # Gives (B, num_poses, pose_dim)
        pose_data = torch.stack([bpd.pose[:, -1] for bpd in batched_pose_data], dim=1)
        masked_pose_data = pose_data * mask.unsqueeze(-1)
        # PoseEncoder expects flat (B, num_poses * pose_dim) input
        batch_size = masked_pose_data.shape[0]
        return encoder(masked_pose_data.reshape(batch_size, -1))  # (B, feat_dim)

    def _encode_point_cloud_data(
        self,
        batched_nc_data: list[BatchedNCData],
        mask: torch.Tensor,
        encoders: nn.ModuleList,
    ) -> torch.Tensor:
        batched_pc_data = cast(list[BatchedPointCloudData], batched_nc_data)
        assert len(batched_pc_data) == len(
            encoders
        ), "Number of point cloud inputs does not match number of encoders."
        feats = []
        for encoder, input in zip(encoders, batched_pc_data):
            last_points = input.points[:, -1, :, :]  # (B, 3, N)
            feats.append(encoder(last_points))
        combined_feats = torch.stack(feats, dim=1)  # (B, feat_dim, N)
        combined_feats *= mask.unsqueeze(-1)
        #  (B, num_points, feat_dim) -> (B, num_points * feat_dim)
        return combined_feats.view(combined_feats.shape[0], -1)

    def _encode_language_data(
        self,
        batched_nc_data: list[BatchedNCData],
        mask: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        """Encode language data using input_ids and attention_mask.

        Args:
            batched_nc_data: List of BatchedLanguageData instances
            mask: Mask tensor (not used for language data)
            encoder: Encoder module

        Returns:
            Encoded language features (B, final_dim)
        """
        batched_language_data = cast(list[BatchedLanguageData], batched_nc_data)

        # Grab the last language group and last timestep
        language_data = batched_language_data[-1]

        input_ids = language_data.input_ids  # (B, T, L)
        attention_mask = language_data.attention_mask  # (B, T, L)

        # Flatten B and T dimensions
        B, T, L = input_ids.shape
        input_ids = input_ids.view(B * T, L)  # (B*T, L)
        attention_mask = attention_mask.view(B * T, L)  # (B*T, L)

        # Embed tokens
        embeds = self.embedding_encoder(input_ids)  # (B*T, L, embed_dim)

        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1)  # (B*T, L, 1)
        sum_embeds = (embeds * mask_expanded).sum(dim=1)  # (B*T, embed_dim)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B*T, 1)
        pooled = sum_embeds / sum_mask  # (B*T, embed_dim)

        # Pass through MLP
        output = encoder[0](pooled)  # (B*T, output_dim)

        # Reshape back to (B, T, output_dim)
        output = output.view(B, T, -1)

        return output[:, -1, :]  # (B, final_dim)

    def _encode_inputs(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, torch.Tensor]:
        """Process all visual modalities and return features."""
        features: dict[DataType, torch.Tensor] = {}

        # Encode proprio types together with joint normalization
        proprio_features = self._encode_proprio(batch)
        features.update(proprio_features)

        for data_type, batched_nc_data in batch.inputs.items():
            # Skip proprio types since they're already handled
            if data_type in PROPRIO_DATA_TYPES:
                continue
            if data_type not in self.input_data_types:
                continue

            mask = batch.inputs_mask[data_type]
            if data_type == DataType.RGB_IMAGES:
                features[data_type] = self._encode_rgb_data(
                    batched_nc_data,
                    mask,
                    cast(nn.ModuleList, self.encoders[data_type]),
                )
            elif data_type == DataType.DEPTH_IMAGES:
                features[data_type] = self._encode_depth_data(
                    batched_nc_data,
                    mask,
                    cast(nn.ModuleList, self.encoders[data_type]),
                )
            elif data_type == DataType.POSES:
                features[data_type] = self._encode_pose_data(
                    batched_nc_data,
                    mask,
                    self.encoders[data_type],
                )
            elif data_type == DataType.POINT_CLOUDS:
                features[data_type] = self._encode_point_cloud_data(
                    batched_nc_data,
                    mask,
                    cast(nn.ModuleList, self.encoders[data_type]),
                )
            elif data_type == DataType.LANGUAGE:
                features[data_type] = self._encode_language_data(
                    batched_nc_data,
                    mask,
                    self.encoders[data_type],
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
        encoded_features = self._encode_inputs(batch)
        combined_features = self.fusion(encoded_features)

        # Forward through MLP to get entire sequence
        mlp_out = self.mlp(combined_features)
        action_preds = mlp_out.view(
            batch_size, self.output_prediction_horizon, self.max_output_size
        )
        return action_preds

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Perform inference to predict action sequence.

        Output will look like:
        {
            DataType.JOINT_TARGET_POSITIONS: [
                BatchedJointData,  # for each joint output
            ],
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: [
                BatchedParallelGripperOpenAmountData,  # for each gripper output
            ],
            ...
        }

        Args:
            batch: Input batch with observations

        Returns:
            BatchedInferenceOutputs: Model predictions with action sequences
        """
        action_preds = self._predict_action(batch)
        # (B, T, action_dim)
        predictions = self.action_normalizer.unnormalize(action_preds)

        output_tensors: dict[DataType, list[BatchedNCData]] = {}
        for data_type in self.ordered_output_data_types:
            start_idx, end_idx = self.output_dims[data_type]
            output_width = end_idx - start_idx
            dt_preds = predictions[:, :, start_idx:end_idx]  # (B, T, dt_size)
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs: list[BatchedNCData] = []
                for i in range(output_width):
                    joint_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(BatchedJointData(value=joint_preds))
                output_tensors[data_type] = batched_outputs
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                batched_outputs = []
                for i in range(output_width):
                    gripper_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(
                        BatchedParallelGripperOpenAmountData(open_amount=gripper_preds)
                    )
                output_tensors[data_type] = batched_outputs
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")
        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Predicts action sequences and computes mean squared error loss
        against target actions.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        inference_sample = BatchedInferenceInputs(
            inputs=batch.inputs,
            inputs_mask=batch.inputs_mask,
            batch_size=batch.batch_size,
        )

        if not set(self.output_data_types).issubset(set(batch.outputs.keys())):
            raise ValueError(
                "Batch is missing required output types."
                f" Model needs {self.output_data_types}, batch has "
                f"{list(batch.outputs.keys())}"
            )

        action_targets = []
        for data_type in self.ordered_output_data_types:
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend([bjd.value for bjd in batched_joints])
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData], batch.outputs[data_type]
                )
                action_targets.extend([gripper.open_amount for gripper in grippers])
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)
        action_data = action_data.view(
            batch.batch_size, self.output_prediction_horizon, -1
        )  # (B, T, action_dim)
        target_actions = self.action_normalizer.normalize(action_data)
        action_predictions = self._predict_action(inference_sample)

        # Build a per-dimension action mask from output masks so missing robot
        # channels in mixed-robot batches do not contribute to loss.
        action_mask = torch.cat(
            [
                batch.outputs_mask[data_type]
                for data_type in self.ordered_output_data_types
            ],
            dim=-1,
        ).unsqueeze(1)
        action_mask = action_mask.expand(-1, self.output_prediction_horizon, -1)

        losses: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        if self.training:
            absolute_error = torch.abs(action_predictions - target_actions)
            masked_absolute_error = absolute_error * action_mask
            losses["l1_loss"] = masked_absolute_error.sum() / torch.clamp(
                action_mask.sum(), min=1.0
            )

        return BatchedTrainingOutputs(
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(
        self,
    ) -> list[torch.optim.Optimizer]:
        """Configure optimizer with different learning rates.

        Uses separate learning rates for image encoder backbone and other
        model parameters.

        Returns:
            list[torch.optim.Optimizer]: List of optimizers for model parameters
        """
        return [torch.optim.AdamW(self.param_groups, weight_decay=self.weight_decay)]

    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        """Get the input data types supported by this model.

        Returns:
            set[DataType]: Set of supported input data types
        """
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.POSES,
            DataType.RGB_IMAGES,
            DataType.DEPTH_IMAGES,
            DataType.POINT_CLOUDS,
            DataType.LANGUAGE,
        }

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        """Get the output data types supported by this model.

        Returns:
            set[DataType]: Set of supported output data types
        """
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        }
