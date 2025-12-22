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

        data_stats: dict[DataType, DataItemStats] = {}

        if DataType.JOINT_POSITIONS in self.data_types:
            stats = self.dataset_statistics[DataType.JOINT_POSITIONS]
            stats = cast(list[JointDataStats], stats)
            combined_stats = DataItemStats()
            for stat in stats:
                combined_stats = combined_stats.concatenate(stat.value)
            data_stats[DataType.JOINT_POSITIONS] = combined_stats
            if DataType.JOINT_POSITIONS in self.input_data_types:
                self.encoder_output_dims[DataType.JOINT_POSITIONS] = cnn_output_dim
                self.encoders[DataType.JOINT_POSITIONS] = nn.Linear(
                    len(stats), cnn_output_dim
                )

        if DataType.JOINT_TARGET_POSITIONS in self.data_types:
            stats = self.dataset_statistics[DataType.JOINT_TARGET_POSITIONS]
            stats = cast(list[JointDataStats], stats)
            combined_stats = DataItemStats()
            for stat in stats:
                combined_stats = combined_stats.concatenate(stat.value)
            data_stats[DataType.JOINT_TARGET_POSITIONS] = combined_stats
            if DataType.JOINT_TARGET_POSITIONS in self.input_data_types:
                self.encoder_output_dims[DataType.JOINT_TARGET_POSITIONS] = (
                    cnn_output_dim
                )
                self.encoders[DataType.JOINT_TARGET_POSITIONS] = nn.Linear(
                    len(stats), cnn_output_dim
                )

        if DataType.JOINT_VELOCITIES in self.data_types:
            stats = self.dataset_statistics[DataType.JOINT_VELOCITIES]
            stats = cast(list[JointDataStats], stats)
            combined_stats = DataItemStats()
            for stat in stats:
                combined_stats = combined_stats.concatenate(stat.value)
            data_stats[DataType.JOINT_VELOCITIES] = combined_stats
            if DataType.JOINT_VELOCITIES in self.input_data_types:
                self.encoder_output_dims[DataType.JOINT_VELOCITIES] = cnn_output_dim
                self.encoders[DataType.JOINT_VELOCITIES] = nn.Linear(
                    len(stats), cnn_output_dim
                )

        if DataType.JOINT_TORQUES in self.data_types:
            stats = self.dataset_statistics[DataType.JOINT_TORQUES]
            stats = cast(list[JointDataStats], stats)
            combined_stats = DataItemStats()
            for stat in stats:
                combined_stats = combined_stats.concatenate(stat.value)
            data_stats[DataType.JOINT_TORQUES] = combined_stats
            if DataType.JOINT_TORQUES in self.input_data_types:
                self.encoder_output_dims[DataType.JOINT_TORQUES] = cnn_output_dim
                self.encoders[DataType.JOINT_TORQUES] = nn.Linear(
                    len(stats), cnn_output_dim
                )

        if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.data_types:
            stats = self.dataset_statistics[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS]
            stats = cast(list[ParallelGripperOpenAmountDataStats], stats)
            combined_stats = DataItemStats()
            for stat in stats:
                combined_stats = combined_stats.concatenate(stat.open_amount)
            data_stats[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS] = combined_stats
            if DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS in self.input_data_types:
                self.encoder_output_dims[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS] = (
                    cnn_output_dim
                )
                self.encoders[DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS] = nn.Linear(
                    len(stats), cnn_output_dim
                )

        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats], self.dataset_statistics[DataType.RGB_IMAGES]
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
                list[CameraDataStats], self.dataset_statistics[DataType.DEPTH_IMAGES]
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
                self.dataset_statistics[DataType.POINT_CLOUDS],
            )
            max_pcs = len(stats)
            self.encoder_output_dims[DataType.POINT_CLOUDS] = max_pcs * cnn_output_dim
            self.encoders[DataType.POINT_CLOUDS] = nn.ModuleList()
            for i in range(max_pcs):
                encoder = PointCloudEncoder(output_dim=cnn_output_dim)
                self.encoders[DataType.POINT_CLOUDS].append(encoder)

        # All poses will share the same encoder
        if DataType.POSES in self.input_data_types:
            stats = self.dataset_statistics[DataType.POSES]
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

        self.max_output_size = 0
        output_stats = []
        # Flatten norm_means into single parameters for outputs
        for data_type in self.output_data_types:
            output_stats.append(data_stats[data_type])
            self.max_output_size += len(self.dataset_statistics[data_type])

        input_stats = []
        self.proprio_dims = {}
        current_dim = 0
        for data_type in self.input_data_types:
            if data_type not in data_stats:
                continue
            input_stats.append(data_stats[data_type])
            dim = len(data_stats[data_type].mean)
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
        backbone_params, other_params = [], []
        for name, param in self.named_parameters():
            if any(backbone in name for backbone in ["rgb", "depth", "point_cloud"]):
                backbone_params.append(param)
            else:
                other_params.append(param)
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
        for data_type in [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        ]:
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

        all_proprio = torch.cat(proprio_list, dim=-1)

        # Normalize once on all proprio
        if self.proprio_normalizer is None:
            raise ValueError(
                "Proprioception inputs were provided but no statistics were available."
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
        return encoder(masked_pose_data)  # (B, feat_dim)

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
            if data_type in [
                DataType.JOINT_POSITIONS,
                DataType.JOINT_VELOCITIES,
                DataType.JOINT_TORQUES,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
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
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: [
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
        start_slice_idx = 0
        for data_type in self.output_data_types:
            end_slice_idx = start_slice_idx + len(self.dataset_statistics[data_type])
            dt_preds = predictions[
                :, :, start_slice_idx:end_slice_idx
            ]  # (B, T, dt_size)
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs = []
                for i in range(len(self.dataset_statistics[data_type])):
                    joint_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(BatchedJointData(value=joint_preds))
                output_tensors[data_type] = batched_outputs
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched_outputs = []
                for i in range(len(self.dataset_statistics[data_type])):
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

        if set(batch.outputs.keys()) != set(self.output_data_types):
            raise ValueError(
                "Batch outputs do not match model output configuration."
                f" Expected {self.output_data_types}, got {list(batch.outputs.keys())}"
            )

        action_targets = []
        for i, data_type in enumerate(self.output_data_types):
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend([bjd.value for bjd in batched_joints])
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData], batch.outputs[data_type]
                )
                action_targets.extend([gripper.open_amount for gripper in grippers])
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)  # (B, 1, T * action_dim)
        action_data = action_data.view(
            batch.batch_size, self.output_prediction_horizon, -1
        )  # (B, T, action_dim)
        target_actions = self.action_normalizer.normalize(action_data)
        action_predictions = self._predict_action(inference_sample)

        losses: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        if self.training:
            losses["l1_loss"] = nn.functional.l1_loss(
                action_predictions, target_actions
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
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        }
