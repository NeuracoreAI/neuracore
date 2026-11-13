"""CNN+MLP model for robot manipulation with sequence prediction.

This module implements a simple baseline model that combines convolutional
neural networks for visual feature extraction with multi-layer perceptrons
for action sequence prediction. The model processes single timestep inputs
and outputs entire action sequences.

When ``DataType.DONE`` is an output, a separate classification head also
predicts whether the task is done at the current step, from attention-pooled
image features and optionally the joint and gripper state. Without DONE the
model is plain CNNMLP.
"""

import os
from enum import Enum
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from neuracore_types import (
    BatchedDepthData,
    BatchedDoneData,
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
    DoneDataStats,
    GPUType,
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
from neuracore.ml.utils.hf_hub_retry import call_with_hf_hub_retry
from neuracore.ml.utils.pretrained_cache import ensure_pretrained_model

from .modules import ImageEncoder

proprio_normalizer = MeanStdNormalizer  # or MinMaxNormalizer
action_normalizer = MeanStdNormalizer  # or MinMaxNormalizer
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]
# Done frames are rare, so BCE up-weights them by (1 - p) / p. The default
# multiplier stops here so one missed frame cannot dominate the loss. At 50 the
# weight is exact down to a ~2% positive rate.
MAX_DONE_POS_WEIGHT = 50.0

_ACTION_OUTPUT_DATA_TYPES = frozenset({
    DataType.JOINT_TARGET_POSITIONS,
    DataType.JOINT_POSITIONS,
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
})

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


class ImageBackbone(str, Enum):
    """ResNet offered as the image encoder backbone."""

    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"


class CNNMLP(NeuracoreModel):
    """CNN+MLP model with single timestep input and sequence output.

    A baseline model architecture that uses separate CNN encoders for each
    camera view, combines visual features with proprioceptive state, and
    predicts entire action sequences through a multi-layer perceptron.

    The model processes current observations and outputs a fixed-length
    sequence of future actions, making it suitable for action chunking
    approaches in robot manipulation.

    When ``DataType.DONE`` is an output, a lightweight MLP on spatially
    attended image features also predicts the current-step done flag.
    ``done_use_proprio`` adds the current joint positions and gripper opening.
    ``detach_done_encoder`` stops the done loss from updating any of those
    inputs.
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
        image_backbone: ImageBackbone = ImageBackbone.RESNET18,
        use_resnet_stats: bool = True,
        hidden_dim: int = 512,
        cnn_output_dim: int = 512,
        num_layers: int = 3,
        lr: float = 1e-4,
        freeze_backbone: bool = False,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        done_loss_weight: float = 0.1,
        done_head_dim: int = 128,
        done_head_dropout: float = 0.5,
        detach_done_encoder: bool = False,
        done_use_proprio: bool = False,
        done_max_pos_weight: float = MAX_DONE_POS_WEIGHT,
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
            done_loss_weight: Weight for the done-flag BCE loss relative to the
                L1 action loss
            done_head_dim: Hidden width of the done MLP. The default stays
                narrow so the head is less able to memorize training episodes.
            done_head_dropout: Dropout probability inside the done MLP.
            detach_done_encoder: When true, the done loss does not flow into
                any done-head input: the image feature maps and the joint and
                gripper state. The done modules still train. The action loss
                still updates the encoder.
            done_use_proprio: When true, a zero-initialized linear map adds the
                current normalized joint positions and gripper opening onto the
                done logits.
            done_max_pos_weight: Upper bound on the BCE ``pos_weight``
                ``(1 - p) / p``, where ``p`` is the dataset share of done
                frames. Raise it if done recall stays low; lower it if the
                done loss is noisy or the head predicts done too early.
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
        self.done_loss_weight = done_loss_weight
        self.done_head_dim = done_head_dim
        self.done_head_dropout = done_head_dropout
        self.detach_done_encoder = detach_done_encoder
        self.done_use_proprio = done_use_proprio
        self.done_max_pos_weight = done_max_pos_weight
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

        max_rgb_cameras = 0
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats],
                self.input_dataset_statistics[DataType.RGB_IMAGES],
            )
            max_cameras = len(stats)
            max_rgb_cameras = max_cameras
            self.rgb_normalizers = nn.ModuleList()
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
                self.rgb_normalizers.append(T.Normalize(mean=mean, std=std))
                self.encoders[DataType.RGB_IMAGES].append(
                    ImageEncoder(output_dim=cnn_output_dim, backbone=image_backbone)
                )

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
                self.encoders[DataType.DEPTH_IMAGES].append(
                    DepthImageEncoder(output_dim=cnn_output_dim)
                )

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

            ensure_pretrained_model(LANGUAGE_MODEL_NAME)
            _tokenizer = call_with_hf_hub_retry(
                partial(AutoTokenizer.from_pretrained, LANGUAGE_MODEL_NAME)
            )
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

        # Continuous action outputs only (DONE is handled by the done head).
        self.action_output_data_types = [
            data_type
            for data_type in self.ordered_output_data_types
            if data_type in _ACTION_OUTPUT_DATA_TYPES
        ]
        if not self.action_output_data_types:
            raise ValueError(
                "CNNMLP requires at least one continuous action "
                "output type (joint or gripper positions); DONE alone is not "
                "supported."
            )

        output_stats = []
        self.output_dims: dict[DataType, tuple[int, int]] = {}
        current_output_dim = 0
        for data_type in self.action_output_data_types:
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

        self.action_normalizer = action_normalizer(
            name="actions", statistics=output_stats
        )
        self.proprio_normalizer = (
            proprio_normalizer(name="proprioception", statistics=input_stats)
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

        self._setup_done_head(max_rgb_cameras)

        # Setup parameter groups
        self._setup_optimizer_param_groups()

    def _setup_done_head(self, max_rgb_cameras: int) -> None:
        """Create the done classifier when DONE is an output.

        Leaves the model untouched without DONE, so no done modules or buffers
        appear in its state dict and older checkpoints still load.

        Args:
            max_rgb_cameras: Number of RGB cameras the model encodes.

        Raises:
            ValueError: If DONE is requested without exactly one done flag in
                the dataset statistics, without RGB images, or with
                ``done_use_proprio`` but no joint or gripper input.
        """
        self.done_head: nn.Sequential | None = None
        self.done_proprio_to_logits: nn.Linear | None = None
        self.done_proprio_slices: list[tuple[int, int]] = []
        self.vision_cond_dim = 0
        if DataType.DONE not in self.output_data_types:
            return

        done_stats = cast(
            list[DoneDataStats],
            self.output_dataset_statistics[DataType.DONE],
        )
        # A robot logs a single done stream holding one 0/1 flag.
        if len(done_stats) != 1 or len(done_stats[0].done.mean) != 1:
            raise ValueError(
                "DONE is configured as an output but the dataset statistics do "
                "not hold exactly one done flag."
            )
        if max_rgb_cameras == 0:
            raise ValueError("CNNMLP requires RGB_IMAGES input to predict DONE flags.")

        # Done labels are 0/1 and rare, so the mean is the positive rate and
        # pos_weight = (1 - p) / p rebalances BCE toward the positive class.
        # Cap it at done_max_pos_weight so a very small p cannot make one
        # missed frame dominate the batch.
        pos_rate = torch.as_tensor(done_stats[0].done.mean, dtype=torch.float32).clamp(
            1e-4, 1.0 - 1e-4
        )
        pos_weight = ((1.0 - pos_rate) / pos_rate).clamp(max=self.done_max_pos_weight)
        self.register_buffer("done_pos_weight", pos_weight)

        # The head pools each camera's projected ResNet feature map, the map
        # that ImageEncoder flattens into its linear layer for the action MLP.
        # The score starts at zero so the pool begins as a uniform average,
        # then learns which locations matter for the done flag.
        self.vision_cond_dim = self.cnn_output_dim * max_rgb_cameras
        self.done_spatial_score = nn.Linear(self.cnn_output_dim, 1)
        nn.init.zeros_(self.done_spatial_score.weight)
        nn.init.zeros_(self.done_spatial_score.bias)
        self.done_head = nn.Sequential(
            nn.Linear(self.vision_cond_dim, self.done_head_dim),
            nn.ReLU(),
            nn.Dropout(self.done_head_dropout),
            nn.Linear(self.done_head_dim, 1),
        )
        if self.done_use_proprio:
            self.done_proprio_slices = [
                self.proprio_dims[data_type]
                for data_type in (
                    DataType.JOINT_POSITIONS,
                    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
                )
                if data_type in self.proprio_dims
            ]
            done_proprio_dim = sum(
                end - start for start, end in self.done_proprio_slices
            )
            if done_proprio_dim == 0:
                raise ValueError(
                    "done_use_proprio requires JOINT_POSITIONS or "
                    "PARALLEL_GRIPPER_OPEN_AMOUNTS inputs."
                )
            # Zero weights keep the first step identical to the vision head.
            # Joint and gripper values are added onto the logits, not mixed
            # inside the vision MLP.
            self.done_proprio_to_logits = nn.Linear(done_proprio_dim, 1)
            nn.init.zeros_(self.done_proprio_to_logits.weight)
            nn.init.zeros_(self.done_proprio_to_logits.bias)

    def _spatial_attention_pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Pool a feature map by attending over its spatial locations.

        The pooled vector is a weighted sum of the features. A zero score layer
        is a uniform average, which is where training starts. CNNMLP has no
        position embeddings, so the score reads the features alone.

        Args:
            feature_map: Feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Pooled features of shape (B, C).
        """
        tokens = feature_map.flatten(2).transpose(1, 2)  # (B, HW, C)
        scores = self.done_spatial_score(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bn,bnc->bc", weights, tokens)

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode the last RGB frame of each camera.

        Args:
            batched_nc_data: One BatchedRGBData per camera.
            mask: Camera mask of shape (B, num_cams).
            encoders: One ImageEncoder per camera.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Flattened camera features
            ``(B, num_cams * feat_dim)`` for the action MLP, plus pooled
            per-camera features ``(B, vision_cond_dim)`` for the done head, or
            None without one.
        """
        batched_rgb_data = cast(list[BatchedRGBData], batched_nc_data)
        assert len(batched_rgb_data) == len(
            encoders
        ), "Number of camera inputs does not match number of encoders."
        feats = []
        done_pooled = []
        for cam_idx, (normalizer, encoder, input) in enumerate(
            zip(self.rgb_normalizers, encoders, batched_rgb_data)
        ):
            last_frame = input.frame[:, -1, :, :, :]  # (B, 3, H, W)
            if self.done_head is None:
                feats.append(encoder(normalizer(last_frame)))
                continue
            feat, feature_map = encoder.forward_with_spatial(normalizer(last_frame))
            feats.append(feat)
            # Mask like the action features, so a missing camera pools to zero.
            feature_map = feature_map * mask[:, cam_idx].view(-1, 1, 1, 1)
            if self.detach_done_encoder:
                feature_map = feature_map.detach()
            done_pooled.append(self._spatial_attention_pool(feature_map))
        combined_feats = torch.stack(feats, dim=1)  # (B, num_cams, feat_dim)
        combined_feats *= mask.unsqueeze(-1)
        vision_cond = torch.cat(done_pooled, dim=-1) if done_pooled else None
        # (B, num_cams, feat_dim) -> (B, num_cams * feat_dim)
        return combined_feats.view(combined_feats.shape[0], -1), vision_cond

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
    ) -> tuple[dict[DataType, torch.Tensor], torch.Tensor | None]:
        """Encode all proprioceptive data with joint normalization.

        Args:
            batch: Input batch with observations

        Returns:
            tuple[dict[DataType, torch.Tensor], torch.Tensor | None]: Encoded
            features per proprio type, plus the normalized proprio state
            ``(B, total_proprio_dim)``, or None without proprio inputs.
        """
        # Concatenate all proprio data
        proprio_list = []
        for data_type in PROPRIO_DATA_TYPES:
            if data_type not in self.input_data_types:
                continue
            if data_type not in batch.inputs:
                start_idx, end_idx = self.proprio_dims[data_type]
                proprio_list.append(
                    torch.zeros(
                        (len(batch), end_idx - start_idx),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
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
            return {}, None

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

        return features, normalized_proprio

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
    ) -> tuple[dict[DataType, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        """Process all input modalities and return features.

        Args:
            batch: Input batch with observations

        Returns:
            tuple: Encoded features per data type, pooled image features for
            the done head (None without one), and the normalized proprio
            state (None without proprio inputs).
        """
        features: dict[DataType, torch.Tensor] = {}
        vision_cond: torch.Tensor | None = None

        # Encode proprio types together with joint normalization
        proprio_features, proprio_state = self._encode_proprio(batch)
        features.update(proprio_features)

        for data_type, batched_nc_data in batch.inputs.items():
            # Skip proprio types since they're already handled
            if data_type in PROPRIO_DATA_TYPES:
                continue
            if data_type not in self.input_data_types:
                continue

            mask = batch.inputs_mask[data_type]
            if data_type == DataType.RGB_IMAGES:
                features[data_type], vision_cond = self._encode_rgb_data(
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
        if self.done_head is not None and vision_cond is None:
            # No RGB in this batch: the done head sees every camera as missing.
            vision_cond = torch.zeros(
                len(batch), self.vision_cond_dim, device=self.device
            )
        return features, vision_cond, proprio_state

    def _predict_action(
        self, batch: BatchedInferenceInputs
    ) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
        """Predict action sequence for the given batch.

        Processes visual and proprioceptive inputs through separate encoders,
        combines features, and predicts the entire action sequence through MLP.
        With a done head, also predicts the current-step done logits from the
        pooled image features (and optionally the proprio state) only.

        Args:
            batch: Input batch with observations

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor | None]: Predicted action
            sequence [B, T, action_dim] and done logits [B, 1, 1], or
            None without a done head.
        """
        batch_size = len(batch)
        encoded_features, vision_cond, proprio_state = self._encode_inputs(batch)
        combined_features = self.fusion(encoded_features)

        # Forward through MLP to get entire sequence
        mlp_out = self.mlp(combined_features)
        action_preds = mlp_out.view(
            batch_size, self.output_prediction_horizon, self.max_output_size
        )
        done_logits = None
        if self.done_head is not None:
            assert vision_cond is not None
            done_logits = self._predict_done_logits(vision_cond, proprio_state)
        return action_preds, done_logits

    def _predict_done_logits(
        self,
        vision_cond: torch.Tensor,
        proprio_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the current-step done flag.

        The MLP reads pooled image features. When ``done_use_proprio`` is set,
        a linear map adds the current joint positions and gripper opening onto
        those logits. ``detach_done_encoder`` detaches that state as well as
        the image features.

        Args:
            vision_cond: Pooled image features with shape (B, vision_cond_dim).
            proprio_state: Normalized proprioception with shape (B, state_dim),
                or None when the model has no proprioception.

        Returns:
            torch.Tensor: Done logits with shape (B, 1, 1).
        """
        assert self.done_head is not None
        logits = self.done_head(vision_cond)
        if self.done_proprio_to_logits is not None:
            if proprio_state is None:
                source_dim = max(end for _, end in self.done_proprio_slices)
                proprio_state = torch.zeros(
                    vision_cond.shape[0],
                    source_dim,
                    device=vision_cond.device,
                    dtype=vision_cond.dtype,
                )
            proprio = torch.cat(
                [
                    proprio_state[:, start:end]
                    for start, end in self.done_proprio_slices
                ],
                dim=-1,
            )
            if self.detach_done_encoder:
                proprio = proprio.detach()
            logits = logits + self.done_proprio_to_logits(proprio)
        return logits.unsqueeze(1)

    def _build_done_mask(self, batch: BatchedTrainingSamples) -> torch.Tensor:
        """Build the mask for the done target.

        Args:
            batch: Training batch.

        Returns:
            torch.Tensor: Mask with shape (B, 1).
        """
        if DataType.DONE not in batch.outputs_mask:
            return torch.zeros(batch.batch_size, 1, device=self.device)
        return batch.outputs_mask[DataType.DONE][:, :1]

    @staticmethod
    def _done_classification_metrics(
        done_logits: torch.Tensor,
        done_targets: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score the current-step done flag at probability 0.5.

        A logit at or above zero is a positive prediction. Masked features
        are left out of every count. An empty denominator is reported as
        zero: no predicted positives gives precision 0, and no labeled
        positives gives recall 0.

        Args:
            done_logits: Logits with shape (B, 1, D).
            done_targets: Targets with shape (B, 1, D), in ``[0, 1]``.
            done_mask: Feature mask with shape (B, D).

        Returns:
            dict[str, torch.Tensor]: ``done_accuracy``, ``done_precision``,
            and ``done_recall``.
        """
        targets = done_targets.clamp(0.0, 1.0)
        predicted_positive = (done_logits >= 0).to(dtype=targets.dtype)
        actual_positive = (targets >= 0.5).to(dtype=targets.dtype)
        mask = done_mask.unsqueeze(1).to(dtype=targets.dtype)

        true_positive = (predicted_positive * actual_positive * mask).sum()
        predicted_positive_count = (predicted_positive * mask).sum()
        actual_positive_count = (actual_positive * mask).sum()
        correct = (
            (predicted_positive == actual_positive).to(dtype=targets.dtype) * mask
        ).sum()
        evaluated = mask.sum()
        return {
            "done_accuracy": correct / torch.clamp(evaluated, min=1.0),
            "done_precision": true_positive
            / torch.clamp(predicted_positive_count, min=1.0),
            "done_recall": true_positive / torch.clamp(actual_positive_count, min=1.0),
        }

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
            and, with DONE as an output, the current-step done probability as
            a single ``BatchedDoneData`` of shape (B, 1, 1).
        """
        action_preds, done_logits = self._predict_action(batch)
        # (B, T, action_dim)
        predictions = self.action_normalizer.unnormalize(action_preds)

        output_tensors: dict[DataType, list[BatchedNCData]] = {}
        for data_type in self.action_output_data_types:
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
                raise ValueError(f"Unsupported action output data type: {data_type}")
        if done_logits is not None:
            output_tensors[DataType.DONE] = [
                BatchedDoneData(done=torch.sigmoid(done_logits))
            ]
        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Predicts action sequences and computes masked L1 loss
        against target actions. When DONE outputs are configured, also
        computes a masked BCE loss on the current done flag, plus accuracy,
        precision, and recall at probability 0.5. ``pos_weight`` comes from the
        dataset positive rate and is capped at ``done_max_pos_weight`` so rare
        positives cannot dominate.

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

        action_targets = []
        for data_type in self.action_output_data_types:
            start, end = self.output_dims[data_type]
            dim = end - start
            if data_type not in batch.outputs:
                action_targets.append(
                    torch.zeros(
                        batch.batch_size,
                        self.output_prediction_horizon,
                        dim,
                        device=self.device,
                    )
                )
            elif data_type in [
                DataType.JOINT_TARGET_POSITIONS,
                DataType.JOINT_POSITIONS,
            ]:
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
                raise ValueError(f"Unsupported action output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)
        action_data = action_data.view(
            batch.batch_size, self.output_prediction_horizon, -1
        )  # (B, T, action_dim)
        target_actions = self.action_normalizer.normalize(action_data)
        action_predictions, done_logits = self._predict_action(inference_sample)

        # Build a per-dimension action mask from output masks so missing robot
        # channels in mixed-robot batches do not contribute to loss.
        action_mask = torch.cat(
            [
                (
                    batch.outputs_mask[data_type]
                    if data_type in batch.outputs_mask
                    else torch.zeros(
                        batch.batch_size,
                        self.output_dims[data_type][1] - self.output_dims[data_type][0],
                        device=self.device,
                    )
                )
                for data_type in self.action_output_data_types
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

        if done_logits is not None:
            if DataType.DONE in batch.outputs:
                done_output = cast(list[BatchedDoneData], batch.outputs[DataType.DONE])[
                    0
                ]
                # The first step of the chunk is the current action step.
                current_targets = done_output.done[:, :1]  # (B, 1, 1)
            else:
                current_targets = torch.zeros_like(done_logits)

            done_mask = self._build_done_mask(batch)  # (B, 1)
            bce = F.binary_cross_entropy_with_logits(
                done_logits,
                current_targets.clamp(0.0, 1.0),
                reduction="none",
                pos_weight=self.done_pos_weight,
            )
            done_bce_loss = (bce * done_mask.unsqueeze(1)).sum() / torch.clamp(
                done_mask.sum(), min=1.0
            )
            # Like l1_loss, the done loss is only returned in training mode.
            if self.training:
                losses["done_bce_loss"] = self.done_loss_weight * done_bce_loss
            metrics["done_bce_loss"] = done_bce_loss
            metrics.update(
                self._done_classification_metrics(
                    done_logits, current_targets, done_mask
                )
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
            DataType.DONE,
        }

    @staticmethod
    def get_supported_gpus() -> frozenset[GPUType]:
        """Get the GPU types recommended for training this model."""
        return frozenset({
            GPUType.NVIDIA_A100_80GB,
            GPUType.NVIDIA_TESLA_A100,
            GPUType.NVIDIA_TESLA_V100,
            GPUType.NVIDIA_TESLA_T4,
            GPUType.NVIDIA_L4,
        })
