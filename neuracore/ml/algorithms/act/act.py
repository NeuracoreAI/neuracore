"""ACT: Action Chunking with Transformers implementation.

This module implements the ACT (Action Chunking with Transformers) model
from "Learning fine-grained bimanual manipulation with low-cost hardware"
(Zhao et al., 2023). ACT uses a transformer architecture with latent variable
modeling to predict action sequences for robot manipulation tasks.

Reference: Zhao, Tony Z., et al. "Learning fine-grained bimanual manipulation
with low-cost hardware." arXiv preprint arXiv:2304.13705 (2023).
"""

import logging
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    BatchedRGBData,
    CameraDataStats,
    DataItemStats,
    DataType,
    JointDataStats,
    ModelInitDescription,
    ParallelGripperOpenAmountDataStats,
)

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import MeanStdNormalizer

from .modules import (
    ACTImageEncoder,
    PositionalEncoding,
    TransformerDecoder,
    TransformerEncoder,
)

logger = logging.getLogger(__name__)

PROPRIO_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer
ACTION_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]


class ACT(NeuracoreModel):
    """Implementation of ACT (Action Chunking Transformer) model.

    ACT is a transformer-based architecture that learns to predict sequences
    of robot actions by encoding visual observations and proprioceptive state
    into a latent representation, then decoding action chunks autoregressively.

    The model uses a variational autoencoder framework with separate encoders
    for visual features and action sequences, combined with a transformer
    decoder for action generation.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        nheads: int = 8,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
        use_resnet_stats: bool = True,
        lr: float = 1e-4,
        freeze_backbone: bool = False,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        kl_weight: float = 10.0,
        latent_dim: int = 512,
    ):
        """Initialize the ACT model.

        Args:
            model_init_description: Model initialization parameters
            hidden_dim: Hidden dimension for transformer layers
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            nheads: Number of attention heads
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            use_resnet_stats: Whether to use ResNet normalization statistics
            lr: Learning rate for main parameters
            freeze_backbone: Whether to freeze image encoder backbone
            lr_backbone: Learning rate for image encoder backbone
            weight_decay: Weight decay for optimizer
            kl_weight: Weight for KL divergence loss
            latent_dim: Dimension of latent variable space
        """
        super().__init__(model_init_description)
        self.hidden_dim = hidden_dim
        self.use_resnet_stats = use_resnet_stats
        self.freeze_backbone = freeze_backbone
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim

        data_stats: dict[DataType, DataItemStats] = {}

        # Setup proprioceptive data
        self.proprio_dims: dict[DataType, tuple[int, int]] = {}
        proprio_stats = []
        current_dim = 0

        for data_type in [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        ]:
            if data_type in self.data_types:
                if data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                    stats = cast(
                        list[ParallelGripperOpenAmountDataStats],
                        self.dataset_statistics[data_type],
                    )
                    combined_stats = DataItemStats()
                    for stat in stats:
                        combined_stats = combined_stats.concatenate(stat.open_amount)
                    data_stats[data_type] = combined_stats
                else:
                    stats = cast(
                        list[JointDataStats], self.dataset_statistics[data_type]
                    )
                    combined_stats = DataItemStats()
                    for stat in stats:
                        combined_stats = combined_stats.concatenate(stat.value)
                    data_stats[data_type] = combined_stats

                if data_type in self.input_data_types:
                    proprio_stats.append(combined_stats)
                    dim = len(combined_stats.mean)
                    self.proprio_dims[data_type] = (current_dim, current_dim + dim)
                    current_dim += dim

        # State embedding
        state_input_dim = current_dim
        self.state_embed = None
        if state_input_dim > 0:
            self.state_embed = nn.Linear(state_input_dim, hidden_dim)

        # Setup output data
        self.max_output_size = 0
        output_stats = []
        for data_type in self.output_data_types:
            if data_type == DataType.JOINT_TARGET_POSITIONS:
                stats = cast(
                    list[JointDataStats],
                    self.dataset_statistics[data_type],
                )
                combined_stats = DataItemStats()
                for stat in stats:
                    combined_stats = combined_stats.concatenate(stat.value)
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                stats = cast(
                    list[ParallelGripperOpenAmountDataStats],
                    self.dataset_statistics[data_type],
                )
                combined_stats = DataItemStats()
                for stat in stats:
                    combined_stats = combined_stats.concatenate(stat.open_amount)
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

            data_stats[data_type] = combined_stats
            output_stats.append(combined_stats)
            self.max_output_size += len(combined_stats.mean)

        # Action embedding
        self.action_embed = nn.Linear(self.max_output_size, hidden_dim)

        # Setup normalizers
        self.proprio_normalizer = PROPRIO_NORMALIZER(
            name="proprioception", statistics=proprio_stats
        )
        self.action_normalizer = ACTION_NORMALIZER(
            name="actions", statistics=output_stats
        )

        # Vision components
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats], self.dataset_statistics[DataType.RGB_IMAGES]
            )
            max_cameras = len(stats)
            self.image_encoders = nn.ModuleList()
            for i in range(max_cameras):
                if use_resnet_stats:
                    mean, std = RESNET_MEAN, RESNET_STD
                else:
                    mean_c_h_w, std_c_h_w = stats[i].frame.mean, stats[i].frame.std
                    mean = mean_c_h_w.mean(axis=(1, 2)).tolist()
                    std = std_c_h_w.mean(axis=(1, 2)).tolist()

                encoder = nn.ModuleDict({
                    "transform": torch.nn.Sequential(
                        T.Resize((224, 224)),
                        T.Normalize(mean=mean, std=std),
                    ),
                    "encoder": ACTImageEncoder(output_dim=hidden_dim),
                })
                self.image_encoders.append(encoder)

        # CLS token embedding for latent encoder
        self.cls_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Main transformer for vision and action generation
        self.transformer = nn.ModuleDict({
            "encoder": TransformerEncoder(
                d_model=hidden_dim,
                nhead=nheads,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            "decoder": TransformerDecoder(
                d_model=hidden_dim,
                nhead=nheads,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
        })

        # Separate encoder for latent space
        self.latent_encoder = TransformerEncoder(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Output heads
        self.action_head = nn.Linear(hidden_dim, self.max_output_size)

        # Latent projections
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

        # Query embedding for decoding
        self.query_embed = nn.Parameter(
            torch.randn(self.output_prediction_horizon, 1, hidden_dim)
        )

        # Additional position embeddings for proprio and latent
        self.additional_pos_embed = nn.Parameter(torch.randn(2, 1, hidden_dim))

        # Setup parameter groups
        self._setup_optimizer_param_groups()

    def _setup_optimizer_param_groups(self) -> None:
        """Setup parameter groups for optimizer."""
        backbone_params, other_params = [], []
        for name, param in self.named_parameters():
            if any(backbone in name for backbone in ["image_encoders"]):
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

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparametrization trick.

        During training, samples from the distribution N(mu, exp(logvar)).
        During inference, returns the mean mu.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution

        Returns:
            torch.Tensor: Sampled latent variable
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _combine_proprio(self, batch: BatchedInferenceInputs) -> torch.FloatTensor:
        """Combine different types of joint state data.

        Concatenates joint positions, velocities, and torques into a single
        feature vector, applying masks and normalization.

        Args:
            batch: Input batch containing joint state data

        Returns:
            torch.FloatTensor: Combined and normalized joint state features
        """
        if self.state_embed is None:
            return None

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
        all_proprio = torch.cat(proprio_list, dim=-1)

        # Normalize once on all proprio
        normalized_proprio = self.proprio_normalizer.normalize(all_proprio)

        return normalized_proprio

    def _encode_latent(
        self,
        state: torch.FloatTensor,
        actions: torch.FloatTensor,
        actions_mask: torch.FloatTensor,
        actions_sequence_mask: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Encode actions to latent space during training.

        Uses a separate transformer encoder to encode the action sequence
        along with proprioceptive state into latent distribution parameters.

        Args:
            state: Proprioceptive state features
            actions: Target action sequence
            actions_mask: Mask for valid action dimensions
            actions_sequence_mask: Mask for valid sequence positions

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor]: Latent mean and log variance
        """
        batch_size = state.shape[0]

        # Project joint positions and actions
        state_embed = (
            self.state_embed(state) if self.state_embed is not None else None
        )  # [B, H]
        action_embed = self.action_embed(
            actions * actions_mask.unsqueeze(1)
        )  # [B, T, H]

        # Reshape to sequence first
        state_embed = (
            state_embed.unsqueeze(0) if state_embed is not None else None
        )  # [1, B, H]
        action_embed = action_embed.transpose(0, 1)  # [T, B, H]

        # Concatenate [CLS, state_emb, action_embed]
        cls_token = self.cls_embed.expand(-1, batch_size, -1)  # [1, B, H]
        encoder_input = torch.cat([cls_token, state_embed, action_embed], dim=0)

        # # Update padding mask
        if actions_sequence_mask is not None:
            cls_joint_pad = torch.zeros(
                batch_size, 2, dtype=torch.bool, device=self.device
            )
            actions_sequence_mask = torch.cat(
                [cls_joint_pad, actions_sequence_mask], dim=1
            )

        # Add positional encoding
        encoder_input = self.pos_encoder(encoder_input)

        # Encode sequence
        memory = self.latent_encoder(
            encoder_input, src_key_padding_mask=actions_sequence_mask
        )

        # Get latent parameters from CLS token
        mu = self.latent_mu(memory[0])  # Take CLS token output
        logvar = self.latent_logvar(memory[0])
        return mu, logvar

    def _encode_visual(
        self,
        states: torch.FloatTensor,
        batched_nc_data: list[BatchedNCData],
        camera_images_mask: torch.FloatTensor,
        latent: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Encode visual inputs with latent and proprioceptive features.

        Processes RGB images through vision encoders and combines them with
        proprioceptive state and latent features using a transformer encoder.

        Args:
            states: Proprioceptive state features
            batched_nc_data: List of BatchedRGBData
            camera_images_mask: Mask for valid camera inputs
            latent: Latent features from action encoding

        Returns:
            torch.FloatTensor: Encoded visual and proprioceptive memory
        """
        batched_rgb_data = cast(list[BatchedRGBData], batched_nc_data)
        batch_size = states.shape[0]

        # Process images
        image_features = []
        image_pos = []
        for cam_id, (encoder_dict, input_rgb) in enumerate(
            zip(self.image_encoders, batched_rgb_data)
        ):
            last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)
            transformed = encoder_dict["transform"](last_frame)
            features, pos = encoder_dict["encoder"](
                transformed
            )  # Vision backbone provides features and pos
            features *= camera_images_mask[:, cam_id].view(batch_size, 1, 1, 1)
            image_features.append(features)
            image_pos.append(pos)

        # Combine image features and positions
        combined_features = torch.cat(image_features, dim=3)  # [B, C, H, W]
        combined_pos = torch.cat(image_pos, dim=3)  # [B, C, H, W]

        # Convert to sequence [H*W, B, C]
        src = combined_features.flatten(2).permute(2, 0, 1)
        pos = combined_pos.flatten(2).permute(2, 0, 1)

        # Process joint positions and latent
        state_features = (
            self.state_embed(states) if self.state_embed is not None else None
        )  # [B, H]

        # Stack latent and proprio features
        additional_features = torch.stack([latent, state_features], dim=0)  # [2, B, H]

        # Add position embeddings from additional_pos_embed
        additional_pos = self.additional_pos_embed.expand(
            -1, batch_size, -1
        )  # [2, B, H]

        # Concatenate everything
        src = torch.cat([additional_features, src], dim=0)
        pos = torch.cat([additional_pos, pos], dim=0)

        # Fuse positional embeddings with source
        src = src + pos

        # Encode
        memory = self.transformer["encoder"](src)

        return memory

    def _decode(
        self,
        latent: torch.FloatTensor,
        memory: torch.FloatTensor,
    ) -> torch.Tensor:
        """Decode latent and visual features to action sequence.

        Uses a transformer decoder with learned query embeddings to generate
        a sequence of action predictions conditioned on visual and latent features.

        Args:
            latent: Latent features
            memory: Encoded visual and proprioceptive memory

        Returns:
            torch.Tensor: Predicted action sequence [B, T, action_dim]
        """
        batch_size = latent.shape[0]

        # Convert to sequence first and expand
        query_embed = self.query_embed.expand(-1, batch_size, -1)  # [T, B, H]
        latent = latent.unsqueeze(0).expand_as(query_embed)  # [T, B, H]

        # Add latent to query embedding
        query_embed = query_embed + latent

        # Initialize target with zeros
        tgt = torch.zeros_like(query_embed)

        # Decode sequence
        hs = self.transformer["decoder"](tgt, memory, query_pos=query_embed)

        # Project to action space (keeping sequence first)
        actions = self.action_head(hs)  # [T, B, A]

        # Convert back to batch first
        actions = actions.transpose(0, 1)  # [B, T, A]

        return actions

    def _predict_action(
        self,
        mu: torch.FloatTensor,
        logvar: torch.FloatTensor,
        batch: BatchedInferenceInputs,
    ) -> torch.FloatTensor:
        """Predict action sequence from latent distribution and observations.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            batch: Input observations

        Returns:
            torch.FloatTensor: Predicted action sequence
        """
        # Sample latent
        latent_sample = self._reparametrize(mu, logvar)

        # Project latent
        latent = self.latent_out_proj(latent_sample)  # [B, H]

        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("No RGB images in batch")

        # Encode visual features
        proprio_state = self._combine_proprio(batch)
        memory = self._encode_visual(
            proprio_state,
            batch.inputs[DataType.RGB_IMAGES],
            batch.inputs_mask[DataType.RGB_IMAGES],
            latent,
        )

        # Decode actions
        action_preds = self._decode(latent, memory)
        return action_preds

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Perform inference to predict action sequence.

        Args:
            batch: Input batch with observations

        Returns:
            dict[DataType, list[BatchedNCData]]: Model predictions with action sequences
        """
        batch_size = len(batch)
        mu = torch.zeros(batch_size, self.latent_dim, device=self.device)
        logvar = torch.zeros(batch_size, self.latent_dim, device=self.device)
        action_preds = self._predict_action(mu, logvar, batch)

        # (B, T, action_dim)
        predictions = self.action_normalizer.unnormalize(action_preds)

        output_tensors: dict[DataType, list[BatchedNCData]] = {}

        start_slice_idx = 0
        for data_type in self.output_data_types:
            end_slice_idx = start_slice_idx + len(self.dataset_statistics[data_type])
            dt_preds = predictions[
                :, :, start_slice_idx:end_slice_idx
            ]  # (B, T, dt_size)

            if data_type == DataType.JOINT_TARGET_POSITIONS:
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

            start_slice_idx = end_slice_idx

        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Encodes action sequences to latent space, predicts actions, and computes
        L1 reconstruction loss plus KL divergence regularization.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        if DataType.JOINT_TARGET_POSITIONS not in batch.outputs:
            raise ValueError("Batch output joint target positions missing")

        inference_sample = BatchedInferenceInputs(
            inputs=batch.inputs,
            inputs_mask=batch.inputs_mask,
            batch_size=batch.batch_size,
        )

        # Extract target actions
        # Extract target actions
        action_targets = []
        for data_type in self.output_data_types:
            if data_type == DataType.JOINT_TARGET_POSITIONS:
                batched_joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend([bjd.value for bjd in batched_joints])
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData], batch.outputs[data_type]
                )
                action_targets.extend([gripper.open_amount for gripper in grippers])
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)  # (B, T, action_dim)

        # Get masks
        pred_sequence_mask = torch.ones_like(
            action_data[:, :, 0]
        )  # All time steps valid
        max_action_mask = torch.ones(
            batch.batch_size, self.max_output_size, device=self.device
        )  # All actions valid

        proprio_state = self._combine_proprio(inference_sample)

        # Normalize actions for encoding
        normalized_actions = self.action_normalizer.normalize(action_data)

        mu, logvar = self._encode_latent(
            proprio_state,
            normalized_actions,
            max_action_mask,
            pred_sequence_mask,
        )

        action_preds = self._predict_action(mu, logvar, inference_sample)
        target_actions = self.action_normalizer.normalize(action_data)

        l1_loss_all = F.l1_loss(action_preds, target_actions, reduction="none")
        l1_loss = (l1_loss_all * pred_sequence_mask.unsqueeze(-1)).mean()
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        loss = l1_loss + self.kl_weight * kl_loss

        losses = {
            "l1_and_kl_loss": loss,
        }
        metrics = {
            "l1_loss": l1_loss,
            "kl_loss": kl_loss,
        }

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
            DataType.RGB_IMAGES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        }

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        """Get the output data types supported by this model.

        Returns:
            set[DataType]: Set of supported output data types
        """
        return {DataType.JOINT_TARGET_POSITIONS, DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS}
