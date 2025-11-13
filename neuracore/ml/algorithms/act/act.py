"""ACT: Action Chunking with Transformers implementation.

This module implements the ACT (Action Chunking with Transformers) model
from "Learning fine-grained bimanual manipulation with low-cost hardware"
(Zhao et al., 2023). ACT uses a transformer architecture with latent variable
modeling to predict action sequences for robot manipulation tasks.

Reference: Zhao, Tony Z., et al. "Learning fine-grained bimanual manipulation
with low-cost hardware." arXiv preprint arXiv:2304.13705 (2023).
"""

import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from neuracore_types import DataType, ModelInitDescription, ModelPrediction

from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import (
    ACTImageEncoder,
    PositionalEncoding,
    TransformerDecoder,
    TransformerEncoder,
)

logger = logging.getLogger(__name__)


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
        lr: float = 1e-4,
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
            lr: Learning rate for main parameters
            lr_backbone: Learning rate for image encoder backbone
            weight_decay: Weight decay for optimizer
            kl_weight: Weight for KL divergence loss
            latent_dim: Dimension of latent variable space
        """
        super().__init__(model_init_description)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim

        # Vision components
        self.image_encoders = nn.ModuleList([
            ACTImageEncoder(output_dim=hidden_dim)
            for _ in range(self.dataset_description.rgb_images.max_len)
        ])

        state_input_dim = (
            self.dataset_description.joint_positions.max_len
            + self.dataset_description.joint_velocities.max_len
            + self.dataset_description.joint_torques.max_len
        )
        self.state_embed = None
        if state_input_dim > 0:
            self.state_embed = nn.Linear(state_input_dim, hidden_dim)

        self.action_embed = nn.Linear(
            self.dataset_description.joint_target_positions.max_len, hidden_dim
        )

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
        self.action_head = nn.Linear(
            hidden_dim, self.dataset_description.joint_target_positions.max_len
        )

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

        self.transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        return (joint_state - self.joint_state_mean) / self.joint_state_std

    def _preprocess_target_joint_pos(
        self, target_joint_pos: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Normalize target joint positions using dataset statistics.

        Args:
            target_joint_pos: Raw target joint positions

        Returns:
            torch.FloatTensor: Normalized target joint positions
        """
        return (target_joint_pos - self.joint_target_mean) / self.joint_target_std

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

        # Update padding mask
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
        camera_images: torch.FloatTensor,
        camera_images_mask: torch.FloatTensor,
        latent: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Encode visual inputs with latent and proprioceptive features.

        Processes RGB images through vision encoders and combines them with
        proprioceptive state and latent features using a transformer encoder.

        Args:
            states: Proprioceptive state features
            camera_images: RGB camera images [B, num_cameras, C, H, W]
            camera_images_mask: Mask for valid camera inputs
            latent: Latent features from action encoding

        Returns:
            torch.FloatTensor: Encoded visual and proprioceptive memory
        """
        batch_size = states.shape[0]

        # Process images
        image_features = []
        image_pos = []
        for cam_id, encoder in enumerate(self.image_encoders):
            features, pos = encoder(
                self.transform(camera_images[:, cam_id])
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
        batch: BatchedInferenceSamples,
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

        if batch.rgb_images is not None:
            # Encode visual features
            memory = self._encode_visual(
                self._combine_joint_states(batch),
                batch.rgb_images.data,
                batch.rgb_images.mask,
                latent,
            )

            # Decode actions
            action_preds = self._decode(latent, memory)
            return action_preds
        raise ValueError("No batch rbg_images")

    def _combine_joint_states(
        self, batch: BatchedInferenceSamples
    ) -> torch.FloatTensor:
        """Combine different types of joint state data.

        Concatenates joint positions, velocities, and torques into a single
        feature vector, applying masks and normalization.

        Args:
            batch: Input batch containing joint state data

        Returns:
            torch.FloatTensor: Combined and normalized joint state features
        """
        joint_states = None
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
        return joint_states

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Perform inference to predict action sequence.

        Args:
            batch: Input batch with observations

        Returns:
            ModelPrediction: Model predictions with timing information
        """
        t = time.time()
        batch_size = len(batch)
        mu = torch.zeros(batch_size, self.latent_dim, device=self.device)
        logvar = torch.zeros(batch_size, self.latent_dim, device=self.device)
        action_preds = self._predict_action(mu, logvar, batch)
        prediction_time = time.time() - t
        predictions = (action_preds * self.joint_target_std) + self.joint_target_mean
        predictions = predictions.detach().cpu().numpy()
        return ModelPrediction(
            outputs={DataType.JOINT_TARGET_POSITIONS: predictions},
            prediction_time=prediction_time,
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Encodes action sequences to latent space, predicts actions, and computes
        L1 reconstruction loss plus KL divergence regularization.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        if batch.outputs.joint_target_positions is None:
            raise ValueError("Batch output joint target positions is None")

        pred_sequence_mask = batch.outputs.joint_target_positions.mask[
            :, :, 0
        ]  # [batch_size, T]
        max_action_mask = batch.outputs.joint_target_positions.mask[
            :, 0, :
        ]  # [batch_size, MaxActionSize]
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
        )
        joint_states = self._combine_joint_states(inference_sample)
        mu, logvar = self._encode_latent(
            joint_states,
            batch.outputs.joint_target_positions.data,
            max_action_mask,
            pred_sequence_mask,
        )
        action_preds = self._predict_action(mu, logvar, inference_sample)
        target_actions = self._preprocess_target_joint_pos(
            batch.outputs.joint_target_positions.data
        )

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
            output_predictions=action_preds,
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer with different learning rates for different components.

        Uses separate learning rates for image encoder backbone (typically lower)
        and other model parameters to account for pre-trained vision components.

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
        return [DataType.JOINT_TARGET_POSITIONS]
