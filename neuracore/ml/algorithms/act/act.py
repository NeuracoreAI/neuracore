"""ACT: Action Chunking with Transformers."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from neuracore import (
    BatchedInferenceOutputs,
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    DatasetDescription,
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
    """
    Implementation of ACT (Action Chunking Transformer) model.
    """

    def __init__(
        self,
        dataset_description: DatasetDescription,
        hidden_dim: int = 512,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 7,
        nheads: int = 8,
        dim_feedforward: int = 3200,
        dropout: float = 0.1,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        kl_weight: float = 10.0,
        latent_dim: int = 512,
    ):
        super().__init__(dataset_description)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim

        # Vision components
        self.image_encoders = nn.ModuleList([
            ACTImageEncoder(output_dim=hidden_dim)
            for _ in range(self.dataset_description.max_num_cameras)
        ])
        # Input projections
        self.state_embed = nn.Linear(
            self.dataset_description.max_state_size, hidden_dim
        )
        self.action_embed = nn.Linear(
            self.dataset_description.max_action_size, hidden_dim
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
            hidden_dim, self.dataset_description.max_action_size
        )

        # Latent projections
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)

        # Query embedding for decoding
        self.query_embed = nn.Parameter(
            torch.randn(dataset_description.action_prediction_horizon, 1, hidden_dim)
        )

        # Additional position embeddings for proprio and latent
        self.additional_pos_embed = nn.Parameter(torch.randn(2, 1, hidden_dim))

        self.transform = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def _preprocess_states(self, states: torch.FloatTensor) -> torch.FloatTensor:
        """Preprocess the states."""
        return (
            states - self.dataset_description.state_mean
        ) / self.dataset_description.state_std

    def _preprocess_actions(self, actions: torch.FloatTensor) -> torch.FloatTensor:
        """Preprocess the actions."""
        return (
            actions - self.dataset_description.action_mean
        ) / self.dataset_description.action_std

    def _preprocess_camera_images(
        self, camera_images: torch.FloatTensor
    ) -> torch.FloatTensor:
        for cam_id in range(self.dataset_description.max_num_cameras):
            camera_images[:, cam_id] = self.transform(camera_images[:, cam_id])
        return camera_images

    def _inference_postprocess(
        self, output: BatchedInferenceOutputs
    ) -> BatchedInferenceOutputs:
        """Postprocess the output of the inference."""
        predictions = (
            output.action_predicitons * self.dataset_description.action_std
        ) + self.dataset_description.action_mean
        return BatchedInferenceOutputs(action_predicitons=predictions)

    def _reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparametrization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _encode_latent(
        self,
        state: torch.FloatTensor,
        state_mask: torch.FloatTensor,
        actions: torch.FloatTensor,
        actions_mask: torch.FloatTensor,
        actions_sequence_mask: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Encode actions to latent space during training."""
        batch_size = state.shape[0]

        # Project joint positions and actions
        state_embed = self.state_embed(state * state_mask)  # [B, H]
        action_embed = self.action_embed(
            actions * actions_mask.unsqueeze(1)
        )  # [B, T, H]

        # Reshape to sequence first
        state_embed = state_embed.unsqueeze(0)  # [1, B, H]
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
        states_mask: torch.FloatTensor,
        camera_images: torch.FloatTensor,
        camera_images_mask: torch.FloatTensor,
        latent: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Encode visual inputs with latent and proprioceptive features."""
        batch_size = states.shape[0]

        # Process images
        image_features = []
        image_pos = []
        for cam_id, encoder in enumerate(self.image_encoders):
            features, pos = encoder(
                camera_images[:, cam_id]
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
        state_features = self.state_embed(states * states_mask)  # [B, H]

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
        """Decode latent and visual features to action sequence."""
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
        # Sample latent
        latent_sample = self._reparametrize(mu, logvar)

        # Project latent
        latent = self.latent_out_proj(latent_sample)  # [B, H]

        # Encode visual features
        memory = self._encode_visual(
            batch.states,
            batch.states_mask,
            batch.camera_images,
            batch.camera_images_mask,
            latent,
        )

        # Decode actions
        action_preds = self._decode(latent, memory)
        return action_preds

    def forward(self, batch: BatchedInferenceSamples) -> BatchedInferenceOutputs:
        batch_size = batch.states.shape[0]
        mu = torch.zeros(batch_size, self.latent_dim, device=self.device)
        logvar = torch.zeros(batch_size, self.latent_dim, device=self.device)
        preprocessed_batch = BatchedInferenceSamples(
            states=self._preprocess_states(batch.states),
            states_mask=batch.states_mask,
            camera_images=self._preprocess_camera_images(batch.camera_images),
            camera_images_mask=batch.camera_images_mask,
        )
        action_preds = self._predict_action(mu, logvar, preprocessed_batch)
        return self._inference_postprocess(
            BatchedInferenceOutputs(action_predicitons=action_preds)
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step."""
        mu, logvar = self._encode_latent(
            batch.states,
            batch.states_mask,
            batch.actions,
            batch.actions_mask,
            batch.actions_sequence_mask,
        )
        preprocessed_batch = BatchedInferenceSamples(
            states=self._preprocess_states(batch.states),
            states_mask=batch.states_mask,
            camera_images=self._preprocess_camera_images(batch.camera_images),
            camera_images_mask=batch.camera_images_mask,
        )
        action_preds = self._predict_action(mu, logvar, preprocessed_batch)
        target_actions = self._preprocess_actions(batch.actions)

        l1_loss_all = F.l1_loss(action_preds, target_actions, reduction="none")
        l1_loss = (l1_loss_all * (1 - batch.actions_sequence_mask).unsqueeze(-1)).mean()
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
            action_predicitons=action_preds,
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer with different LRs for different components."""
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
