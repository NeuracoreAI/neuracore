import torch
import torch.nn as nn
import torchvision.transforms as T

from neuracore import BatchInput, BatchOutput, DatasetDescription, NeuracoreModel
from neuracore.ml.algorithms.cnnmlp.modules import ImageEncoder


class CNNMLP(NeuracoreModel):
    """CNN+MLP model with single timestep input and sequence output."""

    def __init__(
        self,
        dataset_description: DatasetDescription,
        hidden_dim: int = 512,
        cnn_output_dim: int = 64,
        num_layers: int = 3,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        super().__init__(dataset_description)
        self.hidden_dim = hidden_dim
        self.cnn_output_dim = cnn_output_dim
        self.num_layers = num_layers
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

        self.image_encoders = nn.ModuleList([
            ImageEncoder(output_dim=self.cnn_output_dim)
            for _ in range(self.dataset_description.max_num_cameras)
        ])

        self.state_embed = nn.Linear(
            self.dataset_description.max_state_size, hidden_dim
        )

        mlp_input_dim = (
            self.dataset_description.max_num_cameras * cnn_output_dim + hidden_dim
        )

        # Predict entire sequence at once
        self.action_output_size = (
            self.dataset_description.max_action_size
            * self.dataset_description.action_prediction_horizon
        )
        self.mlp = self.build_mlp(
            input_dim=mlp_input_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_output_size,
            num_layers=num_layers,
        )

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.action_prediction_horizon = (
            self.dataset_description.action_prediction_horizon
        )
        self.max_action_size = self.dataset_description.max_action_size

    def build_mlp(
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

    def forward(self, batch: BatchInput) -> BatchOutput:
        """Forward pass of the model."""

        batch_size = batch.states.shape[0]

        # Process images from each camera
        image_features = []
        for cam_id, encoder in enumerate(self.image_encoders):
            features = encoder(batch.camera_images[:, cam_id])
            features *= batch.camera_images_mask[:, cam_id : cam_id + 1]
            image_features.append(features)

        # Combine image features
        if image_features:
            combined_image_features = torch.cat(image_features, dim=-1)
        else:
            combined_image_features = torch.zeros(
                batch_size, self.cnn_output_dim, device=self.device
            )

        state_features = self.state_embed(batch.states)

        # Combine all features
        combined_features = torch.cat([state_features, combined_image_features], dim=-1)

        # Forward through MLP to get entire sequence
        mlp_out = self.mlp(combined_features)

        action_preds = mlp_out.view(
            batch_size, self.action_prediction_horizon, self.max_action_size
        )

        losses = {}
        metrics = {}
        if self.training:
            loss = nn.functional.mse_loss(action_preds, batch.actions)
            losses["mse_loss"] = loss

        return BatchOutput(
            action_predicitons=action_preds, losses=losses, metrics=metrics
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
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
        return torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

    def process_batch(self, batch: BatchInput) -> BatchInput:
        """Called by dataloader to process the batch."""
        camera_images = batch.camera_images
        for cam_id in range(self.dataset_description.max_num_cameras):
            camera_images[:, cam_id] = self.transform(batch.camera_images[:, cam_id])
        states = (
            batch.states - self.dataset_description.state_mean
        ) / self.dataset_description.state_std
        actions = (
            batch.actions - self.dataset_description.action_mean
        ) / self.dataset_description.action_std
        return BatchInput(
            states=states,
            states_mask=batch.states_mask,
            camera_images=camera_images,
            camera_images_mask=batch.camera_images_mask,
            actions=actions,
            actions_mask=batch.actions_mask,
            actions_sequence_mask=batch.actions_sequence_mask,
        )
