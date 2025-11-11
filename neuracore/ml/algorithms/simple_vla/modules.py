"""Neural network modules for Vision-Language-Action models.

This module provides the core building blocks for VLA architectures including
image encoders, language encoders, and multimodal fusion components. These
modules are designed to process and combine visual, textual, and proprioceptive
information for robot manipulation tasks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel


class ImageEncoder(nn.Module):
    """Encode images using ResNet backbone with projection layer.

    Uses a pretrained ResNet architecture (without the final classification
    layer) to extract visual features, followed by a linear projection to
    map to the desired output dimension.
    """

    def __init__(self, output_dim: int = 512, backbone: str = "resnet18"):
        """Initialize the image encoder.

        Args:
            output_dim: Desired output feature dimension
            backbone: ResNet architecture name (e.g., "resnet18", "resnet50")
        """
        super().__init__()
        # Use pretrained ResNet but remove final layer
        self.backbone = self._build_backbone(backbone)
        self.proj = nn.Linear(512, output_dim)

    def _build_backbone(self, backbone_name: str) -> nn.Module:
        """Build backbone CNN by removing classification layers.

        Args:
            backbone_name: Name of the ResNet architecture to use

        Returns:
            nn.Module: ResNet backbone without final classification layers
        """
        resnet = models.get_model(backbone_name, weights="DEFAULT")
        return nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through image encoder.

        Processes input images through the ResNet backbone, flattens the
        spatial dimensions, and projects to the desired output dimension.

        Args:
            x: Image tensor of shape (batch, channels, height, width)

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        batch = x.shape[0]
        x = self.backbone(x)
        x = x.view(batch, -1)
        return self.proj(x)


class LanguageEncoder(nn.Module):
    """Encode language tokens using a pretrained language model.

    Uses a frozen pretrained language model (e.g., BERT, DistilBERT) to
    extract semantic features from text instructions, followed by a
    projection layer to map to the desired output dimension.
    """

    def __init__(self, output_dim: int = 512, model_name: str = "bert-base-uncased"):
        """Initialize the language encoder.

        Args:
            output_dim: Desired output feature dimension
            model_name: Name of the pretrained language model to use
        """
        super().__init__()
        # Load pretrained language model
        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Get the output dimension of the backbone
        backbone_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(backbone_dim, output_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through language encoder.

        Processes tokenized text through the pretrained language model
        and extracts the [CLS] token representation as the sequence-level
        feature, then projects to the desired output dimension.

        Args:
            input_ids: Token ids of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        # Get outputs from the language model
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation as the sequence representation
        sequence_output = outputs.last_hidden_state[:, 0, :].detach()
        return self.proj(sequence_output)


class MultimodalFusion(nn.Module):
    """Fuse vision, language, and state features for multimodal understanding.

    Combines visual features, language features, and proprioceptive state
    information through concatenation followed by multi-layer perceptron
    processing with normalization and dropout for robust feature fusion.
    """

    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        state_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        """Initialize the multimodal fusion module.

        Args:
            vision_dim: Dimension of visual features
            language_dim: Dimension of language features
            state_dim: Dimension of state features
            hidden_dim: Hidden layer dimension for fusion network
            output_dim: Output dimension for fused features
        """
        super().__init__()
        # Create fusion layer
        total_input_dim = vision_dim + language_dim + state_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        state_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Fuse multimodal features through concatenation and MLP processing.

        Concatenates visual, language, and optional state features, then
        processes the combined representation through a multi-layer network
        with normalization and dropout for robust feature fusion.

        Args:
            vision_features: Vision features of shape (batch, vision_dim)
            language_features: Language features of shape (batch, language_dim)
            state_features: Optional state features of shape (batch, state_dim)

        Returns:
            torch.Tensor: Fused features of shape (batch, output_dim)
        """
        # Concatenate features
        features = [vision_features, language_features]
        if state_features is not None:
            features.append(state_features)

        concat_features = torch.cat(features, dim=1)
        return self.fusion(concat_features)
