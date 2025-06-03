"""Image encoder module using pretrained ResNet backbone.

This module provides a simple image encoder that uses pretrained ResNet
architectures for feature extraction, followed by a projection layer to
map features to a desired output dimension.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """Encode images using ResNet backbone with projection layer.

    Uses a pretrained ResNet architecture (without the final classification
    layer) to extract visual features, followed by a linear projection to
    map to the desired output dimension. Suitable for various computer
    vision tasks requiring fixed-size feature representations.
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
        resnet = getattr(models, backbone_name)(pretrained=True)
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
