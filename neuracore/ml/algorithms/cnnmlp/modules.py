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

    OUTPUT_CNN_W_H = 7 * 7

    def __init__(self, output_dim: int = 512, backbone: str = "resnet18"):
        """Initialize the image encoder.

        Args:
            output_dim: Desired output feature dimension
            backbone: ResNet architecture name (e.g., "resnet18", "resnet50")
        """
        super().__init__()
        # Use pretrained ResNet but remove final layer
        self.backbone, fc_input_dim = self._build_backbone(backbone)
        self.proj_2d = nn.Conv2d(fc_input_dim, output_dim, kernel_size=1)
        self.linear = nn.Linear(ImageEncoder.OUTPUT_CNN_W_H * output_dim, output_dim)

    def _build_backbone(self, backbone_name: str) -> tuple[nn.Module, int]:
        """Build backbone CNN by removing classification layers.

        Args:
            backbone_name: Name of the ResNet architecture to use

        Returns:
            nn.Module: ResNet backbone without final classification layers
            int: Input dimension of the final fully connected layer
        """
        resnet = models.get_model(backbone_name, weights="DEFAULT")
        return nn.Sequential(*list(resnet.children())[:-2]), resnet.fc.in_features

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
        x = self.proj_2d(x).view(batch, -1)
        return self.linear(x)
