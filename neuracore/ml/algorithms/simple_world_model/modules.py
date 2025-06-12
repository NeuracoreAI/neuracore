"""Neural network modules for world model architectures.

This module provides building blocks for world models including image encoders
and U-Net architectures for image prediction. The U-Net implementation supports
conditioning on external features for generating future visual states.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net architecture.

    Applies two consecutive convolution operations with batch normalization
    and ReLU activation. This is the fundamental building block of U-Net.
    """

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None
    ):
        """Initialize double convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels. If None, uses out_channels
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output after double convolution
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block for U-Net encoder path.

    Applies max pooling for spatial downsampling followed by
    double convolution to increase feature depth.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through downsampling block.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Downsampled and processed tensor
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block for U-Net decoder path.

    Applies upsampling (either bilinear or transpose convolution) followed
    by concatenation with skip connection and double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        """Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bilinear: If True, use bilinear upsampling; otherwise use transpose conv
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampling block.

        Args:
            x1: Input tensor from previous layer
            x2: Skip connection tensor from encoder path

        Returns:
            torch.Tensor: Upsampled and processed tensor
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final convolution layer for U-Net output.

    Applies a 1x1 convolution to map from feature space to the
    desired number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize output convolution layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through output convolution.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor with desired number of channels
        """
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for image prediction with external conditioning.

    Implements a U-Net with encoder-decoder structure and skip connections,
    enhanced with the ability to condition on external features (e.g., robot
    state, actions). The conditioning is injected at the bottleneck layer.

    This architecture is particularly suitable for image-to-image translation
    tasks in world modeling where future images depend on current observations
    and planned actions.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        feature_map_sizes: Optional[List] = None,
        condition_dim: int = 512,
        bilinear: bool = True,
    ):
        """Initialize U-Net with conditioning support.

        Args:
            input_channels: Number of input image channels (e.g., 3 for RGB)
            output_channels: Number of output image channels
            feature_map_sizes: List of feature map sizes for each level
            condition_dim: Dimension of conditioning features
            bilinear: If True, use bilinear upsampling; otherwise use transpose conv
        """
        super().__init__()
        if feature_map_sizes is None:
            feature_map_sizes = [64, 128, 256, 512]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear
        self.condition_dim = condition_dim

        # Conditioning network
        cond_dim_output = 64
        self.condition_processor = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, cond_dim_output),
        )

        # Initial convolution
        self.inc = DoubleConv(input_channels, feature_map_sizes[0])

        # Downsampling path
        self.down1 = Down(feature_map_sizes[0], feature_map_sizes[1])
        self.down2 = Down(feature_map_sizes[1], feature_map_sizes[2])
        self.down3 = Down(feature_map_sizes[2], feature_map_sizes[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(feature_map_sizes[3], feature_map_sizes[3] * 2 // factor)

        # Upsampling path
        self.up1 = Up(
            cond_dim_output + feature_map_sizes[3] * 2,
            feature_map_sizes[2] * 2 // factor,
            bilinear,
        )
        self.up2 = Up(
            feature_map_sizes[2] * 2, feature_map_sizes[1] * 2 // factor, bilinear
        )
        self.up3 = Up(
            feature_map_sizes[1] * 2, feature_map_sizes[0] * 2 // factor, bilinear
        )
        self.up4 = Up(feature_map_sizes[0] * 2, feature_map_sizes[0], bilinear)

        # Final convolution for output
        self.outc = OutConv(feature_map_sizes[0], output_channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass of conditioned U-Net.

        Processes input image through encoder path, injects conditioning
        features at the bottleneck, then generates output through decoder
        path with skip connections.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)
            condition: Condition tensor of shape (batch, condition_dim)

        Returns:
            torch.Tensor: Predicted output image with sigmoid activation
        """
        # Process conditions
        cond_features = self.condition_processor(condition)  # [batch, 64]

        # Down path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Inject condition at the bottleneck
        batch_size = x5.shape[0]
        h, w = x5.shape[2], x5.shape[3]
        cond_features = cond_features.view(batch_size, -1, 1, 1).expand(-1, -1, h, w)
        x5 = torch.cat([x5, cond_features], dim=1)

        # Up path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final convolution and activation
        x = self.outc(x)

        return F.sigmoid(x)
