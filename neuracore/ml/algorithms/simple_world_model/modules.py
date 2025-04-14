import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ImageEncoder(nn.Module):
    """Encode images using ResNet backbone."""

    def __init__(self, output_dim: int = 512, backbone: str = "resnet18"):
        super().__init__()
        # Use pretrained ResNet but remove final layer
        self.backbone = self._build_backbone(backbone)
        self.proj = nn.Linear(512, output_dim)

    def _build_backbone(self, backbone_name: str) -> nn.Module:
        """Build backbone CNN."""
        resnet = getattr(models, backbone_name)(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through image encoder.
        Args:
            x: Image tensor of shape (batch, channels, height, width)
        Returns:
            Encoded features of shape (batch, output_dim)
        """
        batch = x.shape[0]
        x = self.backbone(x)
        x = x.view(batch, -1)
        return self.proj(x)


class DoubleConv(nn.Module):
    """Double convolution block used in UNet."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
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

    def forward(self, x1, x2):
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
    """Final convolution layer."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """UNet architecture for image prediction with conditioning."""

    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        feature_map_sizes=None,
        condition_dim=512,
        bilinear=True,
    ):
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

    def forward(self, x, condition):
        """
        Forward pass of UNet.

        Args:
            x: Input image tensor of shape (batch, channels, height, width)
            condition: Condition tensor of shape (batch, condition_dim)

        Returns:
            Predicted output image
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
