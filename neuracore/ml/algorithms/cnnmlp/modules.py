import torch
import torch.nn as nn
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
