import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ACTImageEncoder(nn.Module):
    """Encode images using ResNet backbone.

    Maintaining spatial dimensions and providing position embeddings.
    Similar to DETR's backbone implementation.
    """

    def __init__(self, output_dim: int = 256):  # Changed default to 256
        super().__init__()
        # Use pretrained ResNet but remove final layers
        self.backbone = self._build_backbone()
        self.proj = nn.Conv2d(512, output_dim, kernel_size=1)  # Project to output_dim

        # Position embeddings should match output_dim
        self.row_embed = nn.Embedding(50, output_dim // 2)  # Half size
        self.col_embed = nn.Embedding(50, output_dim // 2)  # Half size
        self.reset_parameters()

    def _build_backbone(self) -> nn.Module:
        """Build backbone CNN, removing avgpool and fc layers."""
        resnet = getattr(models, "resnet18")(pretrained=True)
        return nn.Sequential(*list(resnet.children())[:-2])

    def reset_parameters(self):
        """Initialize position embeddings."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through image encoder.
        Args:
            x: Image tensor of shape (batch, channels, height, width)
        Returns:
            features: Encoded features of shape (batch, output_dim, height, width)
            pos: Position embeddings of shape (batch, output_dim, height, width)
        """
        # Extract features
        x = self.backbone(x)
        features = self.proj(x)  # Now [B, output_dim, H, W]

        # Create position embeddings
        h, w = features.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)  # [W, output_dim//2]
        y_emb = self.row_embed(j)  # [H, output_dim//2]

        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )  # [1, output_dim, H, W]

        pos = pos.repeat(x.shape[0], 1, 1, 1)  # [B, output_dim, H, W]

        return features, pos


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)





