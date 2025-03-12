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


class TransformerEncoderLayer(nn.Module):
    """Implement a single encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class TransformerDecoderLayer(nn.Module):
    """Implement a single decoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            q, k, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2, _ = self.multihead_attn(
            query=tgt2 if query_pos is None else tgt2 + query_pos,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class TransformerEncoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        return self.norm(output)


class TransformerDecoder(nn.Module):
    """Stack of N decoder layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                query_pos=query_pos,
            )

        return self.norm(output)
