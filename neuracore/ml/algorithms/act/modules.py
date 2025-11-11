"""Neural network modules for the ACT (Action Chunking Transformer) model.

This module provides the core building blocks for the ACT architecture including
image encoders, positional encodings, and transformer encoder/decoder layers.
These components are designed to work together for robot manipulation tasks.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ACTImageEncoder(nn.Module):
    """Encode images using ResNet backbone with spatial position embeddings.

    Maintains spatial dimensions and provides learnable position embeddings
    similar to DETR's backbone implementation. Uses a pretrained ResNet18
    backbone with projection layers for feature extraction.
    """

    def __init__(self, output_dim: int = 256):
        """Initialize the image encoder.

        Args:
            output_dim: Output feature dimension after projection
        """
        super().__init__()
        # Use pretrained ResNet but remove final layers
        self.backbone = self._build_backbone()
        self.proj = nn.Conv2d(512, output_dim, kernel_size=1)  # Project to output_dim

        # Position embeddings should match output_dim
        self.row_embed = nn.Embedding(50, output_dim // 2)  # Half size
        self.col_embed = nn.Embedding(50, output_dim // 2)  # Half size
        self.reset_parameters()

    def _build_backbone(self) -> nn.Module:
        """Build backbone CNN, removing avgpool and fc layers.

        Returns:
            nn.Module: ResNet18 backbone without classification layers
        """
        resnet = models.get_model("resnet18", weights="DEFAULT")
        return nn.Sequential(*list(resnet.children())[:-2])

    def reset_parameters(self) -> None:
        """Initialize position embeddings with uniform distribution."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through image encoder.

        Args:
            x: Image tensor of shape (batch, channels, height, width)

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - features: Encoded features of shape (batch, output_dim, height, width)
                - pos: Position embeddings of shape (batch, output_dim, height, width)
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
    """Sinusoidal positional encoding for transformer sequences.

    Implements the standard sinusoidal positional encoding as described
    in "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length to support
        """
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
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (seq_len, batch, d_model)

        Returns:
            torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-layer normalization.

    Implements a transformer encoder layer following the pre-norm architecture
    with multi-head self-attention and position-wise feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize transformer encoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network hidden dimension
            dropout: Dropout probability
        """
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
        """Forward pass through encoder layer.

        Args:
            src: Input tensor of shape (seq_len, batch, d_model)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
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
    """Single transformer decoder layer with cross-attention.

    Implements a transformer decoder layer with masked self-attention,
    cross-attention to encoder memory, and position-wise feed-forward network.
    Supports query position embeddings for object detection style architectures.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize transformer decoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward network hidden dimension
            dropout: Dropout probability
        """
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
        """Forward pass through decoder layer.

        Args:
            tgt: Target tensor of shape (tgt_len, batch, d_model)
            memory: Memory tensor from encoder of shape (src_len, batch, d_model)
            tgt_mask: Optional target attention mask
            memory_mask: Optional memory attention mask
            tgt_key_padding_mask: Optional target key padding mask
            memory_key_padding_mask: Optional memory key padding mask
            query_pos: Optional query position embeddings

        Returns:
            torch.Tensor: Output tensor of same shape as target
        """
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
    """Stack of transformer encoder layers.

    Composes multiple TransformerEncoderLayer modules with a final layer
    normalization for encoding sequential input data.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize transformer encoder.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Feedforward network hidden dimension
            dropout: Dropout probability
        """
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
        """Forward pass through all encoder layers.

        Args:
            src: Input tensor of shape (seq_len, batch, d_model)
            mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask

        Returns:
            torch.Tensor: Encoded output tensor
        """
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        return self.norm(output)


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers.

    Composes multiple TransformerDecoderLayer modules with a final layer
    normalization for generating sequential output conditioned on memory.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        """Initialize transformer decoder.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward network hidden dimension
            dropout: Dropout probability
        """
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
        """Forward pass through all decoder layers.

        Args:
            tgt: Target tensor of shape (tgt_len, batch, d_model)
            memory: Memory tensor from encoder of shape (src_len, batch, d_model)
            tgt_mask: Optional target attention mask
            memory_mask: Optional memory attention mask
            tgt_key_padding_mask: Optional target key padding mask
            memory_key_padding_mask: Optional memory key padding mask
            query_pos: Optional query position embeddings

        Returns:
            torch.Tensor: Decoded output tensor
        """
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
