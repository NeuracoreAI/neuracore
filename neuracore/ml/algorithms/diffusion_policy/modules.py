"""Diffusion Policy model components including UNet, encoders, and utilities."""

import math
from typing import Any, Dict, Optional, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DiffusionPolicyImageEncoder(nn.Module):
    """Encode images using ResNet backbone.

    Maintaining spatial dimensions and providing position embeddings.
    Similar to DETR's backbone implementation.
    """

    def __init__(
        self,
        spatial_softmax_num_keypoints: int = 32,
        feature_dim: int = 256,
    ):
        """Initialize the image encoder.

        Args:
            spatial_softmax_num_keypoints: Number of keypoints for spatial softmax.
            feature_dim: Dimension of output features.
        """
        super().__init__()

        # Use pretrained ResNet but remove final layers
        self.backbone = self._build_backbone()
        # ResNet18 without avgpool and fc layers outputs (512, 7, 7) for 224x224 input
        self.pool = SpatialSoftmax((512, 7, 7), num_kp=spatial_softmax_num_keypoints)
        self.feature_dim = feature_dim
        self.out = nn.Linear(spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def _build_backbone(self) -> nn.Module:
        """Build backbone CNN, removing avgpool and fc layers."""
        resnet = models.get_model("resnet18", weights="DEFAULT")
        return nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through image encoder.

        Args:
            x: Image tensor of shape (batch, channels, height, width).

        Returns:
            features: Encoded features of shape (batch, feature_dim).
        """
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax operation for visuomotor learning.

    Described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113).

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns
    the "center of mass" of activations of each channel, i.e., keypoints in the
    image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of
    normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations
    (512x120) and computing the dot product with the coordinates (120x2) to
    get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input
    channels).We can optionally provide num_kp != None to control the number of
    keypoints. This is achieved by a first applying a learnable linear mapping
    (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(
        self, input_shape: Tuple[int, int, int], num_kp: Optional[int] = None
    ) -> None:
        """Initialize SpatialSoftmax layer.

        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have
                the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # We could use torch.linspace directly but that seems to behave slightly
        # differently than numpy and causes a small degradation in pc_success of
        # pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass through SpatialSoftmax.

        Args:
            features: (B, C, H, W) input feature maps.

        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean
        # in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original
    diffusion policy code.
    """

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 128,
        use_film_scale_modulation: bool = True,
    ):
        """Initialize the 1D UNet.

        Args:
            action_dim: Dimension of action space.
            global_cond_dim: Dimension of global conditioning features.
            down_dims: Downsampling dimensions for encoder.
            kernel_size: Kernel size for convolutions.
            n_groups: Number of groups for group normalization.
            diffusion_step_embed_dim: Dimension of diffusion step embeddings.
            use_film_scale_modulation: Whether to use FiLM scale modulation.
        """
        super().__init__()

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's
        # encoder. For the decoder, we just reverse these.
        in_out = [(action_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))

        # Unet encoder.
        common_res_block_kwargs: Dict[str, Any] = {
            "cond_dim": cond_dim,
            "kernel_size": kernel_size,
            "n_groups": n_groups,
            "use_film_scale_modulation": use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    DiffusionConditionalResidualBlock1d(
                        dim_in, dim_out, **common_res_block_kwargs
                    ),
                    DiffusionConditionalResidualBlock1d(
                        dim_out, dim_out, **common_res_block_kwargs
                    ),
                    # Downsample as long as it is not the last block.
                    (
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1)
                        if not is_last
                        else nn.Identity()
                    ),
                ])
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(
                down_dims[-1], down_dims[-1], **common_res_block_kwargs
            ),
            DiffusionConditionalResidualBlock1d(
                down_dims[-1], down_dims[-1], **common_res_block_kwargs
            ),
        ])

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    # dim_in * 2, because it takes the encoder's skip connection
                    # as well
                    DiffusionConditionalResidualBlock1d(
                        dim_in * 2, dim_out, **common_res_block_kwargs
                    ),
                    DiffusionConditionalResidualBlock1d(
                        dim_out, dim_out, **common_res_block_kwargs
                    ),
                    # Upsample as long as it is not the last block.
                    (
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1)
                        if not is_last
                        else nn.Identity()
                    ),
                ])
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the UNet.

        Args:
            x: (batch, horizon, input_dim) tensor for input to the Unet.
            timestep: (batch,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (batch, global_cond_dim).

        Returns:
            (batch, horizon, input_dim) diffusion model prediction.
        """
        # Store the original horizon for projection head
        original_horizon = x.shape[1]

        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the
        # timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[torch.Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            skip_feature = encoder_skip_features.pop()

            # Handle size mismatch between upsampled x and skip connection
            if x.shape[-1] != skip_feature.shape[-1]:
                # Interpolate x to match skip feature size
                x = F.interpolate(
                    x, size=skip_feature.shape[-1], mode="linear", align_corners=False
                )

            x = torch.cat((x, skip_feature), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # Rearrange back to (batch, time, features) format
        x = einops.rearrange(x, "b d t -> b t d")

        # Projection head: ensure output horizon matches input horizon
        current_horizon = x.shape[1]
        if current_horizon != original_horizon:
            # Use interpolation to resize the sequence to match original horizon
            # Rearrange to (batch, features, time) for interpolation
            x = einops.rearrange(x, "b t d -> b d t")
            x = F.interpolate(
                x, size=original_horizon, mode="linear", align_corners=False
            )
            # Rearrange back to (batch, time, features)
            x = einops.rearrange(x, "b d t -> b t d")

        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation
        # (defaults to False meaning FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        """Initialize the conditional residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            cond_dim: Dimension of conditioning features.
            kernel_size: Kernel size for convolutions.
            n_groups: Number of groups for group normalization.
            use_film_scale_modulation: Whether to use FiLM scale modulation.
        """
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(
            in_channels, out_channels, kernel_size, n_groups=n_groups
        )

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel
        # bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(
            out_channels, out_channels, kernel_size, n_groups=n_groups
        )

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conditional residual block.

        Args:
            x: (B, in_channels, T).
            cond: (B, cond_dim).

        Returns:
            (B, out_channels, T).
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting
        # in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish."""

    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ):
        """Initialize the conv1d block.

        Args:
            inp_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for convolution.
            n_groups: Number of groups for group normalization.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the conv1d block."""
        return self.block(x)


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        """Initialize sinusoidal position embeddings.

        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sinusoidal embeddings."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
