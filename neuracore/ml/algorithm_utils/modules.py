"""Multimodal encoders for different robot sensor data types.

This module provides encoder architectures for various robot sensor modalities
including depth images, point clouds, poses, end-effectors, and custom data types.
All encoders output features in a consistent format for multimodal fusion.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DepthImageEncoder(nn.Module):
    """Encoder for depth images using CNN backbone.

    Processes single-channel depth images through a convolutional neural network
    to extract spatial features for robotic perception tasks.
    """

    def __init__(self, output_dim: int = 256, backbone: str = "resnet18"):
        """Initialize the depth image encoder.

        Args:
            output_dim: Desired output feature dimension
            backbone: CNN backbone architecture name
        """
        super().__init__()
        # Modify ResNet for single-channel input
        resnet = models.get_model(backbone, weights="DEFAULT")

        # Replace first conv layer for single-channel depth input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove classification layers
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through depth encoder.

        Args:
            x: Depth images of shape (batch, 1, height, width)

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = x.view(batch_size, -1)
        return self.proj(x)


class PointCloudEncoder(nn.Module):
    """Encoder for 3D point clouds using PointNet-style architecture.

    Processes unordered 3D point sets through permutation-invariant networks
    to extract global features for robotic manipulation and perception.
    """

    def __init__(self, output_dim: int = 256, input_dim: int = 3):
        """Initialize the point cloud encoder.

        Args:
            output_dim: Desired output feature dimension
            input_dim: Input point dimension (3 for xyz, more for features)
        """
        super().__init__()
        self.input_dim = input_dim

        # Point-wise MLPs
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Final projection
        self.proj = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through point cloud encoder.

        Args:
            x: Point clouds of shape (batch, num_points, input_dim)

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        # Transpose for conv1d: (batch, input_dim, num_points)
        x = x.transpose(2, 1)

        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Global max pooling
        x = torch.max(x, 2)[0]  # (batch, 256)

        # Final projection
        x = self.dropout(x)
        return self.proj(x)


class PoseEncoder(nn.Module):
    """Encoder for 6DOF pose data.

    Processes pose information (position + orientation) through MLPs with
    appropriate handling of rotational representations for robotic tasks.
    """

    def __init__(self, output_dim: int = 256, max_poses: int = 10):
        """Initialize the pose encoder.

        Args:
            output_dim: Desired output feature dimension
            max_poses: Maximum number of poses to handle
        """
        super().__init__()
        self.max_poses = max_poses

        # Each pose is 6DOF (x, y, z, rx, ry, rz)
        pose_dim = 6

        # Pose-wise encoding
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Global pose set encoding
        self.global_encoder = nn.Sequential(
            nn.Linear(128 * max_poses, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pose encoder.

        Args:
            x: Pose data of shape (batch, max_poses * 6) - flattened poses

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        batch_size = x.shape[0]

        # Reshape to individual poses: (batch, max_poses, 6)
        x = x.view(batch_size, self.max_poses, 6)

        # Encode each pose individually
        pose_features = []
        for i in range(self.max_poses):
            pose_feat = self.pose_encoder(x[:, i, :])  # (batch, 128)
            pose_features.append(pose_feat)

        # Concatenate all pose features
        x = torch.cat(pose_features, dim=1)  # (batch, 128 * max_poses)

        # Global encoding
        return self.global_encoder(x)


class EndEffectorEncoder(nn.Module):
    """Encoder for end-effector state data.

    Processes gripper and tool state information through MLPs to provide
    end-effector state features for manipulation tasks.
    """

    def __init__(self, output_dim: int = 256, max_effectors: int = 2):
        """Initialize the end-effector encoder.

        Args:
            output_dim: Desired output feature dimension
            max_effectors: Maximum number of end-effectors to handle
        """
        super().__init__()
        self.max_effectors = max_effectors

        # End-effector state encoding (e.g., gripper openness, tool activation)
        self.effector_encoder = nn.Sequential(
            nn.Linear(1, 32),  # Assuming 1D state per effector (e.g., gripper openness)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Global effector encoding
        self.global_encoder = nn.Sequential(
            nn.Linear(64 * max_effectors, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through end-effector encoder.

        Args:
            x: End-effector data of shape (batch, max_effectors)

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        x.shape[0]

        # Encode each effector individually
        effector_features = []
        for i in range(self.max_effectors):
            effector_state = x[:, i : i + 1]  # (batch, 1)
            effector_feat = self.effector_encoder(effector_state)  # (batch, 64)
            effector_features.append(effector_feat)

        # Concatenate all effector features
        x = torch.cat(effector_features, dim=1)  # (batch, 64 * max_effectors)

        # Global encoding
        return self.global_encoder(x)


class CustomDataEncoder(nn.Module):
    """Flexible encoder for custom sensor data types.

    Provides a configurable architecture for encoding arbitrary sensor data
    with adaptive input handling for different data dimensions and types.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        hidden_dims: Optional[list[int]] = None,
    ):
        """Initialize the custom data encoder.

        Args:
            input_dim: Input feature dimension
            output_dim: Desired output feature dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 512, 256]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through custom data encoder.

        Args:
            x: Custom data of shape (batch, input_dim)

        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        return self.encoder(x)


class MultimodalFusionEncoder(nn.Module):
    """Fusion module for combining multiple sensor modalities.

    Combines features from different encoders into a unified representation
    for downstream robot learning tasks with attention-based fusion.
    """

    def __init__(self, feature_dims: dict[str, int], output_dim: int = 512):
        """Initialize the multimodal fusion encoder.

        Args:
            feature_dims: Dictionary mapping modality names to feature dimensions
            output_dim: Desired output feature dimension
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.modalities = list(feature_dims.keys())

        # Project each modality to common dimension
        self.projectors = nn.ModuleDict()
        common_dim = 256
        for modality, dim in feature_dims.items():
            self.projectors[modality] = nn.Linear(dim, common_dim)

        # Cross-attention for modality fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=common_dim, num_heads=8, dropout=0.1, batch_first=True
        )

        # Final fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(common_dim * len(self.modalities), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through multimodal fusion.

        Args:
            features: Dictionary mapping modality names to feature tensors

        Returns:
            torch.Tensor: Fused features of shape (batch, output_dim)
        """
        batch_size = next(iter(features.values())).shape[0]

        # Project all modalities to common dimension
        projected_features = []
        for modality in self.modalities:
            if modality in features:
                proj_feat = self.projectors[modality](features[modality])
                projected_features.append(proj_feat)
            else:
                # Handle missing modalities with zero features
                zero_feat = torch.zeros(
                    batch_size, 256, device=next(iter(features.values())).device
                )
                projected_features.append(zero_feat)

        # Stack for attention: (batch, num_modalities, common_dim)
        stacked_features = torch.stack(projected_features, dim=1)

        # Self-attention across modalities
        attended_features, _ = self.attention(
            stacked_features, stacked_features, stacked_features
        )

        # Flatten for final fusion
        flattened = attended_features.reshape(batch_size, -1)

        return self.fusion(flattened)
