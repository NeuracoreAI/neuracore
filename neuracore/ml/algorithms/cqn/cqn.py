"""CQN: Coarse-to-Fine Q-Network for continuous control.

This module implements the CQN (Coarse-to-Fine Q-Network) model for robot
manipulation tasks. CQN uses a coarse-to-fine discretization approach for
continuous action spaces combined with distributional Q-learning.

Reference: Seo et al. "Continuous Control with Coarse-to-fine
Reinforcement Learning." arXiv preprint arXiv:2407.07787 (2024).
"""

import logging
from typing import cast

import numpy as np
import torch
import torchvision.transforms as T
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedRGBData,
    DataType,
    ModelInitDescription,
)

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import (
    C2FCritic,
    MultiViewCNNEncoder,
    RandomShiftsAug,
    TruncatedNormal,
    schedule,
    soft_update_params,
)

logger = logging.getLogger(__name__)


class CQN(NeuracoreModel):
    """Implementation of CQN (Coarse-to-Fine Q-Network).

    CQN uses a hierarchical discretization approach where actions are predicted
    through multiple levels of refinement, combined with distributional Q-learning
    for improved sample efficiency and robustness. Unlike CQN_AS, this version
    predicts single-step actions rather than action sequences.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        lr: float = 5e-5,
        feature_dim: int = 64,
        hidden_dim: int = 512,
        levels: int = 3,
        bins: int = 5,
        atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 2.0,
        bc_lambda: float = 1.0,
        bc_margin: float = 0.01,
        critic_lambda: float = 0.1,
        critic_target_tau: float = 0.02,
        critic_target_interval: int = 1,
        weight_decay: float = 0.1,
        num_exploration_steps: int = 0,
        update_every_steps: int = 1,
        stddev_schedule: float = 0.01,
        discount: float = 0.99,
    ):
        """Initialize the CQN model.

        Args:
            model_init_description: Model initialization parameters
            lr: Learning rate for optimizers
            feature_dim: Feature dimension for the critic network
            hidden_dim: Hidden dimension for the critic network
            levels: Number of coarse-to-fine levels
            bins: Number of bins per level
            atoms: Number of atoms for distributional Q-learning
            v_min: Minimum value for value distribution
            v_max: Maximum value for value distribution
            bc_lambda: Weight for behavior cloning loss
            bc_margin: Margin for behavior cloning margin loss
            critic_lambda: Weight for critic loss
            critic_target_tau: Soft update coefficient for target critic
            critic_target_interval: Interval for updating target critic
            weight_decay: Weight decay for optimizer
            num_exploration_steps: Number of exploration steps
            update_every_steps: Update frequency
            stddev_schedule: Standard deviation schedule for exploration
            discount: Discount factor for RL
        """
        super().__init__(model_init_description)

        self.critic_target_tau = critic_target_tau
        self.critic_target_interval = critic_target_interval
        self.update_every_steps = update_every_steps
        self.num_exploration_steps = num_exploration_steps
        self.stddev_schedule = stddev_schedule
        self.bc_lambda = bc_lambda
        self.bc_margin = bc_margin
        self.critic_lambda = critic_lambda
        self.lr = lr
        self.weight_decay = weight_decay
        self.discount = discount

        # Get dimensions from dataset statistics
        self.num_views = len(self.dataset_statistics[DataType.RGB_IMAGES])
        self.low_dim_size = len(self.dataset_statistics[DataType.JOINT_POSITIONS])
        self.action_dim = len(self.dataset_statistics[DataType.JOINT_TARGET_POSITIONS])

        # Action shape: (action_dim,) - single step actions
        self.action_shape = (self.action_dim,)

        # RGB observation shape: (num_views, channels, height, width)
        # Default to 224x224 images with 3 channels
        rgb_obs_shape = (self.num_views, 3, 224, 224)
        low_dim_obs_shape = (1, self.low_dim_size)

        # Models - don't move to device here, let the training loop handle it
        self.encoder = MultiViewCNNEncoder(rgb_obs_shape)
        self.critic = C2FCritic(
            self.action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        )

        self.critic_target = C2FCritic(
            self.action_shape,
            self.encoder.repr_dim,
            low_dim_obs_shape[-1],
            feature_dim,
            hidden_dim,
            levels,
            bins,
            atoms,
            v_min,
            v_max,
        )

        self.critic_target.load_state_dict(self.critic.state_dict())

        # Data augmentation
        self.aug = RandomShiftsAug(pad=4)

        # Image transform
        self.transform = T.Resize((224, 224))

        # Set training mode
        self.train()
        self.critic_target.eval()

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Forward pass for inference.

        Args:
            batch: Input batch with observations

        Returns:
            dict[DataType, list[BatchedNCData]]: Model predictions
        """
        # Get RGB observations - shape: [B, num_views, C, H, W]
        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("RGB images are required for CQN inference")

        batched_rgb_data = cast(list[BatchedRGBData], batch.inputs[DataType.RGB_IMAGES])
        # Use last frame from each camera: [B, num_views, C, H, W]
        rgb_frames = [rgb_data.frame[:, -1, :, :, :] for rgb_data in batched_rgb_data]
        rgb_obs = torch.stack(rgb_frames, dim=1)

        # Resize images to expected size
        b_size, num_views, channels, height, width = rgb_obs.shape
        rgb_obs = rgb_obs.view(b_size * num_views, channels, height, width)
        rgb_obs = self.transform(rgb_obs)
        rgb_obs = rgb_obs.view(b_size, num_views, channels, 224, 224)

        # Get low-dimensional observations (joint positions)
        if DataType.JOINT_POSITIONS not in batch.inputs:
            raise ValueError("Joint positions are required for CQN inference")

        batched_joint_data = cast(
            list[BatchedJointData], batch.inputs[DataType.JOINT_POSITIONS]
        )
        joint_positions_mask = batch.inputs_mask[DataType.JOINT_POSITIONS]
        # Concatenate all joint values: [B, num_joints] - use last timestep
        joint_values = torch.cat(
            [bjd.value[:, -1, :] for bjd in batched_joint_data],
            dim=-1,
        )
        low_dim_obs = joint_values * joint_positions_mask

        # Predict single-step actions
        actions = self.act(rgb_obs, low_dim_obs, step=0, eval_mode=True)

        # Expand to prediction horizon (repeat single action for all timesteps)
        # Shape: [B, prediction_horizon, action_dim]
        actions = actions.unsqueeze(1).expand(
            b_size,
            self.output_prediction_horizon,
            self.action_dim,
        )

        # Build outputs in the same format as ACT / CQN-AS
        output_tensors: dict[DataType, list[BatchedNCData]] = {}
        batched_outputs: list[BatchedNCData] = []
        for i in range(self.action_dim):
            joint_preds = actions[:, :, i : i + 1]  # (B, T, 1)
            batched_outputs.append(BatchedJointData(value=joint_preds))
        output_tensors[DataType.JOINT_TARGET_POSITIONS] = batched_outputs

        return output_tensors

    def add_noise_to_action(self, action: np.ndarray, step: int) -> np.ndarray:
        """Add exploration noise to action.

        Args:
            action: Action to add noise to
            step: Current training step

        Returns:
            Action with added noise
        """
        if step < self.num_exploration_steps:
            action = np.random.uniform(-1.0, 1.0, size=action.shape).astype(
                action.dtype
            )
        else:
            stddev = schedule(self.stddev_schedule, step)
            action = np.clip(
                action
                + np.random.normal(0, stddev, size=action.shape).astype(action.dtype),
                -1.0,
                1.0,
            )
        return action

    def act(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        step: int,
        eval_mode: bool,
    ) -> torch.Tensor:
        """Get action from observations.

        Args:
            rgb_obs: RGB observation tensor [B, V, C, H, W]
            low_dim_obs: Low-dimensional observation tensor [B, D]
            step: Current step
            eval_mode: Whether in evaluation mode

        Returns:
            Action tensor [B, action_dim]
        """
        rgb_obs = self.encoder(rgb_obs)
        stddev = schedule(self.stddev_schedule, step)
        action = self.critic_target.get_action(rgb_obs, low_dim_obs)
        stddev = torch.ones_like(action) * stddev
        dist = TruncatedNormal(action, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_exploration_steps:
                action.uniform_(-1.0, 1.0)
        action = self.critic.encode_decode_action(action)
        return action

    def get_critic_loss(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_rgb_obs: torch.Tensor,
        next_low_dim_obs: torch.Tensor,
        demos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute critic loss.

        Args:
            rgb_obs: Current RGB observations (encoded)
            low_dim_obs: Current low-dimensional observations
            action: Actions taken
            reward: Rewards received
            discount: Discount factors
            next_rgb_obs: Next RGB observations (encoded)
            next_low_dim_obs: Next low-dimensional observations
            demos: Demo mask indicating which samples are demonstrations

        Returns:
            Critic loss tensor
        """
        with torch.no_grad():
            next_action = self.critic.get_action(next_rgb_obs, next_low_dim_obs)
            target_q_probs_a = self.critic_target.compute_target_q_dist(
                next_rgb_obs, next_low_dim_obs, next_action, reward, discount
            )

        # Cross entropy loss for C51
        q_probs, q_probs_a, log_q_probs, log_q_probs_a = self.critic(
            rgb_obs, low_dim_obs, action
        )
        q_critic_loss = -torch.sum(target_q_probs_a * log_q_probs_a, 3).mean()
        critic_loss = self.critic_lambda * q_critic_loss

        demos = demos.float().squeeze(-1)  # [B,]

        # BC - First-order stochastic dominance loss
        q_probs_cdf = torch.cumsum(q_probs, -1)
        q_probs_a_cdf = torch.cumsum(q_probs_a, -1)
        bc_fsd_loss = (
            (q_probs_a_cdf.unsqueeze(-2) - q_probs_cdf)
            .clamp(min=0)
            .sum(-1)
            .mean([-1, -2, -3])
        )
        # Avoid division by zero
        demos_sum = demos.sum().clamp(min=1.0)
        bc_fsd_loss = (bc_fsd_loss * demos).sum() / demos_sum
        critic_loss = critic_loss + self.bc_lambda * bc_fsd_loss

        # BC - Margin loss
        qs = (q_probs * self.critic.support.expand_as(q_probs)).sum(-1)
        qs_a = (q_probs_a * self.critic.support.expand_as(q_probs_a)).sum(-1)
        margin_loss = torch.clamp(
            self.bc_margin - (qs_a.unsqueeze(-1) - qs), min=0
        ).mean([-1, -2, -3])
        margin_loss = (margin_loss * demos).sum() / demos_sum
        critic_loss = critic_loss + self.bc_lambda * margin_loss

        return critic_loss

    def update_target_critic(self, step: int) -> None:
        """Update target critic with soft update.

        Args:
            step: Current training step
        """
        if step % self.critic_target_interval == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        CQN currently uses a BC-style training setup, where all samples are treated
        as demonstrations and synthetic rewards are used (no explicit RL rewards).

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        # Update target critic
        self.update_target_critic(1)

        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("RGB images are required for CQN training")
        if DataType.JOINT_POSITIONS not in batch.inputs:
            raise ValueError("Joint positions are required for CQN training")
        if DataType.JOINT_TARGET_POSITIONS not in batch.outputs:
            raise ValueError("Joint target positions are required for CQN training")

        b_size = batch.batch_size

        # Get RGB observations - current and next are the same for BC-style training
        batched_rgb_data = cast(list[BatchedRGBData], batch.inputs[DataType.RGB_IMAGES])
        rgb_frames = [rgb_data.frame[:, -1, :, :, :] for rgb_data in batched_rgb_data]
        rgb_obs = torch.stack(rgb_frames, dim=1)

        # Resize images
        _, num_views, channels, height, width = rgb_obs.shape
        rgb_obs_flat = rgb_obs.view(b_size * num_views, channels, height, width)
        rgb_obs_flat = self.transform(rgb_obs_flat)
        rgb_obs = rgb_obs_flat.view(b_size, num_views, channels, 224, 224)
        next_rgb_obs = rgb_obs

        # Get low-dimensional observations (joint positions)
        batched_joint_data = cast(
            list[BatchedJointData], batch.inputs[DataType.JOINT_POSITIONS]
        )
        joint_positions_mask = batch.inputs_mask[DataType.JOINT_POSITIONS]
        joint_values = torch.cat(
            [bjd.value[:, -1, :] for bjd in batched_joint_data],
            dim=-1,
        )
        low_dim_obs = joint_values * joint_positions_mask
        next_low_dim_obs = low_dim_obs

        # Get target actions - single-step prediction from first timestep
        batched_target_joints = cast(
            list[BatchedJointData], batch.outputs[DataType.JOINT_TARGET_POSITIONS]
        )
        target_actions = torch.cat(
            [bjd.value for bjd in batched_target_joints],
            dim=-1,
        )  # [B, T, action_dim]
        action = target_actions[:, 0, :]  # [B, action_dim]

        # Synthetic rewards for BC-style training
        reward = torch.ones(b_size, 1, device=action.device)

        # Discount tensor - all transitions considered non-terminal
        not_terminal = torch.ones(b_size, 1, device=action.device)
        discount = self.discount * not_terminal

        # All samples are demonstrations in BC-style training
        demos = torch.ones(b_size, 1, device=action.device)

        # Apply augmentation to images
        rgb_obs_aug = torch.stack(
            [self.aug(rgb_obs[:, v]) for v in range(rgb_obs.shape[1])],
            1,
        )
        next_rgb_obs_aug = torch.stack(
            [self.aug(next_rgb_obs[:, v]) for v in range(next_rgb_obs.shape[1])],
            1,
        )

        # Encode observations
        rgb_obs_encoded = self.encoder(rgb_obs_aug)
        with torch.no_grad():
            next_rgb_obs_encoded = self.encoder(next_rgb_obs_aug)

        # Compute critic loss
        critic_loss = self.get_critic_loss(
            rgb_obs_encoded,
            low_dim_obs,
            action,
            reward,
            discount,
            next_rgb_obs_encoded,
            next_low_dim_obs,
            demos,
        )

        return BatchedTrainingOutputs(
            losses={"critic_loss": critic_loss},
            metrics={"critic_loss": critic_loss},
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure and return optimizers for the model.

        Returns:
            list[torch.optim.Optimizer]: List of optimizers
        """
        self.encoder_opt = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.critic_opt = torch.optim.AdamW(
            self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [self.encoder_opt, self.critic_opt]

    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        """Return the data types supported by the model for input.

        Returns:
            set[DataType]: Set of supported input data types
        """
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGES,
        }

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        """Return the data types supported by the model for output.

        Returns:
            set[DataType]: Set of supported output data types
        """
        return {
            DataType.JOINT_TARGET_POSITIONS,
        }
