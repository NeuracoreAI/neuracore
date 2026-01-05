"""Module components for CQN-AS (Coarse-to-Fine Q-Network with Action Sequence).

This module contains the neural network components used in the CQN-AS algorithm
for continuous control via coarse-to-fine reinforcement learning with action sequences.

"""

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class C2FCritic(nn.Module):
    """Coarse-to-Fine Critic with distributional Q-learning for action sequences."""

    def __init__(
        self,
        action_shape: tuple,
        repr_dim: int,
        low_dim: int,
        feature_dim: int,
        hidden_dim: int,
        levels: int,
        bins: int,
        atoms: int,
        v_min: float,
        v_max: float,
        gru_layers: int,
        rgb_encoder_layers: int,
        use_parallel_impl: bool,
    ) -> None:
        """Initialize C2F Critic.

        Args:
            action_shape: Shape of action space (sequence_len, action_dim)
            repr_dim: Dimension of visual representation
            low_dim: Dimension of low-dimensional observations
            feature_dim: Feature dimension for encoders
            hidden_dim: Hidden dimension for networks
            levels: Number of coarse-to-fine levels
            bins: Number of bins per level
            atoms: Number of atoms for distributional Q-learning
            v_min: Minimum value for value distribution
            v_max: Maximum value for value distribution
            gru_layers: Number of GRU layers
            rgb_encoder_layers: Number of additional RGB encoder layers
            use_parallel_impl: Whether to use parallel implementation
        """
        super().__init__()

        self.levels = levels
        self.bins = bins
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.use_parallel_impl = use_parallel_impl
        actor_dim = action_shape[0] * action_shape[1]  # action_sequence * action_dim
        self.initial_low = nn.Parameter(
            torch.FloatTensor([-1.0] * actor_dim), requires_grad=False
        )
        self.initial_high = nn.Parameter(
            torch.FloatTensor([1.0] * actor_dim), requires_grad=False
        )
        self.support = nn.Parameter(
            torch.linspace(v_min, v_max, atoms), requires_grad=False
        )
        self.delta_z = (v_max - v_min) / (atoms - 1)

        self.network = C2FCriticNetwork(
            repr_dim,
            low_dim,
            action_shape,
            feature_dim,
            hidden_dim,
            gru_layers,
            rgb_encoder_layers,
            levels,
            bins,
            atoms,
        )

    def get_action(
        self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor
    ) -> torch.Tensor:
        """Get action from observations using coarse-to-fine selection.

        Args:
            rgb_obs: Encoded RGB observations
            low_dim_obs: Low-dimensional observations

        Returns:
            Continuous action tensor
        """
        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        features = self.network.encode(rgb_obs, low_dim_obs)
        for level in range(self.levels):
            q_logits = self.network.forward_each_level(
                level, features, (low + high) / 2
            )
            q_probs = F.softmax(q_logits, 3)
            qs = (q_probs * self.support.expand_as(q_probs).detach()).sum(3)
            argmax_q = random_action_if_within_delta(qs)
            if argmax_q is None:
                argmax_q = qs.max(-1)[1]  # [..., D]
            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        continuous_action = (high + low) / 2.0  # [..., D]
        return continuous_action

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        continuous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute value distributions for given obs and action.

        Args:
            rgb_obs: [B, repr_dim] shaped feature tensor
            low_dim_obs: [B, low_dim] shaped feature tensor
            continuous_action: [B, D] shaped action tensor

        Returns:
            Tuple of (q_probs, q_probs_a, log_q_probs, log_q_probs_a)
        """
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )

        q_probs_per_level = []
        q_probs_a_per_level = []
        log_q_probs_per_level = []
        log_q_probs_a_per_level = []

        low = self.initial_low.repeat(rgb_obs.shape[0], 1).detach()
        high = self.initial_high.repeat(rgb_obs.shape[0], 1).detach()

        if self.use_parallel_impl:
            # Pre-compute previous actions for all the levels
            prev_actions = []
            for level in range(self.levels):
                prev_actions.append((low + high) / 2)
                argmax_q = discrete_action[
                    ..., level, :
                ].long()  # [..., L, D] -> [..., D]
                low, high = zoom_in(low, high, argmax_q, self.bins)
            q_logits_all = self.network(
                rgb_obs, low_dim_obs, torch.stack(prev_actions, 1)
            )
        else:
            features = self.network.encode(rgb_obs, low_dim_obs)
        for level in range(self.levels):
            if self.use_parallel_impl:
                q_logits = q_logits_all[:, level]
            else:
                q_logits = self.network.forward_each_level(
                    level, features, (low + high) / 2
                )
            argmax_q = discrete_action[..., level, :].long()  # [..., L, D] -> [..., D]

            # (Log) Probs [..., D, bins, atoms]
            # (Log) Probs_a [..., D, atoms]
            q_probs = F.softmax(q_logits, 3)  # [B, D, bins, atoms]
            q_probs_a = torch.gather(
                q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            q_probs_a = q_probs_a[..., 0, :]  # [B, D, atoms]

            log_q_probs = F.log_softmax(q_logits, 3)  # [B, D, bins, atoms]
            log_q_probs_a = torch.gather(
                log_q_probs,
                dim=-2,
                index=argmax_q.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat_interleave(self.atoms, -1),
            )
            log_q_probs_a = log_q_probs_a[..., 0, :]  # [B, D, atoms]

            q_probs_per_level.append(q_probs)
            q_probs_a_per_level.append(q_probs_a)
            log_q_probs_per_level.append(log_q_probs)
            log_q_probs_a_per_level.append(log_q_probs_a)

            # Zoom-in
            low, high = zoom_in(low, high, argmax_q, self.bins)

        q_probs = torch.stack(q_probs_per_level, -4)  # [B, L, D, bins, atoms]
        q_probs_a = torch.stack(q_probs_a_per_level, -3)  # [B, L, D, atoms]
        log_q_probs = torch.stack(log_q_probs_per_level, -4)
        log_q_probs_a = torch.stack(log_q_probs_a_per_level, -3)
        return q_probs, q_probs_a, log_q_probs, log_q_probs_a

    def compute_target_q_dist(
        self,
        next_rgb_obs: torch.Tensor,
        next_low_dim_obs: torch.Tensor,
        next_continuous_action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Compute target distribution for distributional critic.

        Based on https://github.com/Kaixhin/Rainbow/blob/master/agent.py

        Args:
            next_rgb_obs: [B, repr_dim] shaped feature tensor
            next_low_dim_obs: [B, low_dim] shaped feature tensor
            next_continuous_action: [B, D] shaped action tensor
            reward: [B, 1] shaped reward tensor
            discount: [B, 1] shaped discount tensor

        Returns:
            Target distribution tensor of shape [B, L, D, atoms]
        """
        next_q_probs_a = self.forward(
            next_rgb_obs, next_low_dim_obs, next_continuous_action
        )[1]

        shape = next_q_probs_a.shape  # [B, L, D, atoms]
        next_q_probs_a = next_q_probs_a.view(-1, self.atoms)
        batch_size = next_q_probs_a.shape[0]

        # Compute Tz for [B, atoms]
        tz = reward + discount * self.support.unsqueeze(0).detach()
        tz = tz.clamp(min=self.v_min, max=self.v_max)
        # Compute L2 projection of Tz onto fixed support z
        b = (tz - self.v_min) / self.delta_z
        # Mask for conditions
        lower, upper = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        lower_mask = (upper > 0) & (lower == upper)
        upper_mask = (lower < (self.atoms - 1)) & (lower == upper)
        # Apply masks separately
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        # Repeat Tz for (L * D) times -> [B * L * D, atoms]
        multiplier = batch_size // lower.shape[0]
        b = torch.repeat_interleave(b, multiplier, 0)
        lower = torch.repeat_interleave(lower, multiplier, 0)
        upper = torch.repeat_interleave(upper, multiplier, 0)

        # Distribute probability of Tz
        m = torch.zeros_like(next_q_probs_a)
        offset = (
            torch.linspace(
                0,
                ((batch_size - 1) * self.atoms),
                batch_size,
                device=lower.device,
                dtype=lower.dtype,
            )
            .unsqueeze(1)
            .expand(batch_size, self.atoms)
        )
        m.view(-1).index_add_(
            0,
            (lower + offset).view(-1),
            (next_q_probs_a * (upper.float() - b)).view(-1),
        )  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(
            0,
            (upper + offset).view(-1),
            (next_q_probs_a * (b - lower.float())).view(-1),
        )  # m_u = m_u + p(s_t+n, a*)(b - l)

        m = m.view(*shape)  # [B, L, D, atoms]
        return m

    def encode_decode_action(self, continuous_action: torch.Tensor) -> torch.Tensor:
        """Encode and decode actions to discretize them.

        Args:
            continuous_action: Continuous action tensor

        Returns:
            Discretized continuous action tensor
        """
        discrete_action = encode_action(
            continuous_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        continuous_action = decode_action(
            discrete_action,
            self.initial_low,
            self.initial_high,
            self.levels,
            self.bins,
        )
        return continuous_action


class C2FCriticNetwork(nn.Module):
    """Coarse-to-Fine Critic Network with GRU for action sequences."""

    def __init__(
        self,
        repr_dim: int,
        low_dim: int,
        action_shape: tuple,
        feature_dim: int,
        hidden_dim: int,
        gru_layers: int,
        rgb_encoder_layers: int,
        levels: int,
        bins: int,
        atoms: int,
    ) -> None:
        """Initialize C2F Critic Network.

        Args:
            repr_dim: Dimension of visual representation
            low_dim: Dimension of low-dimensional observations
            action_shape: Shape of action space (sequence_len, action_dim)
            feature_dim: Feature dimension for encoders
            hidden_dim: Hidden dimension for networks
            gru_layers: Number of GRU layers
            rgb_encoder_layers: Number of additional RGB encoder layers
            levels: Number of coarse-to-fine levels
            bins: Number of bins per level
            atoms: Number of atoms for distributional Q-learning
        """
        super().__init__()
        self._levels = levels
        self._action_sequence, self._actor_dim = action_shape
        self._bins = bins

        # Advantage stream in Dueling network
        # RGB encoder for advantage stream
        adv_rgb_encoder_net = []
        input_dim = repr_dim
        for _ in range(rgb_encoder_layers):
            adv_rgb_encoder_net += [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            input_dim = hidden_dim
        adv_rgb_encoder_net = adv_rgb_encoder_net + [
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        ]
        self.adv_rgb_encoder = nn.Sequential(*adv_rgb_encoder_net)

        # Low-dimensional encoder for advantage stream
        self.adv_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        # Main network for advantage stream
        self.adv_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.adv_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.adv_head = nn.Linear(
            hidden_dim,
            self._actor_dim * bins * atoms,
        )
        self.adv_output_shape = (self._action_sequence * self._actor_dim, bins, atoms)

        # Value stream in Dueling network
        # RGB encoder for value stream
        value_rgb_encoder_net = []
        input_dim = repr_dim
        for _ in range(rgb_encoder_layers):
            value_rgb_encoder_net += [
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
            input_dim = hidden_dim
        value_rgb_encoder_net = value_rgb_encoder_net + [
            nn.Linear(input_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        ]
        self.value_rgb_encoder = nn.Sequential(*value_rgb_encoder_net)

        # Low-dimensional encoder for value stream
        self.value_low_dim_encoder = nn.Sequential(
            nn.Linear(low_dim, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        # Main network for value stream
        self.value_net = nn.Sequential(
            nn.Linear(
                feature_dim * 2 + self._action_sequence + self._actor_dim + levels,
                hidden_dim,
                bias=False,
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.value_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.value_head = nn.Linear(
            hidden_dim,
            self._actor_dim * 1 * atoms,
        )
        self.value_output_shape = (self._action_sequence * self._actor_dim, 1, atoms)

        self.apply(weight_init)
        self.adv_head.weight.data.fill_(0.0)
        self.adv_head.bias.data.fill_(0.0)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)

    def encode(
        self, rgb_obs: torch.Tensor, low_dim_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations into value and advantage features.

        Args:
            rgb_obs: RGB observation tensor
            low_dim_obs: Low-dimensional observation tensor

        Returns:
            Tuple of (value_features, advantage_features)
        """
        value_h = torch.cat(
            [self.value_rgb_encoder(rgb_obs), self.value_low_dim_encoder(low_dim_obs)],
            -1,
        )
        adv_h = torch.cat(
            [self.adv_rgb_encoder(rgb_obs), self.adv_low_dim_encoder(low_dim_obs)],
            -1,
        )
        return value_h, adv_h

    def forward_each_level(
        self,
        level: int,
        features: tuple[torch.Tensor, torch.Tensor],
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Process forward step for each level.

        Args:
            level: Level index
            features: Tuple of (value_h, adv_h) shared for all levels
            prev_action: [B, D] shaped prev action for the current level

        Returns:
            Q-logits tensor [B, action_sequence * action_dimensions, bins, atoms]
        """
        value_h, adv_h = features

        level_id = (
            torch.eye(self._levels, device=value_h.device, dtype=value_h.dtype)[level]
            .unsqueeze(0)
            .repeat_interleave(value_h.shape[0], 0)
        )
        level_id = level_id.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        prev_action = prev_action.view(
            -1, self._action_sequence, self._actor_dim
        )  # [B, T, D]
        action_sequence_id = (
            torch.eye(self._action_sequence, device=value_h.device, dtype=value_h.dtype)
            .unsqueeze(0)
            .repeat_interleave(value_h.shape[0], 0)
        )  # [B, T, T]

        # Value
        value_h = value_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        value_x = torch.cat(
            [value_h, prev_action, action_sequence_id, level_id], -1
        )  # [B, T, D]
        # Process through MLP for each action sequence step
        value_feats = self.value_net(value_x)
        # Process through GRU
        value_feats, _ = self.value_gru(value_feats)
        values = self.value_head(value_feats).view(-1, *self.value_output_shape)

        # Advantage
        adv_h = adv_h.unsqueeze(1).repeat_interleave(self._action_sequence, 1)
        adv_x = torch.cat(
            [adv_h, prev_action, action_sequence_id, level_id], -1
        )  # [B, T, D]
        # Process through MLP for each action sequence step
        adv_feats = self.adv_net(adv_x)
        # Process through GRU
        adv_feats, _ = self.adv_gru(adv_feats)
        advantages = self.adv_head(adv_feats).view(-1, *self.adv_output_shape)

        q_logits = values + advantages - advantages.mean(-2, keepdim=True)
        return q_logits

    def forward(
        self,
        rgb_obs: torch.Tensor,
        low_dim_obs: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Process forward step for all levels at once.

        This is possible because we have pre-computed prev_actions for all levels
        when computing Q(s,a) for given action a. For getting actions for given s,
        use forward_each_level instead.

        Args:
            rgb_obs: [B, D] shaped RGB observation tensor
            low_dim_obs: [B, D] shaped low-dim observation tensor
            prev_actions: [B, L, D] shaped previous actions tensor

        Returns:
            Q-logits tensor [B, L, action_sequence * action_dimensions, bins, atoms]
        """
        device, dtype = rgb_obs.device, rgb_obs.dtype
        levels = prev_actions.size(1)
        B, L, T = prev_actions.size(0), levels, self._action_sequence

        # Reshape previous actions
        prev_actions = prev_actions.view(-1, L, T, self._actor_dim)  # [B, L, T, D]

        # Action sequence id - [T, T] -> [B, L, T, T]
        action_sequence_id = torch.eye(T, device=device, dtype=dtype)[
            None, None, :, :
        ].repeat(B, L, 1, 1)

        # level id - [L, L] -> [B, L, T, L]
        level_id = torch.eye(L, device=device, dtype=dtype)[None, :, None, :].repeat(
            B, 1, T, 1
        )

        # Encode features
        value_h, adv_h = self.encode(rgb_obs, low_dim_obs)

        # Value
        value_h = value_h[:, None, None, :].repeat(1, L, T, 1)
        value_x = torch.cat([value_h, prev_actions, action_sequence_id, level_id], -1)
        value_feats = self.value_net(value_x)
        # Process through GRU
        value_feats = value_feats.view(B * L, T, -1)
        value_feats = self.value_gru(value_feats)[0]
        values = self.value_head(value_feats).view(B, L, *self.value_output_shape)

        # Advantage
        adv_h = adv_h[:, None, None, :].repeat(1, L, T, 1)
        adv_x = torch.cat([adv_h, prev_actions, action_sequence_id, level_id], -1)
        adv_feats = self.adv_net(adv_x)
        # Process through GRU
        adv_feats = adv_feats.view(B * L, T, -1)
        adv_feats = self.adv_gru(adv_feats)[0]
        advantages = self.adv_head(adv_feats).view(B, L, *self.adv_output_shape)

        q_logits = values + advantages - advantages.mean(-2, keepdim=True)
        return q_logits


def weight_init(m: nn.Module) -> None:
    """Initialize weights for neural network layers.

    Args:
        m: Neural network module to initialize
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def random_action_if_within_delta(
    qs: torch.Tensor, delta: float = 0.0001
) -> torch.Tensor | None:
    """Return random action if Q-values are within delta of each other.

    Args:
        qs: Q-value tensor
        delta: Threshold for considering Q-values equal

    Returns:
        Random action indices if Q-values are within delta, None otherwise
    """
    q_diff = qs.max(-1).values - qs.min(-1).values
    random_action_mask = q_diff < delta
    argmax_q = qs.max(-1)[1]
    random_actions = torch.randint(
        0, qs.size(-1), random_action_mask.shape, device=qs.device
    )
    argmax_q = torch.where(random_action_mask, random_actions, argmax_q)
    return argmax_q


def encode_action(
    continuous_action: torch.Tensor,
    initial_low: torch.Tensor,
    initial_high: torch.Tensor,
    levels: int,
    bins: int,
) -> torch.Tensor:
    """Encode continuous action to discrete action.

    Args:
        continuous_action: [..., D] shape tensor
        initial_low: [D] shape tensor consisting of -1
        initial_high: [D] shape tensor consisting of 1
        levels: Number of coarse-to-fine levels
        bins: Number of bins per level

    Returns:
        Discrete action tensor [..., L, D] where L is the level
    """
    low = initial_low.repeat(*continuous_action.shape[:-1], 1)
    high = initial_high.repeat(*continuous_action.shape[:-1], 1)

    indices = []
    for _ in range(levels):
        # Put continuous values into bin
        slice_range = (high - low) / bins
        idx = torch.floor((continuous_action - low) / slice_range)
        idx = torch.clip(idx, 0, bins - 1)
        indices.append(idx)

        # Re-compute low/high for each bin (i.e., Zoom-in)
        recalculated_action = low + slice_range * idx
        recalculated_action = torch.clip(recalculated_action, -1.0, 1.0)
        low = recalculated_action
        high = recalculated_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)
    discrete_action = torch.stack(indices, -2)
    return discrete_action


def decode_action(
    discrete_action: torch.Tensor,
    initial_low: torch.Tensor,
    initial_high: torch.Tensor,
    levels: int,
    bins: int,
) -> torch.Tensor:
    """Decode discrete action to continuous action.

    Args:
        discrete_action: [..., L, D] shape tensor
        initial_low: [D] shape tensor consisting of -1
        initial_high: [D] shape tensor consisting of 1
        levels: Number of coarse-to-fine levels
        bins: Number of bins per level

    Returns:
        Continuous action tensor [..., D]
    """
    low = initial_low.repeat(*discrete_action.shape[:-2], 1)
    high = initial_high.repeat(*discrete_action.shape[:-2], 1)
    for i in range(levels):
        slice_range = (high - low) / bins
        continuous_action = low + slice_range * discrete_action[..., i, :]
        low = continuous_action
        high = continuous_action + slice_range
        low = torch.maximum(-torch.ones_like(low), low)
        high = torch.minimum(torch.ones_like(high), high)
    continuous_action = (high + low) / 2.0
    return continuous_action


def zoom_in(
    low: torch.Tensor, high: torch.Tensor, argmax_q: torch.Tensor, bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zoom-in to the selected interval.

    Args:
        low: [D] shape tensor that denotes minimum of the current interval
        high: [D] shape tensor that denotes maximum of the current interval
        argmax_q: Selected bin indices
        bins: Number of bins

    Returns:
        Tuple of (low, high) tensors for the next interval
    """
    slice_range = (high - low) / bins
    continuous_action = low + slice_range * argmax_q
    low = continuous_action
    high = continuous_action + slice_range
    low = torch.maximum(-torch.ones_like(low), low)
    high = torch.minimum(torch.ones_like(high), high)
    return low, high


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """Soft update target network parameters.

    Args:
        net: Source network
        target_net: Target network to update
        tau: Soft update coefficient
    """
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class RandomShiftsAug(nn.Module):
    """Random shift augmentation for RGB observations."""

    def __init__(self, pad: int) -> None:
        """Initialize random shifts augmentation.

        Args:
            pad: Padding size for shifts
        """
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random shift augmentation.

        Args:
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Augmented tensor of same shape
        """
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        lin_range = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        lin_range = lin_range.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([lin_range, lin_range.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class ImgChLayerNorm(nn.Module):
    """Channel-wise layer normalization for images."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        """Initialize channel-wise layer normalization.

        Args:
            num_channels: Number of input channels
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise layer normalization.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Normalized tensor of same shape
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MultiViewCNNEncoder(nn.Module):
    """Multi-view CNN encoder for RGB observations."""

    def __init__(self, obs_shape: tuple[int, ...]) -> None:
        """Initialize multi-view CNN encoder.

        Args:
            obs_shape: Shape of observations (num_views, channels, height, width)
        """
        super().__init__()

        assert len(obs_shape) == 4
        self.num_views = obs_shape[0]

        self.conv_nets = nn.ModuleList()
        for _ in range(self.num_views):
            conv_net = nn.Sequential(
                nn.Conv2d(obs_shape[1], 32, 4, stride=2, padding=1),
                ImgChLayerNorm(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, stride=2, padding=1),
                ImgChLayerNorm(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                ImgChLayerNorm(128),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, stride=2, padding=1),
                ImgChLayerNorm(256),
                nn.SiLU(),
            )
            self.conv_nets.append(conv_net)

        dummy_input = torch.zeros(1, *obs_shape)
        output = self.forward(dummy_input)
        self.repr_dim = output.shape[-1]

        self.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode multi-view RGB observations.

        Args:
            obs: Input tensor of shape [B, V, C, H, W]

        Returns:
            Encoded features of shape [B, repr_dim]
        """
        obs = obs / 255.0 - 0.5
        hs = []
        for v in range(self.num_views):
            h = self.conv_nets[v](obs[:, v])
            h = h.view(h.shape[0], -1)
            hs.append(h)
        h = torch.cat(hs, -1)
        return h


def schedule(schedule_str: str | float, step: int) -> float:
    """Parse and compute schedule value.

    Args:
        schedule_str: Schedule string or float value
        step: Current step

    Returns:
        Computed schedule value
    """
    try:
        return float(schedule_str)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", str(schedule_str))
        if match:
            init, final, duration = (float(g) for g in match.groups())
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", str(schedule_str))
        if match:
            init, final1, duration1, final2, duration2 = (
                float(g) for g in match.groups()
            )
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(str(schedule_str))


class TruncatedNormal(pyd.Normal):
    """Truncated normal distribution."""

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """Initialize truncated normal distribution.

        Args:
            loc: Mean of the distribution
            scale: Standard deviation of the distribution
            low: Lower bound for truncation
            high: Upper bound for truncation
            eps: Small constant for numerical stability
        """
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp values to truncation bounds."""
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(
        self, clip: float | None = None, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Sample from truncated normal distribution.

        Args:
            clip: Optional clipping value for noise
            sample_shape: Shape of samples to draw

        Returns:
            Sampled tensor
        """
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
