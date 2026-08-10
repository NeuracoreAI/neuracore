"""Diffusion Policy with an explicit done-flag classification head."""

import logging
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from neuracore_types import (
    BatchedCustom1DData,
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    BatchedRGBData,
    CameraDataStats,
    Custom1DDataStats,
    DataItemStats,
    DataType,
    JointDataStats,
    ModelInitDescription,
    ParallelGripperOpenAmountDataStats,
)

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import MinMaxNormalizer
from neuracore.ml.core.neuracore_model import DEFAULT_OUTPUT_DATA_TYPE_ORDER

from .modules import DiffusionConditionalUnet1d, DiffusionPolicyImageEncoder

logger = logging.getLogger(__name__)

proprio_normalizer = MinMaxNormalizer  # or MeanStdNormalizer
action_normalizer = MinMaxNormalizer  # or MeanStdNormalizer
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

_ACTION_OUTPUT_DATA_TYPES = frozenset({
    DataType.JOINT_TARGET_POSITIONS,
    DataType.JOINT_POSITIONS,
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
})


class DiffusionPolicyWithDone(NeuracoreModel):
    """Diffusion Policy with a separate done-flag classification head.

    Continuous joint/gripper actions are predicted via diffusion or flow
    matching. Binary done signals are predicted as ``DataType.CUSTOM_1D`` from
    a lightweight MLP on the shared vision-proprio conditioning features.
    """

    CANONICAL_OUTPUT_DATA_TYPE_ORDER = DEFAULT_OUTPUT_DATA_TYPE_ORDER + (
        DataType.CUSTOM_1D,
    )

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 64,
        unet_down_dims: tuple[int, ...] = (
            256,
            512,
            1024,
        ),
        unet_kernel_size: int = 5,
        unet_n_groups: int = 8,
        unet_diffusion_step_embed_dim: int = 128,
        spatial_softmax_num_keypoints: int = 32,
        unet_use_film_scale_modulation: bool = True,
        use_pretrained_weights: bool = True,
        use_resnet_stats: bool = True,
        process_type: str = "diffusion",
        noise_scheduler_type: str = "DDPM",  # diffusion only
        num_train_timesteps: int = 100,  # diffusion only
        num_inference_steps: int = 100,
        beta_start: float = 0.0001,  # diffusion only
        beta_end: float = 0.02,  # diffusion only
        beta_schedule: str = "squaredcos_cap_v2",  # diffusion only
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        lr: float = 1e-4,
        freeze_backbone: bool = False,
        lr_backbone: float = 1e-4,
        weight_decay: float = 1e-6,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        prediction_type: str = "epsilon",  # diffusion only
        lr_scheduler_type: str = "cosine",
        lr_scheduler_num_warmup_steps: int = 500,
        done_loss_weight: float = 0.1,
    ):
        """Initialize the Diffusion Policy with Done model.

        Args:
            model_init_description: Model initialization configuration.
            hidden_dim: Hidden dimension for image encoders and done head.
            unet_down_dims: Downsampling dimensions for UNet.
            unet_kernel_size: Kernel size for UNet convolutions.
            unet_n_groups: Number of groups for group normalization.
            unet_diffusion_step_embed_dim: Dimension of diffusion step embeddings.
            spatial_softmax_num_keypoints: Number of keypoints for spatial softmax.
            unet_use_film_scale_modulation: Whether to use FiLM scale modulation.
            use_pretrained_weights: Whether to load pretrained ResNet weights.
            use_resnet_stats: Whether to use ResNet normalization statistics.
            process_type: Generative process to use, "diffusion" (default) or
                "flow_matching". See the note below for which parameters apply.
            noise_scheduler_type: Type of noise scheduler ("DDPM" or "DDIM").
                (diffusion only)
            num_train_timesteps: Number of timesteps for training. (diffusion only)
            num_inference_steps: Number of sampling steps at inference. For flow
                matching this is the number of Euler integration steps
                (typically ~10).
            beta_start: Starting beta value for noise schedule. (diffusion only)
            beta_end: Ending beta value for noise schedule. (diffusion only)
            beta_schedule: Beta schedule type. (diffusion only)
            clip_sample: Whether to clip samples (both processes).
            clip_sample_range: Range for clipping samples (both processes).
            lr: Learning rate for main parameters.
            freeze_backbone: Whether to freeze image encoder backbone
            lr_backbone: Learning rate for backbone parameters.
            weight_decay: Weight decay for optimization.
            optimizer_betas: Betas for optimizer.
            optimizer_eps: Epsilon for optimizer.
            prediction_type: Type of prediction ("epsilon" or "sample").
                (diffusion only)
            lr_scheduler_type: Type of the learning rate scheduler
                ("cosine", "linear", etc.).
            lr_scheduler_num_warmup_steps: Number of warmup steps for the scheduler.
            done_loss_weight: Weight for the done-flag BCE loss relative to the
                diffusion MSE loss.
        """
        super().__init__(model_init_description)
        self.use_resnet_stats = use_resnet_stats
        self.lr = lr
        self.freeze_backbone = freeze_backbone
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_scheduler_num_warmup_steps = lr_scheduler_num_warmup_steps
        self.prediction_type = prediction_type
        self.num_inference_steps = num_inference_steps
        self.done_loss_weight = done_loss_weight
        self.hidden_dim = hidden_dim

        if process_type not in ("diffusion", "flow_matching"):
            raise ValueError(
                f"Unsupported process_type {process_type!r}; "
                "expected 'diffusion' or 'flow_matching'."
            )
        self.process_type = process_type
        # Used by both processes (flow matching clamps each Euler step).
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        data_stats: dict[DataType, DataItemStats] = {}

        # Setup proprioceptive data
        self.proprio_dims: dict[DataType, tuple[int, int]] = {}
        proprio_stats = []
        current_dim = 0

        for data_type in [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        ]:
            if data_type in self.input_data_types:
                if data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                    stats = cast(
                        list[ParallelGripperOpenAmountDataStats],
                        self.input_dataset_statistics[data_type],
                    )
                    combined_stats = DataItemStats()
                    for stat in stats:
                        combined_stats = combined_stats.concatenate(stat.open_amount)
                else:
                    stats = cast(
                        list[JointDataStats], self.input_dataset_statistics[data_type]
                    )
                    combined_stats = DataItemStats()
                    for stat in stats:
                        combined_stats = combined_stats.concatenate(stat.value)
                    data_stats[data_type] = combined_stats

                proprio_stats.append(combined_stats)
                dim = len(combined_stats.mean)
                self.proprio_dims[data_type] = (current_dim, current_dim + dim)
                current_dim += dim

        global_cond_dim = current_dim

        # Continuous action outputs only (CUSTOM_1D is handled by the done head).
        self.action_output_data_types = [
            data_type
            for data_type in self.ordered_output_data_types
            if data_type in _ACTION_OUTPUT_DATA_TYPES
        ]
        if not self.action_output_data_types:
            raise ValueError(
                "DiffusionPolicyWithDone requires at least one continuous action "
                "output type (joint or gripper positions); CUSTOM_1D alone is not "
                "supported."
            )

        # Setup continuous output data for the diffusion / flow head
        self.max_output_size = 0
        output_stats = []
        self.output_dims: dict[DataType, tuple[int, int]] = {}
        current_output_dim = 0

        for data_type in self.action_output_data_types:
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                stats = cast(
                    list[JointDataStats], self.output_dataset_statistics[data_type]
                )
                combined_stats = DataItemStats()
                for stat in stats:
                    combined_stats = combined_stats.concatenate(stat.value)
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                stats = cast(
                    list[ParallelGripperOpenAmountDataStats],
                    self.output_dataset_statistics[data_type],
                )
                combined_stats = DataItemStats()
                for stat in stats:
                    combined_stats = combined_stats.concatenate(stat.open_amount)
            else:
                raise ValueError(f"Unsupported action output data type: {data_type}")
            data_stats[data_type] = combined_stats
            output_stats.append(combined_stats)
            dim = len(combined_stats.mean)
            self.output_dims[data_type] = (current_output_dim, current_output_dim + dim)
            current_output_dim += dim
            self.max_output_size += dim

        # Setup CUSTOM_1D done-flag dims (classification head, not diffused)
        self.custom_1d_slot_dims: list[int] = []
        self.custom_1d_dim = 0
        self.done_head: nn.Sequential | None = None
        if DataType.CUSTOM_1D in self.output_data_types:
            custom_stats = cast(
                list[Custom1DDataStats],
                self.output_dataset_statistics[DataType.CUSTOM_1D],
            )
            for stat in custom_stats:
                slot_dim = len(stat.data.mean)
                self.custom_1d_slot_dims.append(slot_dim)
                self.custom_1d_dim += slot_dim
            if self.custom_1d_dim == 0:
                raise ValueError(
                    "CUSTOM_1D is configured as an output but has zero feature "
                    "dimensions in the dataset statistics."
                )

        # Setup normalizers
        # Only create proprio_normalizer if there are proprioception stats
        # This allows the algorithm to work without proprioception (visual-only)
        self.proprio_normalizer = (
            proprio_normalizer(name="proprioception", statistics=proprio_stats)
            if proprio_stats
            else None
        )
        self.action_normalizer = action_normalizer(
            name="actions", statistics=output_stats
        )

        # Vision components
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats],
                self.input_dataset_statistics[DataType.RGB_IMAGES],
            )
            max_cameras = len(stats)
            self.image_normalizers = nn.ModuleList()
            self.image_encoders = nn.ModuleList()
            for i in range(max_cameras):
                if use_resnet_stats:
                    mean, std = RESNET_MEAN, RESNET_STD
                else:
                    mean_c_h_w, std_c_h_w = stats[i].frame.mean, stats[i].frame.std
                    mean = mean_c_h_w.mean(axis=(1, 2)).tolist()
                    std = std_c_h_w.mean(axis=(1, 2)).tolist()
                self.image_normalizers.append(T.Normalize(mean=mean, std=std))
                self.image_encoders.append(
                    DiffusionPolicyImageEncoder(
                        feature_dim=hidden_dim,
                        spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
                        use_pretrained_weights=use_pretrained_weights,
                    )
                )

            global_cond_dim += self.image_encoders[0].feature_dim * max_cameras

        self.global_cond_dim = global_cond_dim
        self.unet = DiffusionConditionalUnet1d(
            action_dim=self.max_output_size,
            global_cond_dim=global_cond_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
            diffusion_step_embed_dim=unet_diffusion_step_embed_dim,
            use_film_scale_modulation=unet_use_film_scale_modulation,
        )

        if self.custom_1d_dim > 0:
            self.done_head = nn.Sequential(
                nn.Linear(global_cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim, self.output_prediction_horizon * self.custom_1d_dim
                ),
            )

        # Flow matching integrates a learned velocity field directly and needs no
        # diffusers scheduler; only build one for the diffusion process.
        self.noise_scheduler: DDPMScheduler | DDIMScheduler | None
        if self.process_type == "diffusion":
            kwargs: dict[str, Any] = {
                "num_train_timesteps": num_train_timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "beta_schedule": beta_schedule,
                "clip_sample": clip_sample,
                "clip_sample_range": clip_sample_range,
                "prediction_type": prediction_type,
            }
            self.noise_scheduler = self._make_noise_scheduler(
                noise_scheduler_type, **kwargs
            )
        else:
            self.noise_scheduler = None

        # Setup parameter groups
        self._setup_optimizer_param_groups()

    def _setup_optimizer_param_groups(self) -> None:
        """Setup parameter groups for optimizer."""
        backbone_params, other_params = [], []
        for name, param in self.named_parameters():
            if any(backbone in name for backbone in ["image_encoders"]):
                backbone_params.append(param)
            else:
                other_params.append(param)

        if self.freeze_backbone:
            for param in backbone_params:
                param.requires_grad = False
            self.param_groups = [{"params": other_params, "lr": self.lr}]
        else:
            self.param_groups = [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": other_params, "lr": self.lr},
            ]

    def _combine_proprio(self, batch: BatchedInferenceInputs) -> torch.FloatTensor:
        """Combine proprioceptive inputs into a single normalized feature vector.

        Types configured in self.input_data_types but absent from batch.inputs
        are zero-filled to preserve positional alignment for cross-embodiment
        batches. Returns None when no proprioceptive types are configured.

        Args:
            batch: Input batch containing joint state data

        Returns:
            torch.FloatTensor: Combined and normalized joint state features
        """
        proprio_list = []
        for data_type in [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        ]:
            if data_type not in self.input_data_types:
                continue
            if data_type not in batch.inputs:
                start_idx, end_idx = self.proprio_dims[data_type]
                proprio_list.append(
                    torch.zeros(
                        (len(batch), end_idx - start_idx),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                continue

            batched_nc_data = batch.inputs[data_type]
            mask = batch.inputs_mask[data_type]

            if data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched_gripper_data = cast(
                    list[BatchedParallelGripperOpenAmountData], batched_nc_data
                )
                proprio_data = torch.cat(
                    [bgd.open_amount for bgd in batched_gripper_data], dim=-1
                )
            else:
                batched_joint_data = cast(list[BatchedJointData], batched_nc_data)
                proprio_data = torch.cat(
                    [bjd.value for bjd in batched_joint_data], dim=-1
                )

            last_proprio = proprio_data[:, -1, :]  # (B, num_features)
            masked_proprio = last_proprio * mask
            proprio_list.append(masked_proprio)

        # If no proprioception data is available, return None
        # This allows the algorithm to work with visual-only inputs
        if not proprio_list:
            return None

        # Concatenate all proprio together: (B, total_proprio_dim)
        all_proprio = torch.cat(proprio_list, dim=-1)

        # Normalize once on all proprio
        # Check if normalizer exists (it should if we have proprio data)
        if self.proprio_normalizer is None:
            raise ValueError(
                "Proprioception inputs were provided but no normalizer was available."
            )
        normalized_proprio = self.proprio_normalizer.normalize(all_proprio)

        return normalized_proprio

    def _conditional_sample(
        self,
        batch_size: int,
        prediction_horizon: int,
        global_cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample action sequence conditioned on the observations.

        Args:
            batch_size: Batch size
            prediction_horizon: Action sequence prediction horizon
            global_cond: Global conditioning tensor
            generator: Random number generator

        Returns:
            torch.Tensor: Sampled action sequence with shape
            (B, prediction_horizon, action_dim)
        """
        # Both processes start from a Gaussian prior.
        sample = torch.randn(
            size=(
                batch_size,
                prediction_horizon,
                self.max_output_size,
            ),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        if self.process_type == "flow_matching":
            return self._flow_matching_sample(sample, global_cond)

        assert self.noise_scheduler is not None
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample

        return sample

    def _flow_matching_sample(
        self,
        sample: torch.Tensor,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the learned velocity field from noise (t=0) to data (t=1).

        Uses a forward-Euler ODE solver: starting from the Gaussian prior, it
        repeatedly steps ``x <- x + dt * v(x, t)`` over ``num_inference_steps``
        equal steps with ``dt = 1 / num_inference_steps``, optionally clamping
        each step to ``clip_sample_range``.

        Args:
            sample: Initial Gaussian sample, shape (B, prediction_horizon,
                action_dim).
            global_cond: Global conditioning tensor.

        Returns:
            torch.Tensor: Integrated action sequence, same shape as ``sample``.
        """
        dt = 1.0 / self.num_inference_steps
        for k in range(self.num_inference_steps):
            # Continuous time shared across the batch, ascending from 0 to 1.
            t = torch.full(
                sample.shape[:1], k * dt, dtype=torch.float32, device=sample.device
            )
            sample = sample + dt * self.unet(sample, t, global_cond=global_cond)
            if self.clip_sample:
                sample = torch.clamp(
                    sample, -self.clip_sample_range, self.clip_sample_range
                )
        return sample

    def _prepare_global_conditioning(
        self,
        joint_states: torch.FloatTensor | None,
        batched_nc_data: list[BatchedNCData],
        camera_images_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Encode image features and concatenate with the state vector.

        Args:
            joint_states: Joint state tensor, or None if no proprioception.
            batched_nc_data: List of BatchedRGBData.
            camera_images_mask: Camera image mask tensor.

        Returns:
            Global conditioning tensor.
        """
        batched_rgb_data = cast(list[BatchedRGBData], batched_nc_data)
        global_cond_feats = []
        # Only include joint states if available (allows visual-only inputs)
        if joint_states is not None:
            global_cond_feats.append(joint_states)
            batch_size = joint_states.shape[0]
        else:
            # Get batch size from image data
            batch_size = batched_rgb_data[0].frame.shape[0]

        # Extract image features.
        for cam_id, (normalizer, encoder, input_rgb) in enumerate(
            zip(self.image_normalizers, self.image_encoders, batched_rgb_data)
        ):
            last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)
            transformed = normalizer(last_frame)
            features = encoder(transformed)
            features = features * camera_images_mask[:, cam_id].view(batch_size, 1)
            global_cond_feats.append(features)

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    @staticmethod
    def _make_noise_scheduler(
        noise_scheduler_type: str, **kwargs: dict[str, Any]
    ) -> DDPMScheduler | DDIMScheduler:
        """Factory for noise scheduler instances.

        All kwargs are passed to the scheduler.

        Args:
            noise_scheduler_type: Type of scheduler to create.
            **kwargs: Additional arguments for scheduler.

        Returns:
            Noise scheduler instance.
        """
        if noise_scheduler_type == "DDPM":
            return DDPMScheduler(**kwargs)
        elif noise_scheduler_type == "DDIM":
            return DDIMScheduler(**kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {noise_scheduler_type}")

    def _build_global_cond(self, batch: BatchedInferenceInputs) -> torch.Tensor:
        """Encode observations into the UNet's global conditioning vector.

        Shared by inference, training, and real-time chunking so the image
        encoders are described in exactly one place.

        Args:
            batch: Input observations.

        Returns:
            torch.Tensor: Global conditioning with shape (B, global_cond_dim).
        """
        batch_size = len(batch)
        # Normalize and combine joint states
        joint_states = self._combine_proprio(batch)

        # Build global conditioning from images (if available) or proprio only
        if (
            DataType.RGB_IMAGES in self.input_data_types
            and DataType.RGB_IMAGES in batch.inputs
        ):
            global_cond = self._prepare_global_conditioning(
                joint_states,
                batch.inputs[DataType.RGB_IMAGES],
                batch.inputs_mask[DataType.RGB_IMAGES],
            )  # (B, global_cond_dim)
        elif DataType.RGB_IMAGES in self.input_data_types:
            # RGB configured but absent in this batch: zero-pad to full cond dim
            global_cond = torch.zeros(
                batch_size, self.global_cond_dim, device=self.device
            )
            if joint_states is not None:
                global_cond[:, : joint_states.shape[-1]] = joint_states
        else:
            global_cond = joint_states  # proprio-only model, dims already match

        return global_cond

    def _predict_done_logits(self, global_cond: torch.Tensor) -> torch.Tensor:
        """Predict done-flag logits from global conditioning.

        Args:
            global_cond: Global conditioning with shape (B, global_cond_dim).

        Returns:
            torch.Tensor: Done logits with shape (B, T, custom_1d_dim).
        """
        assert self.done_head is not None
        batch_size = global_cond.shape[0]
        logits = self.done_head(global_cond)
        return logits.view(
            batch_size, self.output_prediction_horizon, self.custom_1d_dim
        )

    def _split_done_outputs(
        self, done_probs: torch.Tensor
    ) -> list[BatchedCustom1DData]:
        """Split a concatenated done probability tensor into per-slot outputs.

        Args:
            done_probs: Probabilities with shape (B, T, custom_1d_dim).

        Returns:
            list[BatchedCustom1DData]: One entry per CUSTOM_1D embodiment slot.
        """
        batched_outputs: list[BatchedCustom1DData] = []
        start = 0
        for slot_dim in self.custom_1d_slot_dims:
            end = start + slot_dim
            batched_outputs.append(
                BatchedCustom1DData(data=done_probs[:, :, start:end])
            )
            start = end
        return batched_outputs

    def _predict_action(
        self,
        batch: BatchedInferenceInputs,
        prediction_horizon: int,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict action sequence from observations.

        Args:
            batch: Input observations
            prediction_horizon: action sequence prediction horizon
            global_cond: Optional precomputed global conditioning. When None,
                conditioning is built from ``batch``.

        Returns:
            torch.FloatTensor: Predicted action sequence with shape
            (B, prediction_horizon, action_dim)
        """
        if global_cond is None:
            global_cond = self._build_global_cond(batch)

        # run sampling
        actions = self._conditional_sample(
            len(batch),
            prediction_horizon,
            global_cond=global_cond,
        )

        return actions

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Forward pass for inference.

        Args:
            batch: Batch of inference samples.

        Returns:
            dict[DataType, list[BatchedNCData]]: Model predictions with action
            sequences and optional done-flag probabilities.
        """
        prediction_horizon = self.output_prediction_horizon
        global_cond = self._build_global_cond(batch)
        action_preds = self._predict_action(
            batch, prediction_horizon, global_cond=global_cond
        )

        # (B, T, action_dim)
        outputs = self._split_outputs(self.action_normalizer.unnormalize(action_preds))

        if self.done_head is not None:
            done_probs = torch.sigmoid(self._predict_done_logits(global_cond))
            outputs[DataType.CUSTOM_1D] = self._split_done_outputs(done_probs)

        return outputs

    def _split_outputs(
        self, predictions: torch.Tensor
    ) -> dict[DataType, list[BatchedNCData]]:
        """Split an unnormalized action tensor into per-sensor batched outputs.

        Args:
            predictions: Unnormalized actions with shape (B, T, action_dim).

        Returns:
            dict[DataType, list[BatchedNCData]]: One entry per output sensor,
            ordered by the tensor column layout in ``self.output_dims``.
        """
        output_tensors: dict[DataType, list[BatchedNCData]] = {}

        for data_type in self.action_output_data_types:
            start_idx, end_idx = self.output_dims[data_type]
            output_width = end_idx - start_idx
            dt_preds = predictions[:, :, start_idx:end_idx]  # (B, T, dt_size)

            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs = []
                for i in range(output_width):
                    joint_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(BatchedJointData(value=joint_preds))
                output_tensors[data_type] = batched_outputs
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                batched_outputs = []
                for i in range(output_width):
                    gripper_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(
                        BatchedParallelGripperOpenAmountData(open_amount=gripper_preds)
                    )
                output_tensors[data_type] = batched_outputs
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        return output_tensors

    def _diffusion_pred_target(
        self,
        target_actions: torch.Tensor,
        global_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the diffusion prediction and its regression target.

        Samples a discrete timestep per item, adds the scheduler's noise to the
        trajectory, runs the network, and returns the prediction together with
        the target selected by ``prediction_type`` ("epsilon" or "sample").

        Args:
            target_actions: Normalized action trajectory (B, T, action_dim).
            global_cond: Global conditioning tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (network prediction, target).
        """
        assert self.noise_scheduler is not None
        # Sample noise to add to the trajectory.
        eps = torch.randn(target_actions.shape, device=target_actions.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(target_actions.shape[0],),
            device=target_actions.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude
        # at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(
            target_actions, eps, timesteps
        )
        # Run the denoising network (that might denoise the trajectory, or
        # attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # The target is either the original trajectory, or the noise.
        if self.prediction_type == "epsilon":
            target = eps
        elif self.prediction_type == "sample":
            target = target_actions
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")
        return pred, target

    def _flow_matching_pred_target(
        self,
        target_actions: torch.Tensor,
        global_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the flow-matching prediction and its target velocity.

        Samples a continuous time t ~ U(0, 1) per item, forms the point
        ``x_t = (1 - t) * noise + t * action`` on the straight-line path from
        noise to data, runs the network, and returns the prediction together
        with the constant target velocity ``action - noise``.

        Args:
            target_actions: Normalized action trajectory (B, T, action_dim).
            global_cond: Global conditioning tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (predicted velocity, target
            velocity).
        """
        noise = torch.randn(target_actions.shape, device=target_actions.device)
        # Continuous timestep in [0, 1) sampled uniformly per item in the batch.
        timesteps = torch.rand(target_actions.shape[0], device=target_actions.device)
        t = timesteps[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * target_actions
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
        target = target_actions - noise
        return pred, target

    def _build_done_mask(self, batch: BatchedTrainingSamples) -> torch.Tensor:
        """Build a per-feature mask for CUSTOM_1D done targets.

        Expands the per-slot mask ``(B, num_slots)`` across each slot's feature
        width so it aligns with concatenated done targets ``(B, custom_1d_dim)``.

        Args:
            batch: Training batch.

        Returns:
            torch.Tensor: Mask with shape (B, custom_1d_dim).
        """
        if DataType.CUSTOM_1D not in batch.outputs_mask:
            return torch.zeros(batch.batch_size, self.custom_1d_dim, device=self.device)

        slot_mask = batch.outputs_mask[DataType.CUSTOM_1D]  # (B, num_slots)
        expanded: list[torch.Tensor] = []
        for slot_idx, slot_dim in enumerate(self.custom_1d_slot_dims):
            expanded.append(slot_mask[:, slot_idx : slot_idx + 1].expand(-1, slot_dim))
        return torch.cat(expanded, dim=-1)

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Adds noise to the target actions and computes the masked MSE between the
        network output and its target. When CUSTOM_1D outputs are configured,
        also computes a masked BCE loss on the done-flag classification head.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        inference_sample = BatchedInferenceInputs(
            inputs=batch.inputs,
            inputs_mask=batch.inputs_mask,
            batch_size=batch.batch_size,
        )

        global_cond = self._build_global_cond(inference_sample)

        # Concatenate continuous action targets; zero-fill types absent from batch
        action_targets = []
        for data_type in self.action_output_data_types:
            start, end = self.output_dims[data_type]
            dim = end - start
            if data_type not in batch.outputs:
                action_targets.append(
                    torch.zeros(
                        batch.batch_size,
                        self.output_prediction_horizon,
                        dim,
                        device=self.device,
                    )
                )
            elif data_type in [
                DataType.JOINT_TARGET_POSITIONS,
                DataType.JOINT_POSITIONS,
            ]:
                batched_joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend([bjd.value for bjd in batched_joints])
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData],
                    batch.outputs[data_type],
                )
                action_targets.extend([gripper.open_amount for gripper in grippers])
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        # Per-sample, per-sensor mask: correctly handles robots with different
        # sensor counts for the same type in the same batch.
        action_mask = torch.cat(
            [
                (
                    batch.outputs_mask[data_type]
                    if data_type in batch.outputs_mask
                    else torch.zeros(
                        batch.batch_size,
                        self.output_dims[data_type][1] - self.output_dims[data_type][0],
                        device=self.device,
                    )
                )
                for data_type in self.action_output_data_types
            ],
            dim=-1,
        )  # (B, max_output_size)

        action_data = torch.cat(action_targets, dim=-1)  # (B, T, total_action_dim)

        target_actions = self.action_normalizer.normalize(action_data)

        # Compute the network prediction and its regression target for the
        # selected generative process.
        if self.process_type == "flow_matching":
            pred, target = self._flow_matching_pred_target(target_actions, global_cond)
        else:
            pred, target = self._diffusion_pred_target(target_actions, global_cond)

        loss = F.mse_loss(pred, target, reduction="none")
        loss = (loss * action_mask.unsqueeze(1)).sum() / torch.clamp(
            action_mask.sum() * pred.shape[1], min=1.0
        )

        losses = {
            "mse_loss": loss,
        }
        metrics = {
            "mse_loss": loss,
        }

        if self.done_head is not None:
            done_logits = self._predict_done_logits(global_cond)  # (B, T, D)

            if DataType.CUSTOM_1D in batch.outputs:
                custom_outputs = cast(
                    list[BatchedCustom1DData], batch.outputs[DataType.CUSTOM_1D]
                )
                done_targets = torch.cat(
                    [item.data for item in custom_outputs], dim=-1
                )  # (B, T, D)
            else:
                done_targets = torch.zeros_like(done_logits)

            done_mask = self._build_done_mask(batch)  # (B, D)
            bce = F.binary_cross_entropy_with_logits(
                done_logits, done_targets.clamp(0.0, 1.0), reduction="none"
            )
            done_bce_loss = (bce * done_mask.unsqueeze(1)).sum() / torch.clamp(
                done_mask.sum() * done_logits.shape[1], min=1.0
            )
            weighted_done_loss = self.done_loss_weight * done_bce_loss
            losses["done_bce_loss"] = weighted_done_loss
            metrics["done_bce_loss"] = done_bce_loss

        return BatchedTrainingOutputs(
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(
        self,
    ) -> list[torch.optim.Optimizer]:
        """Configure optimizer with different learning rates.

        Uses separate learning rates for image encoder backbone and other
        model parameters.

        Returns:
            list[torch.optim.Optimizer]: List of optimizers for model parameters
        """
        return [
            torch.optim.AdamW(
                self.param_groups,
                weight_decay=self.weight_decay,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
            )
        ]

    def configure_schedulers(
        self,
        optimizers: list[torch.optim.Optimizer],
        num_training_steps: int,
    ) -> list[torch.optim.lr_scheduler._LRScheduler]:
        """Configure scheduler for optimizers.

        Uses diffusers scheduler with warmup steps.
        """
        from diffusers.optimization import get_scheduler

        return [
            get_scheduler(
                name=self.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.lr_scheduler_num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            for optimizer in optimizers
        ]

    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        """Get the input data types supported by this model.

        Returns:
            set[DataType]: Set of supported input data types
        """
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.RGB_IMAGES,
        }

    @staticmethod
    def get_supported_output_data_types() -> set[DataType]:
        """Get the output data types supported by this model.

        Returns:
            set[DataType]: Set of supported output data types
        """
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
            DataType.CUSTOM_1D,
        }
