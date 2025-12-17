"""Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."""

import logging
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from neuracore_types import (
    BatchedJointData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    BatchedRGBData,
    CameraDataStats,
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

from .modules import DiffusionConditionalUnet1d, DiffusionPolicyImageEncoder

logger = logging.getLogger(__name__)

PROPRIO_NORMALIZER = MinMaxNormalizer  # or MeanStdNormalizer
ACTION_NORMALIZER = MinMaxNormalizer  # or MeanStdNormalizer
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]


class DiffusionPolicy(NeuracoreModel):
    """Implementation of Diffusion Policy for visuomotor policy learning.

    This implements the Diffusion Policy model for Visuomotor Policy Learning
    via Action Diffusion as described in the original paper.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 256,
        unet_down_dims: tuple[int, ...] = (
            512,
            1024,
            2048,
        ),
        unet_kernel_size: int = 5,
        unet_n_groups: int = 8,
        unet_diffusion_step_embed_dim: int = 128,
        spatial_softmax_num_keypoints: int = 32,
        unet_use_film_scale_modulation: bool = True,
        use_pretrained_weights: bool = True,
        use_resnet_stats: bool = True,
        noise_scheduler_type: str = "DDPM",
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        lr: float = 1e-4,
        freeze_backbone: bool = False,
        lr_backbone: float = 1e-4,
        weight_decay: float = 1e-6,
        optimizer_betas: tuple[float, float] = (0.9, 0.999),
        optimizer_eps: float = 1e-8,
        prediction_type: str = "epsilon",
        lr_scheduler_type: str = "cosine",
        lr_scheduler_num_warmup_steps: int = 500,
    ):
        """Initialize the Diffusion Policy model.

        Args:
            model_init_description: Model initialization configuration.
            hidden_dim: Hidden dimension for image encoders.
            unet_down_dims: Downsampling dimensions for UNet.
            unet_kernel_size: Kernel size for UNet convolutions.
            unet_n_groups: Number of groups for group normalization.
            unet_diffusion_step_embed_dim: Dimension of diffusion step embeddings.
            spatial_softmax_num_keypoints: Number of keypoints for spatial softmax.
            unet_use_film_scale_modulation: Whether to use FiLM scale modulation.
            use_pretrained_weights: Whether to load pretrained ResNet weights.
            use_resnet_stats: Whether to use ResNet normalization statistics.
            noise_scheduler_type: Type of noise scheduler ("DDPM" or "DDIM").
            num_train_timesteps: Number of timesteps for training.
            num_inference_steps: Number of timesteps for inference.
            beta_start: Starting beta value for noise schedule.
            beta_end: Ending beta value for noise schedule.
            beta_schedule: Beta schedule type.
            clip_sample: Whether to clip samples.
            clip_sample_range: Range for clipping samples.
            lr: Learning rate for main parameters.
            freeze_backbone: Whether to freeze image encoder backbone
            lr_backbone: Learning rate for backbone parameters.
            weight_decay: Weight decay for optimization.
            optimizer_betas: Betas for optimizer.
            optimizer_eps: Epsilon for optimizer.
            prediction_type: Type of prediction ("epsilon" or "sample").
            lr_scheduler_type: Type of the learning rate scheduler
                ("cosine", "linear", etc.).
            lr_scheduler_num_warmup_steps: Number of warmup steps for the scheduler.
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
            if data_type in self.data_types:
                if data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                    stats = cast(
                        list[ParallelGripperOpenAmountDataStats],
                        self.dataset_statistics[data_type],
                    )
                    combined_stats = DataItemStats()
                    for stat in stats:
                        combined_stats = combined_stats.concatenate(stat.open_amount)
                else:
                    stats = cast(
                        list[JointDataStats], self.dataset_statistics[data_type]
                    )
                    combined_stats = DataItemStats()
                    for stat in stats:
                        combined_stats = combined_stats.concatenate(stat.value)
                    data_stats[data_type] = combined_stats

                if data_type in self.input_data_types:
                    proprio_stats.append(combined_stats)
                    dim = len(combined_stats.mean)
                    self.proprio_dims[data_type] = (current_dim, current_dim + dim)
                    current_dim += dim

        global_cond_dim = current_dim

        # Setup output data
        self.max_output_size = 0
        output_stats = []
        self.output_dims: dict[DataType, tuple[int, int]] = {}
        current_output_dim = 0

        for data_type in self.output_data_types:
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                stats = cast(list[JointDataStats], self.dataset_statistics[data_type])
                combined_stats = DataItemStats()
                for stat in stats:
                    combined_stats = combined_stats.concatenate(stat.value)
                data_stats[data_type] = combined_stats
                output_stats.append(combined_stats)
                dim = len(combined_stats.mean)
                self.output_dims[data_type] = (
                    current_output_dim,
                    current_output_dim + dim,
                )
                current_output_dim += dim
                self.max_output_size += dim
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                stats = cast(
                    list[ParallelGripperOpenAmountDataStats],
                    self.dataset_statistics[data_type],
                )
                combined_stats = DataItemStats()
                for stat in stats:
                    combined_stats = combined_stats.concatenate(stat.open_amount)
                data_stats[data_type] = combined_stats
                output_stats.append(combined_stats)
                dim = len(combined_stats.mean)
                self.output_dims[data_type] = (
                    current_output_dim,
                    current_output_dim + dim,
                )
                current_output_dim += dim
                self.max_output_size += dim

        # Setup normalizers
        self.proprio_normalizer = PROPRIO_NORMALIZER(
            name="proprioception", statistics=proprio_stats
        )
        self.action_normalizer = ACTION_NORMALIZER(
            name="actions", statistics=output_stats
        )

        # Vision components
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats], self.dataset_statistics[DataType.RGB_IMAGES]
            )
            max_cameras = len(stats)
            self.image_encoders = nn.ModuleList()
            for i in range(max_cameras):
                if use_resnet_stats:
                    mean, std = RESNET_MEAN, RESNET_STD
                else:
                    mean_c_h_w, std_c_h_w = stats[i].frame.mean, stats[i].frame.std
                    mean = mean_c_h_w.mean(axis=(1, 2)).tolist()
                    std = std_c_h_w.mean(axis=(1, 2)).tolist()

                encoder = nn.ModuleDict({
                    "transform": torch.nn.Sequential(
                        T.Resize((224, 224)),
                        T.Normalize(mean=mean, std=std),
                    ),
                    "encoder": DiffusionPolicyImageEncoder(
                        feature_dim=hidden_dim,
                        spatial_softmax_num_keypoints=spatial_softmax_num_keypoints,
                        use_pretrained_weights=use_pretrained_weights,
                    ),
                })
                self.image_encoders.append(encoder)

            global_cond_dim += (
                self.image_encoders[0]["encoder"].feature_dim * max_cameras
            )

        self.unet = DiffusionConditionalUnet1d(
            action_dim=self.max_output_size,
            global_cond_dim=global_cond_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
            diffusion_step_embed_dim=unet_diffusion_step_embed_dim,
            use_film_scale_modulation=unet_use_film_scale_modulation,
        )

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
        """Combine different types of joint state data.

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
            if data_type not in batch.inputs:
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

        if not proprio_list:
            raise ValueError("No joint states available")

        # Concatenate all proprio together: (B, total_proprio_dim)
        all_proprio = torch.cat(proprio_list, dim=-1)

        # Normalize once on all proprio
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

    def _prepare_global_conditioning(
        self,
        joint_states: torch.FloatTensor,
        batched_nc_data: list[BatchedNCData],
        camera_images_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Encode image features and concatenate with the state vector.

        Args:
            joint_states: Joint state tensor.
            batched_nc_data: List of BatchedRGBData.
            camera_images_mask: Camera image mask tensor.

        Returns:
            Global conditioning tensor.
        """
        batched_rgb_data = cast(list[BatchedRGBData], batched_nc_data)
        global_cond_feats = [joint_states]
        batch_size = joint_states.shape[0]

        # Extract image features.
        for cam_id, (encoder_dict, input_rgb) in enumerate(
            zip(self.image_encoders, batched_rgb_data)
        ):
            last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)
            transformed = encoder_dict["transform"](last_frame)
            features = encoder_dict["encoder"](transformed)
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

    def _predict_action(
        self,
        batch: BatchedInferenceInputs,
        prediction_horizon: int,
    ) -> torch.Tensor:
        """Predict action sequence from observations.

        Args:
            batch: Input observations
            prediction_horizon: action sequence prediction horizon

        Returns:
            torch.FloatTensor: Predicted action sequence with shape
            (B, prediction_horizon, action_dim)
        """
        batch_size = len(batch)
        # Normalize and combine joint states
        joint_states = self._combine_proprio(batch)

        # Encode image features and concatenate them all together along
        # with the state vector.
        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("Failed to find RGB images")
        global_cond = self._prepare_global_conditioning(
            joint_states,
            batch.inputs[DataType.RGB_IMAGES],
            batch.inputs_mask[DataType.RGB_IMAGES],
        )  # (B, global_cond_dim)

        # run sampling
        actions = self._conditional_sample(
            batch_size, prediction_horizon, global_cond=global_cond
        )

        return actions

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Forward pass for inference.

        Args:
            batch: Batch of inference samples.

        Returns:
            dict[DataType, list[BatchedNCData]]: Model predictions with action sequences
        """
        prediction_horizon = self.output_prediction_horizon
        action_preds = self._predict_action(batch, prediction_horizon)

        # (B, T, action_dim)
        predictions = self.action_normalizer.unnormalize(action_preds)

        output_tensors: dict[DataType, list[BatchedNCData]] = {}

        for data_type in self.output_data_types:
            start_idx, end_idx = self.output_dims[data_type]
            dt_preds = predictions[:, :, start_idx:end_idx]  # (B, T, dt_size)

            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs = []
                for i in range(len(self.dataset_statistics[data_type])):
                    joint_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(BatchedJointData(value=joint_preds))
                output_tensors[data_type] = batched_outputs
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched_outputs = []
                for i in range(len(self.dataset_statistics[data_type])):
                    gripper_preds = dt_preds[:, :, i : i + 1]  # (B, T, 1)
                    batched_outputs.append(
                        BatchedParallelGripperOpenAmountData(open_amount=gripper_preds)
                    )
                output_tensors[data_type] = batched_outputs
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Given certain timesteps, add corresponding noise to the target actions, and
        predict the added noise or the target actions, and computes mse loss.

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

        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("Failed to find RGB images")

        joint_states = self._combine_proprio(inference_sample)
        global_cond = self._prepare_global_conditioning(
            joint_states,
            batch.inputs[DataType.RGB_IMAGES],
            batch.inputs_mask[DataType.RGB_IMAGES],
        )

        if set(batch.outputs.keys()) != set(self.output_data_types):
            raise ValueError(
                "Batch outputs do not match model output configuration."
                f" Expected {self.output_data_types}, got {list(batch.outputs.keys())}"
            )

        # Concatenate all output actions
        action_targets = []
        for data_type in self.output_data_types:
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend([bjd.value for bjd in batched_joints])
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData], batch.outputs[data_type]
                )
                action_targets.extend([gripper.open_amount for gripper in grippers])
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)  # (B, T, total_action_dim)

        target_actions = self.action_normalizer.normalize(action_data)
        target_actions = target_actions

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

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.prediction_type == "epsilon":
            target = eps
        elif self.prediction_type == "sample":
            target = target_actions
        else:
            raise ValueError(f"Unsupported prediction type {self.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        # Apply mask and reduce
        loss = loss.mean()

        losses = {
            "mse_loss": loss,
        }
        metrics = {
            "mse_loss": loss,
        }
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
            DataType.JOINT_TARGET_POSITIONS,
            DataType.JOINT_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        }
