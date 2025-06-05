"""Diffusion Policy: Visuomotor Policy Learning via Action Diffusion."""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from diffusers.schedulers.scheduling_ddim import DDIMScheduler, DDPMScheduler

from neuracore.core.nc_types import DataType, ModelInitDescription, ModelPrediction
from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import (
    DiffusionPolicyImageEncoder,
    DiffusionConditionalUnet1d
)

logger = logging.getLogger(__name__)


class DiffusionPolicy(NeuracoreModel):
    """
    Implementation of Diffusion Policy (Visuomotor Policy Learning via Action Diffusion) model.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        hidden_dim: int = 256,
        unet_down_dims: list[int] = [512, 1024, 2048],
        unet_kernel_size: int = 5,
        unet_n_groups: int = 8,
        unet_diffusion_step_embed_dim: int = 128,
        unet_use_film_scale_modulation: bool = True,
        noise_scheduler_type: str = "DDPM",
        num_train_timesteps: int = 100,
        num_inference_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
        clip_sample: bool = True,   
        clip_sample_range: float = 1.0,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        prediction_type: str = "epsilon",
    ):
        super().__init__(model_init_description)
        self.lr = lr
        self.lr_backbone = lr_backbone
        # Vision components
        self.image_encoders = nn.ModuleList([
            DiffusionPolicyImageEncoder(feature_dim=hidden_dim)
            for _ in range(self.dataset_description.max_num_rgb_images)
        ])
        # what if there is no joint velocities or torques?
        global_cond_dim = (
            self.dataset_description.joint_positions.max_len
            + self.dataset_description.joint_velocities.max_len
            + self.dataset_description.joint_torques.max_len
            + self.image_encoders[0].feature_dim * self.dataset_description.max_num_rgb_images
        )

        self.unet = DiffusionConditionalUnet1d(
            action_dim=self.dataset_description.joint_target_positions.max_len,
            global_cond_dim=global_cond_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
            diffusion_step_embed_dim=unet_diffusion_step_embed_dim,
            use_film_scale_modulation=unet_use_film_scale_modulation,
        )

        self.noise_scheduler = _make_noise_scheduler(
            noise_scheduler_type,
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type=prediction_type,
        )
        self.prediction_type = prediction_type
        self.num_inference_steps = num_inference_steps
        # Normalize the images with imagenet mean and std
        self.image_normalizer = torch.nn.Sequential(
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        state_mean = np.concatenate([
            self.dataset_description.joint_positions.mean,
            self.dataset_description.joint_velocities.mean,
            self.dataset_description.joint_torques.mean,
        ])
        state_std = np.concatenate([
            self.dataset_description.joint_positions.std,
            self.dataset_description.joint_velocities.std,
            self.dataset_description.joint_torques.std,
        ])
        self.joint_state_mean = self._to_torch_float_tensor(state_mean)
        self.joint_state_std = self._to_torch_float_tensor(state_std)

        self.joint_target_mean = self._to_torch_float_tensor(
            self.dataset_description.joint_target_positions.mean
        )
        self.joint_target_std = self._to_torch_float_tensor(
            self.dataset_description.joint_target_positions.std
        )

    def _to_torch_float_tensor(self, data: list[float]) -> torch.FloatTensor:
        """Convert list of floats to torch tensor."""
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _preprocess_joint_state(
        self, 
        joint_state: torch.FloatTensor,
        joint_state_mean: torch.FloatTensor,
        joint_state_std: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Preprocess the states."""
        return (joint_state - joint_state_mean) / joint_state_std

    def _unnormalize_actions(
        self, predicted_actions: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Unnormalize the actions."""
        return (predicted_actions * self.joint_target_std) + self.joint_target_mean

    def _combine_joint_states(
        self, batch: BatchedInferenceSamples
    ) -> torch.FloatTensor:
        """Combine joint states."""
        joint_states = None
        if self.state_embed is not None:
            state_inputs = []
            if batch.joint_positions:
                state_inputs.append(
                    batch.joint_positions.data * batch.joint_positions.mask
                )
            if batch.joint_velocities:
                state_inputs.append(
                    batch.joint_velocities.data * batch.joint_velocities.mask
                )
            if batch.joint_torques:
                state_inputs.append(batch.joint_torques.data * batch.joint_torques.mask)
            joint_states = torch.cat(state_inputs, dim=-1)
            joint_states = self._preprocess_joint_state(joint_states, self.joint_state_mean, self.joint_state_std)
        return joint_states

    # ========= inference  ============
    def _conditional_sample(
        self, batch_size: int, prediction_horizon: int, global_cond: torch.Tensor | None = None, generator: torch.Generator | None = None
    ) -> torch.Tensor:

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, prediction_horizon, self.dataset_description.joint_target_positions.max_len),
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
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(
        self,
        states: torch.FloatTensor,
        camera_images: torch.FloatTensor,
        camera_images_mask: torch.FloatTensor,) -> torch.FloatTensor:
        """Encode image features and concatenate them all together along with the state vector."""
        global_cond_feats = self._combine_joint_states(batch)
        if camera_images:
        # Extract image features.
            for cam_id, encoder in enumerate(self.image_encoders):
                features = encoder(self.image_normalizer(camera_images[:, cam_id]))
                features *= camera_images_mask[:, cam_id].view(batch_size, 1, 1, 1)
                global_cond_feats.append(features)

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
        Factory for noise scheduler instances of the requested type. All kwargs are passed
        to the scheduler.
        """
        if name == "DDPM":
            return DDPMScheduler(**kwargs)
        elif name == "DDIM":
            return DDIMScheduler(**kwargs)
        else:
            raise ValueError(f"Unsupported noise scheduler type {name}")

    def _predict_action(
        self,
        batch: BatchedInferenceSamples,
        prediction_horizon: int,
    ) -> torch.FloatTensor:

        batch_size = len(batch)
        # Normalize and combine joint states
        joint_states = self._combine_joint_states(batch)

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(joint_states, batch.rgb_images.data, batch.rgb_images.mask)  # (B, global_cond_dim)

        # run sampling
        actions = self._conditional_sample(batch_size, prediction_horizon, global_cond=global_cond)

        return actions

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        t = time.time()
        prediction_horizon = batch.outputs.joint_target_positions.data.shape[1]
        action_preds = self._predict_action(batch, prediction_horizon)
        prediction_time = time.time() - t
        # unnormalize the actions
        predictions = self._unnormalize_actions(action_preds)
        predictions = predictions.detach().cpu().numpy() 
        return ModelPrediction(
            outputs={DataType.JOINT_TARGET_POSITIONS: predictions},
            prediction_time=prediction_time,
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Training step."""
        prediction_horizon = batch.outputs.joint_target_positions.data.shape[1]
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
            joint_target_positions=batch.outputs.joint_target_positions,
        )
        joint_states = self._combine_joint_states(inference_sample)
        global_cond = self._prepare_global_conditioning(joint_states, batch.rgb_images.data, batch.rgb_images.mask)
        target_actions = self._preprocess_joint_state(
            batch.outputs.joint_target_positions.data,
            self.joint_target_mean,
            self.joint_target_std,
        )
        target_actions = target_actions * batch.outputs.joint_target_positions.mask
        # Sample noise to add to the trajectory.
        eps = torch.randn(target_actions.shape, device=target_actions.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(target_actions.shape[0],),
            device=target_actions.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(target_actions, eps, timesteps)
        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.prediction_type == "epsilon":
            target = eps
        elif self.prediction_type == "sample":
            target = target_actions
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        losses = {
            "mse_loss": loss,
        }
        metrics = {
            "mse_loss": loss,
        }
        return BatchedTrainingOutputs(
            output_predicitons=pred,
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer with different LRs for different components."""
        backbone_params = []
        other_params = []

        for name, param in self.named_parameters():
            if "image_encoders" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": self.lr_backbone},
            {"params": other_params, "lr": self.lr},
        ]

        return [torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)]

    @staticmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Return the data types supported by the model."""
        return [DataType.JOINT_TARGET_POSITIONS]
