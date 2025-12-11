"""π0 wrapper that delegates to the reference implementation.

This preserves the Neuracore-facing `Pi0` class but swaps the internal model
for the upstream `PI0Pytorch/PI0Policy` from `modeling_pi0.py`, keeping the
API while matching the maintained implementation.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Optional

import torch
from neuracore_types import DataType, ModelInitDescription, ModelPrediction
from torch.optim.lr_scheduler import LambdaLR

from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import MeanStdNormalizer

from .modules import PI0Config, PI0Policy, PI0Pytorch, pad_vector, resize_with_pad_torch

logger = logging.getLogger(__name__)

# Global tokenizer for the static helper
_tokenizer = None
LANGUAGE_MODEL_NAME = "google/paligemma-3b-pt-224"

# Normalizers (kept for backward compatibility with the Neuracore pipeline)
JOINT_STATE_NORMALIZER = MeanStdNormalizer
ACTION_NORMALIZER = MeanStdNormalizer


class Pi0(NeuracoreModel):
    """Neuracore-facing wrapper around the reference PI0 implementation."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        vlm_max_text_tokens: int = 128,
        num_inference_steps: int = 10,
        flow_sig_min: float = 0.001,
        flow_alpha: float = 1.5,
        flow_beta: float = 1.0,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        lr_scheduler_warmup_steps: int = 1_000,
        lr_scheduler_num_decay_steps: int = 30_000,
        lr_scheduler_decay_lr: float = 2.5e-6,
        clip_grad_norm: float = 1.0,
        dtype: torch.dtype = torch.float32,
        joint_state_normalizer: str = "MeanStdNormalizer",
        action_normalizer: str = "MeanStdNormalizer",
    ):
        """Initialize the Neuracore Pi0 wrapper around the reference model."""
        super().__init__(model_init_description)

        if not os.environ.get("HF_TOKEN"):
            raise ValueError("Hugging Face token not found. Please set HF_TOKEN.")

        self.action_dim = self.dataset_description.joint_target_positions.max_len
        self.max_state_dim = self.max_action_dim = 32
        self.action_horizon = self.output_prediction_horizon
        self.vlm_max_text_tokens = vlm_max_text_tokens
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.flow_alpha = flow_alpha
        self.flow_beta = flow_beta
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_warmup_steps = lr_scheduler_warmup_steps
        self.lr_scheduler_num_decay_steps = lr_scheduler_num_decay_steps
        self.lr_scheduler_decay_lr = lr_scheduler_decay_lr
        self.clip_grad_norm = clip_grad_norm
        self.dtype = dtype

        # Build PI0 config mirroring the reference defaults
        self.config = PI0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
            chunk_size=self.action_horizon,
            n_action_steps=self.action_horizon,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            num_inference_steps=self.num_inference_steps,
            time_sampling_beta_alpha=self.flow_alpha,
            time_sampling_beta_beta=self.flow_beta,
            time_sampling_offset=self.flow_sig_min,
            scheduler_warmup_steps=lr_scheduler_warmup_steps,
            scheduler_decay_steps=lr_scheduler_num_decay_steps,
            scheduler_decay_lr=lr_scheduler_decay_lr,
            device=str(self.device),
        )

        # Core model/policy from the reference implementation
        self.policy = PI0Policy(self.config)
        self.model: PI0Pytorch = self.policy.model

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Setup Normalizer
        self._setup_normalizer()

    # ------------------------------------------------------------------ helpers
    def _setup_normalizer(self) -> None:
        """Setup normalization statistics for different data types."""
        joint_states = []
        actions = []

        if DataType.JOINT_POSITIONS in self.model_init_description.input_data_types:
            joint_states.append(self.dataset_description.joint_positions)
        if DataType.JOINT_VELOCITIES in self.model_init_description.input_data_types:
            joint_states.append(self.dataset_description.joint_velocities)
        if DataType.JOINT_TORQUES in self.model_init_description.input_data_types:
            joint_states.append(self.dataset_description.joint_torques)
        if (
            DataType.JOINT_TARGET_POSITIONS
            in self.model_init_description.output_data_types
        ):
            actions.append(self.dataset_description.joint_target_positions)

        self.joint_state_normalizer = JOINT_STATE_NORMALIZER(
            name="joint_states", statistics=joint_states
        )
        self.action_normalizer = ACTION_NORMALIZER(name="actions", statistics=actions)

    # ------------------------------------------------------------------ helpers
    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the underlying PI0 model."""
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the underlying PI0 model."""
        self.model.gradient_checkpointing_disable()

    def _combine_normalized_joint_states(
        self, batch: BatchedInferenceSamples
    ) -> torch.Tensor:
        state_inputs = []
        if batch.joint_positions:
            state_inputs.append(batch.joint_positions.data * batch.joint_positions.mask)
        if batch.joint_velocities:
            state_inputs.append(
                batch.joint_velocities.data * batch.joint_velocities.mask
            )
        if batch.joint_torques:
            state_inputs.append(batch.joint_torques.data * batch.joint_torques.mask)

        if not state_inputs:
            raise ValueError("No joint states available")

        joint_states = torch.cat(state_inputs, dim=-1)
        joint_states = self.joint_state_normalizer.normalize(data=joint_states)
        # Pad to the max state dim after normalization to avoid padding artifacts
        joint_states = pad_vector(joint_states, self.max_state_dim)
        return joint_states.to(self.device)

    def _prepare_rgb_images(
        self, batch: BatchedInferenceSamples
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if batch.rgb_images is None:
            raise ValueError("RGB images are required but not provided")
        images: list[torch.Tensor] = []
        image_masks: list[torch.Tensor] = []
        for cam_id in range(self.dataset_description.rgb_images.max_len):
            img = batch.rgb_images.data[:, cam_id]
            # Resize with padding to 224x224 and normalize to [-1, 1]
            img = resize_with_pad_torch(img, 224, 224)
            img = img.to(device=self.device, dtype=torch.float32)
            img = img * 2.0 - 1.0
            images.append(img)
            image_masks.append(batch.rgb_images.mask[:, cam_id].to(self.device))
        return images, image_masks

    def _process_language_tokens(
        self, batch: BatchedInferenceSamples
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        if batch.language_tokens is None:
            language_tokens = torch.zeros(
                batch_size,
                self.vlm_max_text_tokens,
                dtype=torch.long,
                device=self.device,
            )
            language_mask = torch.ones(
                batch_size, self.vlm_max_text_tokens, device=self.device
            )
        else:
            language_tokens = batch.language_tokens.data.to(torch.long).to(self.device)
            language_mask = batch.language_tokens.mask.to(self.device)
        return language_tokens, language_mask

    def _build_inputs_from_inference(
        self, batch: BatchedInferenceSamples
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        images, image_masks = self._prepare_rgb_images(batch)
        lang_tokens, lang_masks = self._process_language_tokens(batch)
        state = self._combine_normalized_joint_states(batch)
        return images, image_masks, lang_tokens, lang_masks, state

    # ---------------------------------------------------------------- inference
    def predict_action_chunk(self, batch: BatchedInferenceSamples) -> torch.Tensor:
        """Run inference to produce one chunk of actions."""
        images, image_masks, lang_tokens, lang_masks, state = (
            self._build_inputs_from_inference(batch)
        )
        actions = self.model.sample_actions(
            images, image_masks, lang_tokens, lang_masks, state
        )
        actions = actions[:, :, : self.action_dim]
        return actions

    @classmethod
    def from_pretrained(
        cls,
        model_init_description: ModelInitDescription,
        pretrained_name_or_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "Pi0":
        """Load a pretrained PI0 model while keeping the Neuracore model interface.

        By default, downloads weights from https://huggingface.co/lerobot/pi0_base
        which contains the π₀ base model from Physical Intelligence.

        Args:
            model_init_description: Neuracore model initialization config.
            pretrained_name_or_path: HuggingFace repo id (e.g. "lerobot/pi0_base")
                or local path. Defaults to "lerobot/pi0_base".
            **kwargs: Additional arguments passed to PI0Policy.from_pretrained
                (e.g. cache_dir, force_download, token, revision).

        Returns:
            Pi0 model with loaded pretrained weights.
        """
        policy = PI0Policy.from_pretrained(pretrained_name_or_path, **kwargs)
        obj = cls(model_init_description)
        obj.policy = policy
        obj.model = policy.model
        obj.config = policy.config
        return obj

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Produce a ModelPrediction given an inference batch."""
        t_start = time.time()
        actions = self.predict_action_chunk(batch)
        actions = self.action_normalizer.unnormalize(actions)
        actions = actions.detach().cpu().float().numpy()
        return ModelPrediction(
            outputs={DataType.JOINT_TARGET_POSITIONS: actions},
            prediction_time=time.time() - t_start,
        )

    # ---------------------------------------------------------------- training
    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Compute loss and metrics for one training batch."""
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
            language_tokens=batch.inputs.language_tokens,
            joint_target_positions=batch.outputs.joint_target_positions,
        )

        images, image_masks, lang_tokens, lang_masks, state = (
            self._build_inputs_from_inference(inference_sample)
        )

        if batch.outputs.joint_target_positions is None:
            raise ValueError("Joint target positions are required")

        target_actions = self.action_normalizer.normalize(
            data=batch.outputs.joint_target_positions.data
        )
        # Pad to the max action dim after normalization to avoid padding artifacts
        target_actions = pad_vector(target_actions, self.max_action_dim).to(self.device)
        target_mask = pad_vector(
            batch.outputs.joint_target_positions.mask, self.max_action_dim
        ).to(self.device)
        target_actions = target_actions * target_mask

        losses = self.model.forward(
            images, image_masks, lang_tokens, lang_masks, state, target_actions
        )
        # Mask to the real action dims
        losses = losses[:, :, : self.action_dim]
        loss = (losses * target_mask[:, :, : self.action_dim]).mean()

        losses_dict = {"mse_loss": loss}
        metrics = {"mse_loss": loss.detach()}

        return BatchedTrainingOutputs(
            output_predictions=None,
            losses=losses_dict,
            metrics=metrics,
        )

    # ----------------------------------------------------------- optim/schedule
    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create the optimizer list used during training."""
        return [
            torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        ]

    def configure_schedulers(
        self, optimizers: list[torch.optim.Optimizer], num_training_steps: int
    ) -> list[LambdaLR]:
        """Configure schedulers with automatic warmup/decay scaling."""
        actual_warmup_steps = self.lr_scheduler_warmup_steps
        actual_decay_steps = self.lr_scheduler_num_decay_steps

        if num_training_steps < self.lr_scheduler_num_decay_steps:
            scale = num_training_steps / self.lr_scheduler_num_decay_steps
            actual_warmup_steps = int(self.lr_scheduler_warmup_steps * scale)
            actual_decay_steps = num_training_steps
            logger.info(
                "Auto-scaling LR scheduler: warmup %s->%s, decay %s->%s (scale %.3f)",
                self.lr_scheduler_warmup_steps,
                actual_warmup_steps,
                self.lr_scheduler_num_decay_steps,
                actual_decay_steps,
                scale,
            )

        def lr_lambda(current_step: int) -> float:
            def linear_warmup(step: int) -> float:
                if step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay(step: int) -> float:
                step = min(step, actual_decay_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.lr_scheduler_decay_lr / self.lr
                return (1 - alpha) * cosine + alpha

            if current_step < actual_warmup_steps:
                return linear_warmup(current_step)
            return cosine_decay(current_step)

        return [LambdaLR(optimizer, lr_lambda, -1) for optimizer in optimizers]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Return supported input data types for the model."""
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
            DataType.LANGUAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Return supported output data types for the model."""
        return [DataType.JOINT_TARGET_POSITIONS]
