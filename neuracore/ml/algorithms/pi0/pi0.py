"""π0 wrapper that delegates to the reference implementation.

This preserves the Neuracore-facing `Pi0` class but swaps the internal model
for the upstream `PI0Pytorch` from `modules.py`, keeping the API while
matching the maintained implementation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal, cast

import torch
from neuracore_types import (
    BatchedJointData,
    BatchedLanguageData,
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
from torch.optim.lr_scheduler import LambdaLR

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import MeanStdNormalizer

from .gemma_pytorch import pad_vector, resize_with_pad_torch
from .modules import PI0Config, PI0Policy

logger = logging.getLogger(__name__)

PROPRIO_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer
ACTION_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer


class Pi0(NeuracoreModel):
    """Neuracore-facing wrapper around the reference PI0 implementation."""

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        vlm_max_text_tokens: int = 48,
        num_inference_steps: int = 10,
        dtype: Literal["bfloat16", "float32"] = "float32",
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        use_pretrained_weights: bool = True,
        pretrained_name_or_path: str | None = "lerobot/pi0_base",
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        gradient_checkpointing: bool = False,
        compile_model: bool = False,
        compile_mode: str = "max-autotune",
        optimizer_lr: float = 2.5e-5,
        optimizer_betas: tuple[float, float] = (0.9, 0.95),
        optimizer_eps: float = 1e-8,
        optimizer_weight_decay: float = 0.01,
        clip_grad_norm: float = 1.0,
        lr_scheduler_warmup_steps: int = 1000,
        lr_scheduler_num_decay_steps: int = 30000,
        lr_scheduler_decay_lr: float = 2.5e-6,
        finetune_action_expert_only: bool = False,
        freeze_language_model_only: bool = False,
    ):
        """Initialize the Neuracore Pi0 wrapper around the reference model."""
        super().__init__(model_init_description)

        self.max_state_dim = self.max_action_dim = 32
        self.action_horizon = self.output_prediction_horizon
        self.vlm_max_text_tokens = vlm_max_text_tokens
        self.num_inference_steps = num_inference_steps
        self.dtype = dtype
        self.time_sampling_beta_alpha = time_sampling_beta_alpha
        self.time_sampling_beta_beta = time_sampling_beta_beta
        self.time_sampling_scale = time_sampling_scale
        self.time_sampling_offset = time_sampling_offset
        self.min_period = min_period
        self.max_period = max_period
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.optimizer_lr = optimizer_lr
        self.optimizer_betas = optimizer_betas
        self.optimizer_eps = optimizer_eps
        self.optimizer_weight_decay = optimizer_weight_decay
        self.lr_scheduler_warmup_steps = lr_scheduler_warmup_steps
        self.lr_scheduler_num_decay_steps = lr_scheduler_num_decay_steps
        self.lr_scheduler_decay_lr = lr_scheduler_decay_lr
        self.use_pretrained_weights = use_pretrained_weights
        self.pretrained_name_or_path = pretrained_name_or_path
        self.finetune_action_expert_only = finetune_action_expert_only
        self.freeze_language_model_only = freeze_language_model_only
        data_stats: dict[DataType, DataItemStats] = {}
        # Track per-data-type feature sizes to preserve ordering when splitting
        self.output_slices: dict[DataType, list[int]] = {}

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
                    data_stats[data_type] = combined_stats
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

        # Setup output data
        self.max_output_size = 0
        output_stats = []

        for data_type in self.output_data_types:
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                stats = cast(list[JointDataStats], self.dataset_statistics[data_type])
                combined_stats = DataItemStats()
                output_slice_sizes = []
                for stat in stats:
                    output_slice_sizes.append(len(stat.value.mean))
                    combined_stats = combined_stats.concatenate(stat.value)
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                stats = cast(
                    list[ParallelGripperOpenAmountDataStats],
                    self.dataset_statistics[data_type],
                )
                combined_stats = DataItemStats()
                output_slice_sizes = []
                for stat in stats:
                    output_slice_sizes.append(len(stat.open_amount.mean))
                    combined_stats = combined_stats.concatenate(stat.open_amount)
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

            data_stats[data_type] = combined_stats
            output_stats.append(combined_stats)
            self.output_slices[data_type] = output_slice_sizes
            self.max_output_size += sum(output_slice_sizes)

        self.action_dim = self.max_output_size
        self.action_horizon = self.output_prediction_horizon

        # Setup normalizers
        self.proprio_normalizer = PROPRIO_NORMALIZER(
            name="proprioception", statistics=proprio_stats
        )
        self.action_normalizer = ACTION_NORMALIZER(
            name="actions", statistics=output_stats
        )

        # Setup RGB cameras
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats], self.dataset_statistics[DataType.RGB_IMAGES]
            )
            len(stats)

        # Build PI0 config
        self.config = PI0Config(
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=dtype,
            chunk_size=self.output_prediction_horizon,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            num_inference_steps=self.num_inference_steps,
            time_sampling_beta_alpha=self.time_sampling_beta_alpha,
            time_sampling_beta_beta=self.time_sampling_beta_beta,
            time_sampling_scale=self.time_sampling_scale,
            time_sampling_offset=self.time_sampling_offset,
            min_period=self.min_period,
            max_period=self.max_period,
            gradient_checkpointing=self.gradient_checkpointing,
            compile_model=self.compile_model,
            compile_mode=self.compile_mode,
            device=self.device,
        )

        # Core model from the reference implementation
        if self.use_pretrained_weights and self.pretrained_name_or_path:
            self.model = PI0Policy.from_pretrained(
                self.pretrained_name_or_path, config=self.config
            )
        else:
            self.model = PI0Policy(self.config)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._setup_optimizer_param_groups()

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the underlying PI0 model."""
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the underlying PI0 model."""
        self.model.gradient_checkpointing_disable()

    def _setup_optimizer_param_groups(self) -> None:
        """Setup optimizer parameter groups for the underlying PI0 model.

        There are two logical groups: the VLM model and the action expert model.
        You can either finetune everything or just the action expert while
        freezing the VLM model.
        """
        # Define parameter name patterns
        ACTION_EXPERT_PATTERNS = [
            "gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

        # Determine which parameters to include
        if self.finetune_action_expert_only:
            patterns = ACTION_EXPERT_PATTERNS
            params = [
                param
                for name, param in self.model.named_parameters()
                if any(pattern in name for pattern in patterns)
            ]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        else:
            # Train all parameters
            self.param_groups = [{
                "params": list(self.model.parameters()),
                "lr": self.optimizer_lr,
            }]

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
        # Pad proprio to max state dim since PI0 expects fixed-size input.
        # Pad after normalization to avoid padding artifacts.
        normalized_proprio = pad_vector(normalized_proprio, self.max_state_dim).to(
            self.device
        )

        return normalized_proprio

    def _prepare_rgb_images(
        self, batch: BatchedInferenceInputs
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prepare the RGB images and masks.

        First resize to 224x224 and then normalize values to [-1,1]. And transform
        the image dimension to (num_cams, B, C, H, W).

        Args:
            batch: Batch of inference samples.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: List of images and masks.
        """
        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("RGB images are required but not provided")

        batched_rgb_data = cast(list[BatchedRGBData], batch.inputs[DataType.RGB_IMAGES])
        camera_mask = batch.inputs_mask[DataType.RGB_IMAGES]

        images = []
        image_masks = []
        for cam_id, input_rgb in enumerate(batched_rgb_data):
            last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)
            image = resize_with_pad_torch(last_frame, 224, 224)
            # Normalize from range [0,1] to [-1,1] as expected by siglip
            image = image * 2.0 - 1.0
            images.append(image)
            image_masks.append(camera_mask[:, cam_id])

        return images, image_masks

    def _process_language_tokens(
        self,
        batch: BatchedInferenceInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the language tokens.

        Args:
            batch: Batch of inference samples.

        Returns:
            torch.Tensor: Language tokens tensor.
            torch.Tensor: Language mask tensor.
        """
        batch_size = len(batch)
        if DataType.LANGUAGE not in batch.inputs:
            # Return zero tensor with appropriate dimensions if no language input
            # Use torch.long for token IDs (embedding layer expects integer indices)
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
            batched_language_data = cast(
                list[BatchedLanguageData], batch.inputs[DataType.LANGUAGE]
            )
            # Grab the last language group and last timestep
            language_data = batched_language_data[-1]
            language_tokens = language_data.input_ids[:, -1, :]  # (B, L)
            language_mask = language_data.attention_mask[:, -1, :]  # (B, L)

        return language_tokens, language_mask

    def _build_inputs_from_batch(
        self, batch: BatchedInferenceInputs
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        images, image_masks = self._prepare_rgb_images(batch)
        lang_tokens, lang_masks = self._process_language_tokens(batch)
        proprios = self._combine_proprio(batch)
        return images, image_masks, lang_tokens, lang_masks, proprios

    def _predict_action(self, batch: BatchedInferenceInputs) -> torch.Tensor:
        """Run inference to produce one chunk of actions."""
        images, image_masks, lang_tokens, lang_masks, proprios = (
            self._build_inputs_from_batch(batch)
        )
        actions = self.model.sample_actions(
            images, image_masks, lang_tokens, lang_masks, proprios
        )
        actions = actions[:, :, : self.action_dim]  # output pad to max action dim
        return actions

    @classmethod
    def from_pretrained(
        cls,
        model_init_description: ModelInitDescription,
        pretrained_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> Pi0:
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
        model = PI0Policy.from_pretrained(pretrained_name_or_path, **kwargs)
        obj = cls(model_init_description)
        obj.model = model
        obj.config = model.config
        return obj

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Produce a ModelPrediction given an inference batch."""
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        self.model.compile_model_enable()

        actions = self._predict_action(batch)
        predictions = self.action_normalizer.unnormalize(actions)
        output_tensors: dict[DataType, list[BatchedNCData]] = {}

        # Use start_slice_idx pattern
        start_slice_idx = 0
        for data_type in self.output_data_types:
            slice_sizes = self.output_slices[data_type]
            end_slice_idx = start_slice_idx + sum(slice_sizes)
            dt_preds = predictions[
                :, :, start_slice_idx:end_slice_idx
            ]  # (B, T, dt_size)

            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs = []
                offset = 0
                for slice_size in slice_sizes:
                    joint_preds = dt_preds[
                        :, :, offset : offset + slice_size
                    ]  # (B, T, slice_size)
                    batched_outputs.append(BatchedJointData(value=joint_preds))
                    offset += slice_size
                output_tensors[data_type] = batched_outputs
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched_outputs = []
                offset = 0
                for slice_size in slice_sizes:
                    gripper_preds = dt_preds[
                        :, :, offset : offset + slice_size
                    ]  # (B, T, slice_size)
                    batched_outputs.append(
                        BatchedParallelGripperOpenAmountData(open_amount=gripper_preds)
                    )
                    offset += slice_size
                output_tensors[data_type] = batched_outputs
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

            start_slice_idx = end_slice_idx

        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

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

        images, image_masks, lang_tokens, lang_masks, proprios = (
            self._build_inputs_from_batch(inference_sample)
        )

        if set(batch.outputs.keys()) != set(self.output_data_types):
            raise ValueError(
                "Batch outputs do not match model output configuration."
                f" Expected {self.output_data_types}, got {list(batch.outputs.keys())}"
            )

        # Concatenate all output actions
        action_targets = []
        for data_type in self.output_data_types:
            expected_slices = self.output_slices[data_type]
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_joints = cast(list[BatchedJointData], batch.outputs[data_type])
                tensors = [bjd.value for bjd in batched_joints]
            elif data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData], batch.outputs[data_type]
                )
                tensors = [gripper.open_amount for gripper in grippers]
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

            if len(tensors) != len(expected_slices):
                raise ValueError(
                    f"Output count for {data_type} does not match statistics. "
                    f"Expected {len(expected_slices)}, got {len(tensors)}"
                )

            for tensor, slice_size in zip(tensors, expected_slices):
                if tensor.shape[-1] != slice_size:
                    raise ValueError(
                        f"Output dim for {data_type} mismatch stats. "
                        f"Expected {slice_size}, got {tensor.shape[-1]}"
                    )
                action_targets.append(tensor)

        action_data = torch.cat(action_targets, dim=-1)  # (B, T, total_action_dim)

        target_actions = self.action_normalizer.normalize(data=action_data)
        # Pad to the max action dim after normalization to avoid padding artifacts
        target_actions = pad_vector(target_actions, self.max_action_dim).to(self.device)

        mse_losses = self.model.forward(
            images, image_masks, lang_tokens, lang_masks, proprios, target_actions
        )
        # Mask to the real action dims
        loss = mse_losses[:, :, : self.action_dim].mean()

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

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create the optimizer list used during training."""
        return [
            torch.optim.AdamW(
                self.param_groups,
                weight_decay=self.optimizer_weight_decay,
                betas=self.optimizer_betas,
                eps=self.optimizer_eps,
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
                alpha = self.lr_scheduler_decay_lr / self.optimizer_lr
                return (1 - alpha) * cosine + alpha

            if current_step < actual_warmup_steps:
                return linear_warmup(current_step)
            return cosine_decay(current_step)

        return [LambdaLR(optimizer, lr_lambda, -1) for optimizer in optimizers]

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
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.LANGUAGE,
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
