"""π0: A Vision-Language-Action Flow Model for General Robot Control.

This module implements the π0 (Pi0) model from the Physical Intelligence
paper. π0 is a vision-language-action model that has a VLM from the pretrained
PaliGemma model and a flow matching action expert. The model uses a mixture
of experts (MoE) to process the input and predict the action sequence.

Reference: Black, Kevin, et al. "π0: A Vision-Language-Action Flow Model
for General Robot Control." arXiv preprint https://arxiv.org/abs/2410.24164.

When ``DataType.DONE`` is an output, a separate classification head also
predicts whether the task is done at the current step, from attention-pooled
image embeddings and optionally the joint and gripper state. Without DONE the
model is plain π0.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuracore_types import (
    BatchedDoneData,
    BatchedJointData,
    BatchedLanguageData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    BatchedRGBData,
    CameraDataStats,
    DataItemStats,
    DataType,
    DoneDataStats,
    GPUType,
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

from .modules import PI0Policy
from .utils import PI0Config, build_lr_lambda, pad_vector

logger = logging.getLogger(__name__)

proprio_normalizer = MeanStdNormalizer  # or MinMaxNormalizer
action_normalizer = MeanStdNormalizer  # or MinMaxNormalizer
IMAGE_RESIZE_SHAPE = (224, 224)
# Done frames are rare, so BCE up-weights them by (1 - p) / p. The default
# multiplier stops here so one missed frame cannot dominate the loss. At 50 the
# weight is exact down to a ~2% positive rate.
MAX_DONE_POS_WEIGHT = 50.0

_ACTION_OUTPUT_DATA_TYPES = frozenset({
    DataType.JOINT_TARGET_POSITIONS,
    DataType.JOINT_POSITIONS,
    DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
})


class GemmaVariant(str, Enum):
    """Gemma model size. gemma_tiny is a test-scale model."""

    GEMMA_300M = "gemma_300m"
    GEMMA_2B = "gemma_2b"
    GEMMA_TINY = "gemma_tiny"


class CompileMode(str, Enum):
    """torch.compile mode."""

    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"
    MAX_AUTOTUNE_NO_CUDAGRAPHS = "max-autotune-no-cudagraphs"


class Pi0(NeuracoreModel):
    """Vision-language-action flow model for robot manipulation.

    Implements the π0 model from Physical Intelligence that combines a
    PaliGemma vision-language model with a Gemma action expert. The model
    uses flow matching to predict action sequences from visual observations,
    proprioceptive state, and optional language instructions.

    The architecture supports flexible finetuning strategies including
    action-expert-only, vision+action, or full model training.

    When ``DataType.DONE`` is an output, a lightweight MLP on attention-pooled
    image embeddings from the vision tower also predicts the current-step done
    flag. It reads observation features only, never the action expert, noisy
    actions or flow-matching time. ``done_use_proprio`` adds the current joint
    positions and gripper opening. ``detach_done_encoder`` stops the done loss
    from updating any of those inputs.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        vlm_max_text_tokens: int = 48,
        num_inference_steps: int = 10,
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        paligemma_variant: GemmaVariant = GemmaVariant.GEMMA_2B,
        action_expert_variant: GemmaVariant = GemmaVariant.GEMMA_300M,
        use_pretrained_weights: bool = True,
        pretrained_name_or_path: str | None = "lerobot/pi0_base",
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        gradient_checkpointing: bool = True,
        compile_model: bool = False,
        compile_mode: CompileMode = CompileMode.MAX_AUTOTUNE,
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
        done_loss_weight: float = 0.1,
        done_head_dim: int = 128,
        done_head_dropout: float = 0.5,
        detach_done_encoder: bool = False,
        done_use_proprio: bool = False,
        done_max_pos_weight: float = MAX_DONE_POS_WEIGHT,
    ):
        """Initialize the Pi0 model.

        Args:
            model_init_description: Model initialization parameters
            vlm_max_text_tokens: Maximum number of language tokens
            num_inference_steps: Number of Euler denoising steps
            dtype: Model precision ("bfloat16" or "float32")
            paligemma_variant: VLM size ("gemma_300m" or "gemma_2b")
            action_expert_variant: Action expert size ("gemma_300m" or "gemma_2b")
            use_pretrained_weights: Whether to load pretrained weights
            pretrained_name_or_path: HuggingFace repo id or local path
            time_sampling_beta_alpha: Alpha for beta distribution time sampling
            time_sampling_beta_beta: Beta for beta distribution time sampling
            time_sampling_scale: Scale factor for sampled time values
            time_sampling_offset: Offset added to sampled time values
            min_period: Minimum period for sinusoidal time embeddings
            max_period: Maximum period for sinusoidal time embeddings
            gradient_checkpointing: Enable gradient checkpointing
            compile_model: Enable torch.compile optimization
            compile_mode: Compilation mode for torch.compile
            optimizer_lr: Learning rate
            optimizer_betas: Adam beta parameters
            optimizer_eps: Adam epsilon
            optimizer_weight_decay: Weight decay
            clip_grad_norm: Gradient clipping norm (unused, for config compatibility)
            lr_scheduler_warmup_steps: Linear warmup steps
            lr_scheduler_num_decay_steps: Cosine decay steps
            lr_scheduler_decay_lr: Final learning rate after decay
            finetune_action_expert_only: Only train action expert parameters
            freeze_language_model_only: Freeze language model, train vision+action
            done_loss_weight: Weight for the done-flag BCE loss relative to the
                flow-matching MSE loss
            done_head_dim: Hidden width of the done MLP. The default stays
                narrow so the head is less able to memorize training episodes.
            done_head_dropout: Dropout probability inside the done MLP.
            detach_done_encoder: When true, the done loss does not flow into
                any done-head input: the image embeddings and the joint and
                gripper state. The done modules still train. The action loss
                still updates the vision tower when it is trainable.
            done_use_proprio: When true, a zero-initialized linear map adds the
                current normalized joint positions and gripper opening onto the
                done logits.
            done_max_pos_weight: Upper bound on the BCE ``pos_weight``
                ``(1 - p) / p``, where ``p`` is the dataset share of done
                frames. Raise it if done recall stays low; lower it if the
                done loss is noisy or the head predicts done too early.
        """
        super().__init__(model_init_description)

        self.max_state_dim = self.max_action_dim = 32
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
        self.done_loss_weight = done_loss_weight
        self.done_head_dim = done_head_dim
        self.done_head_dropout = done_head_dropout
        self.detach_done_encoder = detach_done_encoder
        self.done_use_proprio = done_use_proprio
        self.done_max_pos_weight = done_max_pos_weight

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

        # Continuous action outputs only (DONE is handled by the done head).
        self.action_output_data_types = [
            data_type
            for data_type in self.ordered_output_data_types
            if data_type in _ACTION_OUTPUT_DATA_TYPES
        ]
        if not self.action_output_data_types:
            raise ValueError(
                "Pi0 requires at least one continuous action "
                "output type (joint or gripper positions); DONE alone is not "
                "supported."
            )

        # Setup continuous output data for the flow-matching action expert
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
                data_stats[data_type] = combined_stats
                output_stats.append(combined_stats)
                dim = len(combined_stats.mean)
                self.output_dims[data_type] = (
                    current_output_dim,
                    current_output_dim + dim,
                )
                current_output_dim += dim
                self.max_output_size += dim
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
                data_stats[data_type] = combined_stats
                output_stats.append(combined_stats)
                dim = len(combined_stats.mean)
                self.output_dims[data_type] = (
                    current_output_dim,
                    current_output_dim + dim,
                )
                current_output_dim += dim
                self.max_output_size += dim

        self.action_dim = self.max_output_size

        # Setup DONE flag dims (classification head, not denoised)
        self.done_slot_dims: list[int] = []
        self.done_dim = 0
        self.done_head: nn.Sequential | None = None
        self.done_proprio_to_logits: nn.Linear | None = None
        self.done_proprio_slices: list[tuple[int, int]] = []
        if DataType.DONE in self.output_data_types:
            done_stats = cast(
                list[DoneDataStats],
                self.output_dataset_statistics[DataType.DONE],
            )
            done_means: list[float] = []
            for stat in done_stats:
                slot_dim = len(stat.done.mean)
                self.done_slot_dims.append(slot_dim)
                self.done_dim += slot_dim
                done_means.extend(stat.done.mean.tolist())
            if self.done_dim == 0:
                raise ValueError(
                    "DONE is configured as an output but has zero feature "
                    "dimensions in the dataset statistics."
                )
            if DataType.RGB_IMAGES not in self.input_data_types:
                raise ValueError("Pi0 requires RGB_IMAGES input to predict DONE flags.")
            # Done labels are 0/1 and rare, so the mean is the positive rate and
            # pos_weight = (1 - p) / p rebalances BCE toward the positive class.
            # Cap it at done_max_pos_weight so a very small p cannot make one
            # missed frame dominate the batch.
            pos_rate = torch.tensor(done_means, dtype=torch.float32).clamp(
                1e-4, 1.0 - 1e-4
            )
            pos_weight = ((1.0 - pos_rate) / pos_rate).clamp(
                max=self.done_max_pos_weight
            )
            self.register_buffer("done_pos_weight", pos_weight)

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

        # Track RGB camera statistics for optional image conditioning.
        self.rgb_input_statistics: list[CameraDataStats] = (
            cast(
                list[CameraDataStats],
                self.input_dataset_statistics[DataType.RGB_IMAGES],
            )
            if DataType.RGB_IMAGES in self.input_data_types
            else []
        )

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

        if self.done_dim > 0:
            self._build_done_head()

        self._setup_optimizer_param_groups()

    def _build_done_head(self) -> None:
        """Create the done-flag modules.

        The head pools each camera's image embeddings, concatenates the
        cameras and maps them through a small MLP. Its modules live on this
        class, not on ``self.model``, so they stay float32 and are absent
        from checkpoints trained without DONE.
        """
        # Width of the projected image tokens that enter the language model.
        paligemma_config = self.model.paligemma_with_expert.paligemma.config
        image_emb_dim = int(paligemma_config.vision_config.projection_dim)
        num_cameras = len(self.rgb_input_statistics)
        self.vision_cond_dim = image_emb_dim * num_cameras
        if self.vision_cond_dim == 0:
            raise ValueError("Pi0 requires RGB_IMAGES input to predict DONE flags.")
        # Score starts at zero so the token pool begins as a uniform average,
        # then learns which image tokens matter for the done flag.
        self.done_token_score = nn.Linear(image_emb_dim, 1)
        nn.init.zeros_(self.done_token_score.weight)
        nn.init.zeros_(self.done_token_score.bias)
        self.done_head = nn.Sequential(
            nn.Linear(self.vision_cond_dim, self.done_head_dim),
            nn.ReLU(),
            nn.Dropout(self.done_head_dropout),
            nn.Linear(self.done_head_dim, self.done_dim),
        )
        if self.done_use_proprio:
            self.done_proprio_slices = [
                self.proprio_dims[data_type]
                for data_type in (
                    DataType.JOINT_POSITIONS,
                    DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
                )
                if data_type in self.proprio_dims
            ]
            done_proprio_dim = sum(
                end - start for start, end in self.done_proprio_slices
            )
            if done_proprio_dim == 0:
                raise ValueError(
                    "done_use_proprio requires JOINT_POSITIONS or "
                    "PARALLEL_GRIPPER_OPEN_AMOUNTS inputs."
                )
            # Zero weights keep the first step identical to the vision head.
            # Joint and gripper values are added onto the logits, not mixed
            # inside the vision MLP.
            self.done_proprio_to_logits = nn.Linear(done_proprio_dim, self.done_dim)
            nn.init.zeros_(self.done_proprio_to_logits.weight)
            nn.init.zeros_(self.done_proprio_to_logits.bias)
        self.done_token_score.to(self.device)
        self.done_head.to(self.device)
        if self.done_proprio_to_logits is not None:
            self.done_proprio_to_logits.to(self.device)

    def _done_parameters(self) -> list[nn.Parameter]:
        """Return the done-head parameters, or an empty list without a head.

        Returns:
            list[nn.Parameter]: Parameters of every done module.
        """
        if self.done_head is None:
            return []
        modules: list[nn.Module] = [self.done_token_score, self.done_head]
        if self.done_proprio_to_logits is not None:
            modules.append(self.done_proprio_to_logits)
        return [param for module in modules for param in module.parameters()]

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
        freezing the VLM model. The done head, when present, is always trained
        with the same learning rate.
        """
        # Define parameter name patterns
        ACTION_EXPERT_PARAM_NAMES = [
            "gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        VISION_ENCODER_PARAM_NAMES = ["vision_tower", "multi_modal"]

        # Determine which parameters to include
        if self.finetune_action_expert_only:
            params = [
                param
                for name, param in self.model.named_parameters()
                if any(param_name in name for param_name in ACTION_EXPERT_PARAM_NAMES)
            ]
        elif self.freeze_language_model_only:
            params = [
                param
                for name, param in self.model.named_parameters()
                if any(
                    param_name in name
                    for param_name in ACTION_EXPERT_PARAM_NAMES
                    + VISION_ENCODER_PARAM_NAMES
                )
            ]
        else:
            # Train all parameters
            params = list(self.model.parameters())
        params.extend(self._done_parameters())
        self.param_groups = [{"params": params, "lr": self.optimizer_lr}]

    def _combine_proprio(self, batch: BatchedInferenceInputs) -> torch.FloatTensor:
        """Combine and normalize proprioceptive state data.

        Concatenates joint positions, velocities, torques, and gripper states
        into a single normalized state vector padded to max_state_dim.

        Args:
            batch: Input batch containing joint state data

        Returns:
            Combined and normalized state tensor [B, max_state_dim], or None
            if no proprioceptive data is available.
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

            if data_type in batch.inputs:
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
            else:
                start_idx, end_idx = self.proprio_dims[data_type]
                dim = end_idx - start_idx
                masked_proprio = torch.zeros(
                    (len(batch), dim),
                    dtype=torch.float32,
                    device=self.device,
                )
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
        # Pad proprio to max state dim since PI0 expects fixed-size input.
        # Pad after normalization to avoid padding artifacts.
        normalized_proprio = pad_vector(normalized_proprio, self.max_state_dim).to(
            self.device
        )

        return normalized_proprio

    def _prepare_rgb_images(
        self, batch: BatchedInferenceInputs
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prepare RGB images for the vision encoder.

        Normalizes pixel values to [-1, 1] as expected by the SigLIP vision
        encoder. Spatial resizing is handled upstream by preprocessing config.

        Args:
            batch: Batch of inference samples

        Returns:
            Tuple of (images, masks) where images is a list of tensors
            [B, C, H, W] per camera and masks is a list of [B] tensors.
        """
        # self.input_data_types: types this model was configured to accept (set at
        # init). batch.inputs: types actually present in this specific inference
        # call. A type can be configured but absent at runtime (e.g. RGB-only
        # training with no-camera inference), so both checks are needed.
        if DataType.RGB_IMAGES not in self.input_data_types:
            return [], []
        if DataType.RGB_IMAGES not in batch.inputs:
            batch_size = len(batch)
            black = torch.full(
                (batch_size, 3, *IMAGE_RESIZE_SHAPE),
                -1.0,
                dtype=torch.float32,
                device=self.device,
            )
            num_cameras = len(self.rgb_input_statistics)
            return [black] * num_cameras, [
                torch.zeros(batch_size, device=self.device)
            ] * num_cameras

        batched_rgb_data = cast(list[BatchedRGBData], batch.inputs[DataType.RGB_IMAGES])
        camera_mask = batch.inputs_mask[DataType.RGB_IMAGES]

        images = []
        image_masks = []
        for cam_id, input_rgb in enumerate(batched_rgb_data):
            last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)
            # Normalize from range [0,1] to [-1,1] as expected by siglip
            image = last_frame * 2.0 - 1.0
            images.append(image)
            image_masks.append(camera_mask[:, cam_id])

        return images, image_masks

    def _process_language_tokens(
        self,
        batch: BatchedInferenceInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract language tokens and attention masks from batch.

        Args:
            batch: Batch of inference samples

        Returns:
            Tuple of (tokens, mask) where tokens is [B, L] token IDs
            and mask is [B, L] attention mask.
        """
        batch_size = len(batch)
        if (
            DataType.LANGUAGE not in self.input_data_types
            or DataType.LANGUAGE not in batch.inputs
        ):
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

    def _build_inputs_from_batch(self, batch: BatchedInferenceInputs) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Build model inputs from a batch of inference samples.

        Args:
            batch: Batch of inference samples

        Returns:
            Tuple of (images, image_masks, lang_tokens, lang_masks, proprios).
            Proprios can be None if no proprioception inputs are available.
        """
        images, image_masks = self._prepare_rgb_images(batch)
        lang_tokens, lang_masks = self._process_language_tokens(batch)
        proprios = self._combine_proprio(batch)
        return images, image_masks, lang_tokens, lang_masks, proprios

    def _predict_action(self, batch: BatchedInferenceInputs) -> torch.Tensor:
        """Predict action sequence for the given batch.

        Args:
            batch: Input batch with observations

        Returns:
            Predicted action tensor [B, chunk_size, action_dim]
        """
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
        obj = cls(model_init_description=model_init_description)
        obj.model = model
        obj.config = model.config
        return obj

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Perform inference to predict action sequence.

        With a done head, the images are embedded once; the embeddings feed
        both the action sampler and the done head, and the done probability
        for the current step is added under ``DataType.DONE``.

        Args:
            batch: Input batch with observations

        Returns:
            Dictionary mapping output data types to lists of batched predictions.
        """
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        if self.compile_model:
            self.model.compile_model_enable()

        if self.done_head is None:
            actions = self._predict_action(batch)
            return self._split_outputs(self.action_normalizer.unnormalize(actions))

        # Match self.model.eval() above so the done dropout is off at inference.
        self.done_head.eval()
        images, image_masks, lang_tokens, lang_masks, proprios = (
            self._build_inputs_from_batch(batch)
        )
        # Embed the images once and share them between the action sampler and
        # the done head. Done is computed once, not per denoising step.
        with torch.no_grad():
            image_embs = self.model.embed_images(images)
            done_logits = self._predict_done_logits(image_embs, image_masks, proprios)
        actions = self.model.sample_actions(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            proprios,
            image_embs=image_embs,
        )
        actions = actions[:, :, : self.action_dim]  # output pad to max action dim
        outputs = self._split_outputs(self.action_normalizer.unnormalize(actions))
        outputs[DataType.DONE] = self._split_done_outputs(torch.sigmoid(done_logits))
        return outputs

    def _split_outputs(
        self, predictions: torch.Tensor
    ) -> dict[DataType, list[BatchedNCData]]:
        """Split an unnormalized action tensor into per-sensor batched outputs.

        Args:
            predictions: Unnormalized actions with shape (B, T, action_dim).

        Returns:
            dict[DataType, list[BatchedNCData]]: One entry per action output
            sensor, ordered by the tensor column layout in ``self.output_dims``.
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
                raise ValueError(f"Unsupported action output data type: {data_type}")

        return output_tensors

    def _attention_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool one camera's image tokens by attending over them.

        A zero score layer is a uniform average, which is where training
        starts.

        Args:
            tokens: Image token embeddings of shape (B, N, width).

        Returns:
            torch.Tensor: Pooled features of shape (B, width).
        """
        scores = self.done_token_score(tokens).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bn,bnc->bc", weights, tokens)

    def _predict_done_logits(
        self,
        image_embs: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        proprio_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the current-step done flag from observation features.

        Each camera's image embeddings are attention-pooled and concatenated;
        a masked camera contributes a zero vector. When ``done_use_proprio``
        is set, a linear map adds the current joint positions and gripper
        opening onto the logits. ``detach_done_encoder`` detaches the image
        embeddings and that state.

        Args:
            image_embs: Image token embeddings (B, N, width) per camera, as
                they enter the language model prefix.
            image_masks: Camera masks (B,) per camera.
            proprio_state: Normalized, padded proprioception (B, max_state_dim),
                or None when the model has no proprioception.

        Returns:
            torch.Tensor: Done logits with shape (B, 1, done_dim).
        """
        assert self.done_head is not None
        head_dtype = self.done_token_score.weight.dtype
        pooled = []
        for tokens, mask in zip(image_embs, image_masks, strict=True):
            # The vision tower may run in bfloat16; the head runs in float32.
            tokens = tokens.to(head_dtype)
            if self.detach_done_encoder:
                tokens = tokens.detach()
            camera_mask = mask.to(head_dtype)[:, None]
            pooled.append(self._attention_pool_tokens(tokens) * camera_mask)
        vision_cond = torch.cat(pooled, dim=-1)  # (B, width * num_cameras)

        logits = self.done_head(vision_cond)
        if self.done_proprio_to_logits is not None:
            if proprio_state is None:
                source_dim = max(end for _, end in self.done_proprio_slices)
                proprio_state = torch.zeros(
                    vision_cond.shape[0],
                    source_dim,
                    device=vision_cond.device,
                    dtype=vision_cond.dtype,
                )
            proprio = torch.cat(
                [
                    proprio_state[:, start:end]
                    for start, end in self.done_proprio_slices
                ],
                dim=-1,
            ).to(head_dtype)
            if self.detach_done_encoder:
                proprio = proprio.detach()
            logits = logits + self.done_proprio_to_logits(proprio)
        return logits.unsqueeze(1)

    def _split_done_outputs(self, done_probs: torch.Tensor) -> list[BatchedDoneData]:
        """Split a concatenated done probability tensor into per-slot outputs.

        Args:
            done_probs: Probabilities with shape (B, T, done_dim).

        Returns:
            list[BatchedDoneData]: One entry per DONE embodiment slot.
        """
        batched_outputs: list[BatchedDoneData] = []
        start = 0
        for slot_dim in self.done_slot_dims:
            end = start + slot_dim
            batched_outputs.append(BatchedDoneData(done=done_probs[:, :, start:end]))
            start = end
        return batched_outputs

    def _build_done_mask(self, batch: BatchedTrainingSamples) -> torch.Tensor:
        """Build a per-feature mask for DONE targets.

        Expands the per-slot mask ``(B, num_slots)`` across each slot's feature
        width so it aligns with concatenated done targets ``(B, done_dim)``.

        Args:
            batch: Training batch.

        Returns:
            torch.Tensor: Mask with shape (B, done_dim).
        """
        if DataType.DONE not in batch.outputs_mask:
            return torch.zeros(batch.batch_size, self.done_dim, device=self.device)

        slot_mask = batch.outputs_mask[DataType.DONE]  # (B, num_slots)
        expanded: list[torch.Tensor] = []
        for slot_idx, slot_dim in enumerate(self.done_slot_dims):
            expanded.append(slot_mask[:, slot_idx : slot_idx + 1].expand(-1, slot_dim))
        return torch.cat(expanded, dim=-1)

    @staticmethod
    def _done_classification_metrics(
        done_logits: torch.Tensor,
        done_targets: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score the current-step done flag at probability 0.5.

        A logit at or above zero is a positive prediction. Masked features
        are left out of every count. An empty denominator is reported as
        zero: no predicted positives gives precision 0, and no labeled
        positives gives recall 0.

        Args:
            done_logits: Logits with shape (B, 1, D).
            done_targets: Targets with shape (B, 1, D), in ``[0, 1]``.
            done_mask: Feature mask with shape (B, D).

        Returns:
            dict[str, torch.Tensor]: ``done_accuracy``, ``done_precision``,
            and ``done_recall``.
        """
        targets = done_targets.clamp(0.0, 1.0)
        predicted_positive = (done_logits >= 0).to(dtype=targets.dtype)
        actual_positive = (targets >= 0.5).to(dtype=targets.dtype)
        mask = done_mask.unsqueeze(1).to(dtype=targets.dtype)

        true_positive = (predicted_positive * actual_positive * mask).sum()
        predicted_positive_count = (predicted_positive * mask).sum()
        actual_positive_count = (actual_positive * mask).sum()
        correct = (
            (predicted_positive == actual_positive).to(dtype=targets.dtype) * mask
        ).sum()
        evaluated = mask.sum()
        return {
            "done_accuracy": correct / torch.clamp(evaluated, min=1.0),
            "done_precision": true_positive
            / torch.clamp(predicted_positive_count, min=1.0),
            "done_recall": true_positive / torch.clamp(actual_positive_count, min=1.0),
        }

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Computes the masked flow-matching MSE loss. When DONE outputs are
        configured, also computes a masked BCE loss on the current done flag
        from the image embeddings (never the noisy actions or flow time), plus
        accuracy, precision, and recall at probability 0.5. ``pos_weight``
        comes from the dataset positive rate and is capped at
        ``done_max_pos_weight`` so rare positives cannot dominate.

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

        # Concatenate all output actions; zero-fill types absent from this batch
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
                    list[BatchedParallelGripperOpenAmountData], batch.outputs[data_type]
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
        )  # (B, action_dim)
        # Pad to max_action_dim to match padded target/prediction tensors
        action_mask = torch.nn.functional.pad(
            action_mask, (0, self.max_action_dim - action_mask.shape[-1])
        )  # (B, max_action_dim)
        n_valid = action_mask.sum() * self.output_prediction_horizon
        action_mask = action_mask.unsqueeze(1).expand(
            -1, self.output_prediction_horizon, -1
        )  # (B, T, max_action_dim)

        action_data = torch.cat(action_targets, dim=-1)  # (B, T, total_action_dim)

        target_actions = self.action_normalizer.normalize(data=action_data)
        # Pad to the max action dim after normalization to avoid padding artifacts
        target_actions = pad_vector(target_actions, self.max_action_dim).to(self.device)

        # Embed images once for both the action expert and the done head.
        image_embs = (
            self.model.embed_images(images) if self.done_head is not None else None
        )
        mse_losses = self.model.forward(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            proprios,
            target_actions,
            image_embs=image_embs,
        )
        loss = (mse_losses * action_mask).sum() / torch.clamp(n_valid, min=1.0)

        losses = {"mse_loss": loss}
        metrics = {"mse_loss": loss}

        if self.done_head is not None:
            assert image_embs is not None
            done_logits = self._predict_done_logits(
                image_embs, image_masks, proprios
            )  # (B, 1, D)

            if DataType.DONE in batch.outputs:
                done_outputs = cast(list[BatchedDoneData], batch.outputs[DataType.DONE])
                done_targets = torch.cat(
                    [item.done for item in done_outputs], dim=-1
                )  # (B, T, D)
                # The first step of the chunk is the current action step.
                current_targets = done_targets[:, :1].to(done_logits.dtype)
            else:
                current_targets = torch.zeros_like(done_logits)

            done_mask = self._build_done_mask(batch)  # (B, D)
            bce = F.binary_cross_entropy_with_logits(
                done_logits,
                current_targets.clamp(0.0, 1.0),
                reduction="none",
                pos_weight=self.done_pos_weight,
            )
            done_bce_loss = (bce * done_mask.unsqueeze(1)).sum() / torch.clamp(
                done_mask.sum(), min=1.0
            )
            losses["done_bce_loss"] = self.done_loss_weight * done_bce_loss
            metrics["done_bce_loss"] = done_bce_loss
            metrics.update(
                self._done_classification_metrics(
                    done_logits, current_targets, done_mask
                )
            )

        return BatchedTrainingOutputs(
            losses=losses,
            metrics=metrics,
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer for training.

        Returns:
            List containing a single AdamW optimizer.
        """
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
        """Configure learning rate schedulers.

        Creates schedulers with linear warmup and cosine decay. Automatically
        scales warmup and decay periods if training steps are fewer than
        configured decay steps.

        Args:
            optimizers: List of optimizers to create schedulers for
            num_training_steps: Total number of training steps

        Returns:
            List of LambdaLR schedulers, one per optimizer.
        """
        actual_warmup_steps = self.lr_scheduler_warmup_steps
        actual_decay_steps = self.lr_scheduler_num_decay_steps

        # Auto-scale warmup and decay steps if training steps are fewer than
        # configured decay steps
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

        lr_lambda = build_lr_lambda(
            actual_warmup_steps=actual_warmup_steps,
            actual_decay_steps=actual_decay_steps,
            decay_lr=self.lr_scheduler_decay_lr,
            optimizer_lr=self.optimizer_lr,
        )

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
            DataType.LANGUAGE,
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
            DataType.DONE,
        }

    @staticmethod
    def get_supported_gpus() -> frozenset[GPUType]:
        """Get the GPU types recommended for training this model."""
        return frozenset({
            GPUType.NVIDIA_A100_80GB,
        })
