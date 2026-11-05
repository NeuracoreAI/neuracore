"""π0.5: A Vision-Language-Action Flow Model for General Robot Control.

This module implements the π0.5 (Pi05) model from the Physical Intelligence
paper. π0.5 is a vision-language-action model that has a VLM from the
pretrained PaliGemma model and a flow matching action expert.

Reference: Black, Kevin, et al. "π0.5: A Vision-Language-Action Model with
Open-World Generalization." arXiv preprint
`https://arxiv.org/abs/2504.16054`.

When ``DataType.DONE`` is an output, a separate classification head also
predicts whether the task is done at the current step, from attention-pooled
SigLIP image tokens and optionally the joint and gripper state. Without DONE
the model is plain Pi05.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Literal, cast

import numpy as np
import torch
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
from neuracore_types.batched_nc_data.batched_language_data import LANGUAGE_MODEL_NAME
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizerBase

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import QuantileNormalizer
from neuracore.ml.core.neuracore_model import DEFAULT_OUTPUT_DATA_TYPE_ORDER

from .modules import PI05Policy
from .utils import PI05Config, _load_tokenizer, build_lr_lambda, pad_vector

logger = logging.getLogger(__name__)

proprio_normalizer = QuantileNormalizer
action_normalizer = QuantileNormalizer
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


class Pi05(NeuracoreModel):
    """Vision-language-action flow model for robot manipulation.

    Implements the π0.5 model from Physical Intelligence that combines a
    PaliGemma vision-language model with a Gemma action expert. The model
    uses flow matching to predict action sequences from visual observations.
    Proprioceptive state is discretized and appended to the language prompt.

    The architecture supports flexible finetuning strategies including
    action-expert-only, vision+action, or full model training.

    When ``DataType.DONE`` is an output, a lightweight MLP on attention-pooled
    SigLIP image tokens also predicts the current-step done flag. It reads
    observation features only, never the noisy actions or flow-matching time,
    so training and inference see the same inputs. ``done_use_proprio`` adds
    the current joint positions and gripper opening. ``detach_done_encoder``
    stops the done loss from updating any of those inputs.
    """

    CANONICAL_OUTPUT_DATA_TYPE_ORDER = DEFAULT_OUTPUT_DATA_TYPE_ORDER + (DataType.DONE,)

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        vlm_max_text_tokens: int = 200,
        num_inference_steps: int = 10,
        dtype: Literal["bfloat16", "float32"] = "bfloat16",
        paligemma_variant: GemmaVariant = GemmaVariant.GEMMA_2B,
        action_expert_variant: GemmaVariant = GemmaVariant.GEMMA_300M,
        use_pretrained_weights: bool = True,
        pretrained_name_or_path: str | None = "lerobot/pi05_base",
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
        finetune_vision_encoder_and_action_expert: bool = False,
        discrete_state_input: bool = True,
        done_loss_weight: float = 0.1,
        done_head_dim: int = 128,
        done_head_dropout: float = 0.5,
        detach_done_encoder: bool = False,
        done_use_proprio: bool = False,
        done_max_pos_weight: float = MAX_DONE_POS_WEIGHT,
    ):
        """Initialize the Pi05 model.

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
            finetune_vision_encoder_and_action_expert: Train vision encoder and action
                expert
            discrete_state_input: Whether to encode proprio state into prompt text
            done_loss_weight: Weight for the done-flag BCE loss relative to the
                flow-matching MSE loss
            done_head_dim: Hidden width of the done MLP. The default stays
                narrow so the head is less able to memorize training episodes.
            done_head_dropout: Dropout probability inside the done MLP.
            detach_done_encoder: When true, the done loss does not flow into
                any done-head input: the SigLIP image tokens and the joint and
                gripper state. The done modules still train. The action loss
                still updates the vision tower.
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
        self.finetune_vision_encoder_and_action_expert = (
            finetune_vision_encoder_and_action_expert
        )
        self.discrete_state_input = discrete_state_input
        self.done_loss_weight = done_loss_weight
        self.done_head_dim = done_head_dim
        self.done_head_dropout = done_head_dropout
        self.detach_done_encoder = detach_done_encoder
        self.done_use_proprio = done_use_proprio
        self.done_max_pos_weight = done_max_pos_weight

        self.prompt_tokenizer_name = "google/paligemma-3b-pt-224"
        self.prompt_tokenizer: PreTrainedTokenizerBase = _load_tokenizer(
            self.prompt_tokenizer_name
        )
        self.prompt_tokenizer.padding_side = "right"
        self.language_decode_tokenizer_name = LANGUAGE_MODEL_NAME
        self.language_decode_tokenizer: PreTrainedTokenizerBase = _load_tokenizer(
            self.language_decode_tokenizer_name
        )
        if self.prompt_tokenizer.pad_token is None:
            if self.prompt_tokenizer.eos_token is None:
                raise ValueError(
                    "Pi05 tokenizer must define a pad or eos token for prompt batching."
                )
            self.prompt_tokenizer.pad_token = self.prompt_tokenizer.eos_token

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
            if data_type not in self.input_data_types:
                continue

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
                "Pi05 requires at least one continuous action "
                "output type (joint or gripper positions); DONE alone is not "
                "supported."
            )

        # Setup continuous output data for the action expert
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
        self.done_token_score: nn.Linear | None = None
        self.done_proprio_to_logits: nn.Linear | None = None
        self.done_proprio_slices: list[tuple[int, int]] = []
        if DataType.DONE in self.output_data_types:
            if DataType.RGB_IMAGES not in self.input_data_types:
                raise ValueError(
                    "Pi05 requires RGB_IMAGES input to predict DONE flags."
                )
            done_stats = cast(
                list[DoneDataStats],
                self.output_dataset_statistics[DataType.DONE],
            )
            done_means: list[float] = []
            for done_stat in done_stats:
                slot_dim = len(done_stat.done.mean)
                self.done_slot_dims.append(slot_dim)
                self.done_dim += slot_dim
                done_means.extend(done_stat.done.mean.tolist())
            if self.done_dim == 0:
                raise ValueError(
                    "DONE is configured as an output but has zero feature "
                    "dimensions in the dataset statistics."
                )
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
            self.register_buffer("done_pos_weight", pos_weight.to(self.device))

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

        # Build Pi05 config
        self.config = PI05Config(
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
            dtype=dtype,
            chunk_size=self.output_prediction_horizon,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            discrete_state_input=self.discrete_state_input,
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
            use_adarms=(False, True),
            device=self.device,
        )

        # Core model from the reference implementation
        if self.use_pretrained_weights and self.pretrained_name_or_path:
            self.model = PI05Policy.from_pretrained(
                self.pretrained_name_or_path, config=self.config
            )
        else:
            self.model = PI05Policy(self.config)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.done_dim > 0:
            self._build_done_head()

        self._setup_optimizer_param_groups()

    def _build_done_head(self) -> None:
        """Create the done classifier on top of the SigLIP image tokens.

        Each camera's projected image tokens are attention-pooled to one
        vector, the vectors are concatenated across cameras, and an MLP maps
        them to one logit per done feature. The head is kept in float32.
        """
        num_cameras = len(self.rgb_input_statistics)
        vision_config = self.model.paligemma_with_expert.paligemma.config.vision_config
        token_dim = int(vision_config.projection_dim)
        # Score starts at zero so the token pool begins as a uniform average,
        # then learns which image patches matter for the done flag.
        self.done_token_score = nn.Linear(token_dim, 1)
        nn.init.zeros_(self.done_token_score.weight)
        nn.init.zeros_(self.done_token_score.bias)
        self.done_head = nn.Sequential(
            nn.Linear(token_dim * num_cameras, self.done_head_dim),
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
        for module in self._done_modules():
            module.to(self.device)

    def _done_modules(self) -> list[nn.Module]:
        """Return the done-head modules that exist on this model.

        Returns:
            list[nn.Module]: Empty when DONE is not an output.
        """
        return [
            module
            for module in (
                self.done_token_score,
                self.done_head,
                self.done_proprio_to_logits,
            )
            if module is not None
        ]

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the underlying Pi05 model."""
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the underlying Pi05 model."""
        self.model.gradient_checkpointing_disable()

    def _setup_optimizer_param_groups(self) -> None:
        """Setup optimizer parameter groups for the underlying Pi05 model.

        There are two logical groups: the VLM model and the action expert model.
        You can either finetune everything or just the action expert while
        freezing the VLM model. Done-head parameters, when present, are always
        trained.
        """
        # Define parameter name patterns
        ACTION_EXPERT_PARAM_NAMES = [
            "gemma_expert",
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
        VISION_ENCODER_PARAM_NAMES = ["vision_tower", "multi_modal"]

        # Determine which parameters to include
        if self.finetune_action_expert_only:
            params = [
                param
                for name, param in self.model.named_parameters()
                if any(param_name in name for param_name in ACTION_EXPERT_PARAM_NAMES)
            ]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        elif self.finetune_vision_encoder_and_action_expert:
            params = [
                param
                for name, param in self.model.named_parameters()
                if any(
                    param_name in name
                    for param_name in ACTION_EXPERT_PARAM_NAMES
                    + VISION_ENCODER_PARAM_NAMES
                )
            ]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        else:
            # Train all parameters
            self.param_groups = [{
                "params": list(self.model.parameters()),
                "lr": self.optimizer_lr,
            }]

        # The done head is new, so it always trains at the normal LR, whichever
        # part of the pretrained model is frozen.
        done_params = [
            param for module in self._done_modules() for param in module.parameters()
        ]
        if done_params:
            self.param_groups.append({"params": done_params, "lr": self.optimizer_lr})

    def _combine_proprio(
        self, batch: BatchedInferenceInputs
    ) -> torch.FloatTensor | None:
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
                elif data_type in [
                    DataType.JOINT_POSITIONS,
                    DataType.JOINT_VELOCITIES,
                    DataType.JOINT_TORQUES,
                ]:
                    batched_joint_data = cast(list[BatchedJointData], batched_nc_data)
                    proprio_data = torch.cat(
                        [bjd.value for bjd in batched_joint_data], dim=-1
                    )

                last_proprio = proprio_data[:, -1, :]  # (B, horizon, num_features)
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

        # If no proprioception data is available, return a masked zero vector
        # when state-dependent tokenization is enabled, otherwise None.
        if not proprio_list:
            if self.discrete_state_input:
                return torch.zeros(
                    (len(batch), self.max_state_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
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
        # Pad proprio to max state dim since Pi05 expects fixed-size input.
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

    def _process_language_proprio_tokens(
        self,
        batch: BatchedInferenceInputs,
        proprio: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Pi05 prompt tokens from language and optional discretized state.

        This differs from PI0, where only language tokens are used and states are
        inputs to the action expert.

        Args:
            batch: Batch of inference samples
            proprio: Normalized proprio state [B, max_state_dim] when enabled

        Returns:
            Tuple of (tokens, mask) where tokens is [B, L] token IDs
            and mask is [B, L] attention mask.
        """
        batch_size = len(batch)
        if self.discrete_state_input and proprio is None:
            raise ValueError("State is required when discrete_state_input is enabled.")
        if proprio is not None and proprio.shape[0] != batch_size:
            raise ValueError(
                "State batch size does not match input batch size. "
                f"proprio={proprio.shape[0]}, batch={batch_size}"
            )

        task_texts: list[str]
        if (
            DataType.LANGUAGE not in self.input_data_types
            or DataType.LANGUAGE not in batch.inputs
        ):
            task_texts = [""] * batch_size
        else:
            batched_language_data = cast(
                list[BatchedLanguageData], batch.inputs[DataType.LANGUAGE]
            )
            # Grab the last language group and last timestep.
            language_data = batched_language_data[-1]
            language_tokens = language_data.input_ids[:, -1, :]  # (B, L)
            language_mask = language_data.attention_mask[:, -1, :].to(dtype=torch.bool)

            task_texts = []
            for i in range(batch_size):
                valid_token_ids = language_tokens[i][language_mask[i]].detach().cpu()
                if valid_token_ids.numel() == 0:
                    task_texts.append("")
                    continue
                task_texts.append(
                    self.language_decode_tokenizer.decode(
                        valid_token_ids.tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )

        cleaned_texts = [
            text.strip().replace("_", " ").replace("\n", " ") for text in task_texts
        ]
        prompts: list[str] = []
        if self.discrete_state_input:
            if proprio is None:
                raise ValueError(
                    "State is required when discrete_state_input is enabled."
                )
            state_np = proprio.detach().to(dtype=torch.float32).cpu().numpy()
            state_np = np.clip(state_np, -1.0, 1.0)
            # Map normalized state into 256 bins [0, 255]. Using interior edges
            # avoids producing -1 for slightly out-of-range values.
            discretized_states = np.digitize(
                state_np, bins=np.linspace(-1.0, 1.0, 256 + 1)[1:-1]
            )
            for cleaned_text, discretized_state in zip(
                cleaned_texts, discretized_states, strict=True
            ):
                state_str = " ".join(map(str, discretized_state))
                prompts.append(f"Task: {cleaned_text}, State: {state_str};\nAction: ")
        else:
            prompts = [f"{cleaned_text}\n" for cleaned_text in cleaned_texts]

        tokenized = self.prompt_tokenizer(
            prompts,
            max_length=self.vlm_max_text_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(
            self.device
        )

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
            Tuple of (images, image_masks, lang_tokens, lang_masks, proprio),
            where proprio is the normalized state [B, max_state_dim] or None.
        """
        images, image_masks = self._prepare_rgb_images(batch)
        proprio = self._combine_proprio(batch)
        if self.discrete_state_input and proprio is None:
            raise ValueError("State is required for Pi05 prompt construction.")
        lang_tokens, lang_masks = self._process_language_proprio_tokens(batch, proprio)
        return images, image_masks, lang_tokens, lang_masks, proprio

    def _predict_action(
        self, batch: BatchedInferenceInputs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict action sequence and optional done logits for the given batch.

        With a done head, the image tokens are computed once and shared by the
        done head and the action sampler. The done logits come from those
        tokens only, so they are computed once and not per denoising step.

        Args:
            batch: Input batch with observations

        Returns:
            Predicted action tensor [B, chunk_size, action_dim], and done logits
            [B, 1, done_dim] or None without a done head.
        """
        images, image_masks, lang_tokens, lang_masks, proprio = (
            self._build_inputs_from_batch(batch)
        )
        image_embs = None
        done_logits = None
        if self.done_head is not None:
            with torch.no_grad():
                image_embs = self.model.embed_images(images)
                done_logits = self._predict_done_logits(
                    image_embs, image_masks, proprio
                )
        actions = self.model.sample_actions(
            images, image_masks, lang_tokens, lang_masks, image_embs=image_embs
        )
        actions = actions[:, :, : self.action_dim]  # output pad to max action dim
        return actions, done_logits

    def _attention_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool one camera's image tokens by attending over them.

        A zero score layer is a uniform average, which is where training
        starts.

        Args:
            tokens: Image tokens of shape (B, N, C), float32.

        Returns:
            torch.Tensor: Pooled features of shape (B, C).
        """
        assert self.done_token_score is not None
        scores = self.done_token_score(tokens).squeeze(-1)  # (B, N)
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bn,bnc->bc", weights, tokens)

    def _predict_done_logits(
        self,
        image_embs: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the current-step done flag from observation features.

        Each camera's SigLIP image tokens are cast to float32, attention-pooled
        and zeroed when the camera is masked out, then concatenated across
        cameras and passed through the MLP. When ``done_use_proprio`` is set,
        a linear map adds the current joint positions and gripper opening onto
        those logits. ``detach_done_encoder`` detaches every input. Nothing
        here depends on the noisy actions or the flow-matching time.

        Args:
            image_embs: Image tokens [B, N, C] per camera from the vision tower.
            image_masks: Camera masks [B] per camera.
            proprio: Normalized proprioception [B, max_state_dim], or None when
                the model has no proprioception.

        Returns:
            torch.Tensor: Done logits with shape (B, 1, done_dim).
        """
        assert self.done_head is not None
        pooled = []
        for tokens, mask in zip(image_embs, image_masks, strict=True):
            tokens = tokens.to(dtype=torch.float32)
            if self.detach_done_encoder:
                tokens = tokens.detach()
            camera_mask = mask.to(dtype=torch.float32)[:, None]
            pooled.append(self._attention_pool_tokens(tokens) * camera_mask)
        logits = self.done_head(torch.cat(pooled, dim=-1))
        if self.done_proprio_to_logits is not None:
            if proprio is None:
                source_dim = max(end for _, end in self.done_proprio_slices)
                proprio = torch.zeros(
                    logits.shape[0],
                    source_dim,
                    device=logits.device,
                    dtype=logits.dtype,
                )
            done_proprio = torch.cat(
                [proprio[:, start:end] for start, end in self.done_proprio_slices],
                dim=-1,
            ).to(dtype=logits.dtype)
            if self.detach_done_encoder:
                done_proprio = done_proprio.detach()
            logits = logits + self.done_proprio_to_logits(done_proprio)
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

    @classmethod
    def from_pretrained(
        cls,
        model_init_description: ModelInitDescription,
        pretrained_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> Pi05:
        """Load a pretrained Pi05 model while keeping the Neuracore model interface.

        By default, downloads weights from https://huggingface.co/lerobot/pi05_base
        which contains the π0.5 base model from Physical Intelligence.

        Args:
            model_init_description: Neuracore model initialization config.
            pretrained_name_or_path: HuggingFace repo id (e.g. "lerobot/pi05_base")
                or local path. Defaults to "lerobot/pi05_base".
            **kwargs: Additional arguments passed to PI05Policy.from_pretrained
                (e.g. cache_dir, force_download, token, revision).

        Returns:
            Pi05 model with loaded pretrained weights.
        """
        model = PI05Policy.from_pretrained(pretrained_name_or_path, **kwargs)
        obj = cls(model_init_description=model_init_description)
        obj.model = model
        obj.config = model.config
        return obj

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Perform inference to predict action sequence.

        Args:
            batch: Input batch with observations

        Returns:
            Dictionary mapping output data types to lists of batched predictions,
            including done-flag probabilities when DONE is an output.
        """
        self.model.eval()
        for module in self._done_modules():
            module.eval()
        self.model.gradient_checkpointing_disable()
        if self.compile_model:
            self.model.compile_model_enable()

        actions, done_logits = self._predict_action(batch)
        predictions = self.action_normalizer.unnormalize(actions)
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

        if done_logits is not None:
            output_tensors[DataType.DONE] = self._split_done_outputs(
                torch.sigmoid(done_logits)
            )

        return output_tensors

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Computes the masked flow-matching MSE loss. When DONE outputs are
        configured, also computes a masked BCE loss on the current done flag,
        plus accuracy, precision, and recall at probability 0.5. The done head
        reads the same image tokens as the action prefix, never the noisy
        actions. ``pos_weight`` comes from the dataset positive rate and is
        capped at ``done_max_pos_weight`` so rare positives cannot dominate.

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

        images, image_masks, lang_tokens, lang_masks, proprio = (
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

        # With a done head, embed the images once and share the tokens between
        # the action prefix and the done head.
        image_embs = (
            self.model.embed_images(images) if self.done_head is not None else None
        )
        mse_losses = self.model.forward(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            target_actions,
            image_embs=image_embs,
        )
        loss = (mse_losses * action_mask).sum() / torch.clamp(n_valid, min=1.0)

        losses = {"mse_loss": loss}
        metrics = {"mse_loss": loss}

        if self.done_head is not None:
            assert image_embs is not None
            done_logits = self._predict_done_logits(
                image_embs, image_masks, proprio
            )  # (B, 1, D)

            if DataType.DONE in batch.outputs:
                done_outputs = cast(list[BatchedDoneData], batch.outputs[DataType.DONE])
                done_targets = torch.cat(
                    [item.done for item in done_outputs], dim=-1
                )  # (B, T, D)
                # The first step of the chunk is the current action step.
                current_targets = done_targets[:, :1].to(dtype=done_logits.dtype)
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
