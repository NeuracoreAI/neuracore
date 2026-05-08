"""π0.5: A Vision-Language-Action Flow Model for General Robot Control.

This module implements the π0.5 (Pi05) model from the Physical Intelligence
paper. π0.5 is a vision-language-action model that has a VLM from the
pretrained PaliGemma model and a flow matching action expert.

Reference: Black, Kevin, et al. "π0.5: A Vision-Language-Action Model with
Open-World Generalization." arXiv preprint
`https://arxiv.org/abs/2504.16054`.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import numpy as np
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
from neuracore_types.batched_nc_data.batched_language_data import LANGUAGE_MODEL_NAME
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizerBase

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import QuantileNormalizer

from .modules import PI05FullPolicy
from .utils import (
    PI05FullConfig,
    _load_tokenizer,
    build_lr_lambda,
    fast_tokenize_actions,
    load_fast_tokenizer,
    pad_vector,
    resize_with_pad_torch,
)

logger = logging.getLogger(__name__)

PROPRIO_NORMALIZER = QuantileNormalizer
ACTION_NORMALIZER = QuantileNormalizer
IMAGE_RESIZE_SHAPE = (224, 224)


class Pi05Full(NeuracoreModel):
    """Vision-language-action flow model for robot manipulation.

    Implements the π0.5 model from Physical Intelligence that combines a
    PaliGemma vision-language model with a Gemma action expert. The model
    uses flow matching to predict action sequences from visual observations.
    Proprioceptive state is discretized and appended to the language prompt.

    The architecture supports flexible finetuning strategies including
    action-expert-only, vision+action, or full model training.
    """

    CANONICAL_OUTPUT_DATA_TYPE_ORDER = (
        DataType.JOINT_TARGET_POSITIONS,
        DataType.JOINT_POSITIONS,
        DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
        DataType.SUBTASK_LANGUAGE,
    )

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        vlm_max_text_tokens: int = 200,
        num_inference_steps: int = 10,
        dtype: Literal["bfloat16", "float32"] = "float32",
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        use_pretrained_weights: bool = True,
        pretrained_name_or_path: str | None = "lerobot/pi05_base",
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
        finetune_vision_encoder_and_action_expert: bool = False,
        discrete_state_input: bool = True,
        subtask_loss_weight: float = 10.0,
        fast_token_loss_weight: float = 1.0,
        flow_matching_loss_weight: float = 1.0,
        knowledge_insulation: bool = True,
        max_subtask_tokens: int = 64,
        max_fast_tokens: int = 128,
        max_decoding_steps: int = 200,
        subtask_temperature: float = 0.0,
        fast_tokenizer_name: str = "physical-intelligence/fast",
        fast_skip_tokens: int = 2048,
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
            subtask_loss_weight: Weight for subtask cross-entropy loss
            fast_token_loss_weight: Weight for FAST action-token cross-entropy loss
            flow_matching_loss_weight: Weight for flow-matching MSE loss
            knowledge_insulation: If True, action losses cannot flow gradient into VLM
            max_subtask_tokens: Max length of subtask token segment in the prefix
            max_fast_tokens: Max length of FAST action-token segment in the prefix
            max_decoding_steps: Max number of subtask tokens generated at inference
            subtask_temperature: Sampling temperature for subtask generation
                (0 = greedy)
            fast_tokenizer_name: HF repo id for the FAST action tokenizer
            fast_skip_tokens: Number of vocab slots reserved at the tail for FAST
        """
        super().__init__(model_init_description)

        if DataType.SUBTASK_LANGUAGE not in self.input_data_types:
            raise ValueError(
                "Pi05Full requires SUBTASK_LANGUAGE in inputs. Use the Pi05 "
                "algorithm if your dataset has no subtask annotations."
            )
        if DataType.SUBTASK_LANGUAGE not in self.output_data_types:
            raise ValueError(
                "Pi05Full requires SUBTASK_LANGUAGE in outputs. Add it to your "
                "configured output data types (it is always produced by the model)."
            )

        for name, value in [
            ("subtask_loss_weight", subtask_loss_weight),
            ("fast_token_loss_weight", fast_token_loss_weight),
            ("flow_matching_loss_weight", flow_matching_loss_weight),
        ]:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if (
            subtask_loss_weight == 0.0
            and fast_token_loss_weight == 0.0
            and flow_matching_loss_weight == 0.0
        ):
            raise ValueError(
                "At least one loss weight must be > 0. All zero would yield an "
                "untrainable model."
            )

        if finetune_action_expert_only and subtask_loss_weight > 0:
            logger.warning(
                "subtask CE loss has no effect when finetune_action_expert_only "
                "is True (VLM is frozen). Set subtask_loss_weight=0 to save compute."
            )
        if finetune_action_expert_only and fast_token_loss_weight > 0:
            logger.warning(
                "fast_token CE loss has no effect when finetune_action_expert_only "
                "is True. Set fast_token_loss_weight=0 to save compute."
            )

        self.subtask_loss_weight = subtask_loss_weight
        self.fast_token_loss_weight = fast_token_loss_weight
        self.flow_matching_loss_weight = flow_matching_loss_weight
        self.knowledge_insulation = knowledge_insulation
        self.max_subtask_tokens = max_subtask_tokens
        self.max_fast_tokens = max_fast_tokens
        self.max_decoding_steps = max_decoding_steps
        self.subtask_temperature = subtask_temperature
        self.fast_tokenizer_name = fast_tokenizer_name
        self.fast_skip_tokens = fast_skip_tokens

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

        # Setup output data
        self.max_output_size = 0
        output_stats = []
        self.output_dims: dict[DataType, tuple[int, int]] = {}
        current_output_dim = 0

        for data_type in self.ordered_output_data_types:
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

        # Setup normalizers
        # Only create proprio_normalizer if there are proprioception stats
        # This allows the algorithm to work without proprioception (visual-only)
        self.proprio_normalizer = (
            PROPRIO_NORMALIZER(name="proprioception", statistics=proprio_stats)
            if proprio_stats
            else None
        )
        self.action_normalizer = ACTION_NORMALIZER(
            name="actions", statistics=output_stats
        )

        # Setup RGB cameras
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats],
                self.input_dataset_statistics[DataType.RGB_IMAGES],
            )

        # Build Pi05 config
        self.config = PI05FullConfig(
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
            subtask_loss_weight=self.subtask_loss_weight,
            fast_token_loss_weight=self.fast_token_loss_weight,
            flow_matching_loss_weight=self.flow_matching_loss_weight,
            knowledge_insulation=self.knowledge_insulation,
            max_subtask_tokens=self.max_subtask_tokens,
            max_fast_tokens=self.max_fast_tokens,
            max_decoding_steps=self.max_decoding_steps,
            subtask_temperature=self.subtask_temperature,
            fast_tokenizer_name=self.fast_tokenizer_name,
            fast_skip_tokens=self.fast_skip_tokens,
        )

        self.fast_tokenizer = load_fast_tokenizer(self.fast_tokenizer_name)

        # Core model from the reference implementation
        if self.use_pretrained_weights and self.pretrained_name_or_path:
            self.model = PI05FullPolicy.from_pretrained(
                self.pretrained_name_or_path, config=self.config
            )
        else:
            self.model = PI05FullPolicy(self.config)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._setup_optimizer_param_groups()

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
        freezing the VLM model.
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

        if self.finetune_action_expert_only:
            for name, param in self.model.named_parameters():
                param.requires_grad = any(p in name for p in ACTION_EXPERT_PARAM_NAMES)
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        elif self.finetune_vision_encoder_and_action_expert:
            allowed = ACTION_EXPERT_PARAM_NAMES + VISION_ENCODER_PARAM_NAMES
            for name, param in self.model.named_parameters():
                param.requires_grad = any(p in name for p in allowed)
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.param_groups = [{"params": params, "lr": self.optimizer_lr}]
        else:
            # Train all parameters
            self.param_groups = [{
                "params": list(self.model.parameters()),
                "lr": self.optimizer_lr,
            }]

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

        Resizes images to 224x224 and normalizes pixel values to [-1, 1]
        as expected by the SigLIP vision encoder.

        Args:
            batch: Batch of inference samples

        Returns:
            Tuple of (images, masks) where images is a list of tensors
            [B, C, H, W] per camera and masks is a list of [B] tensors.
        """
        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("RGB images are required but not provided")

        batched_rgb_data = cast(list[BatchedRGBData], batch.inputs[DataType.RGB_IMAGES])
        camera_mask = batch.inputs_mask[DataType.RGB_IMAGES]

        images = []
        image_masks = []
        for cam_id, input_rgb in enumerate(batched_rgb_data):
            last_frame = input_rgb.frame[:, -1, :, :, :]  # (B, 3, H, W)
            image = resize_with_pad_torch(last_frame, *IMAGE_RESIZE_SHAPE)
            # Normalize from range [0,1] to [-1,1] as expected by siglip
            image = image * 2.0 - 1.0
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
        if DataType.LANGUAGE not in batch.inputs:
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

    def _build_inputs_from_batch(
        self, batch: BatchedInferenceInputs
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Build model inputs from a batch of inference samples.

        Args:
            batch: Batch of inference samples

        Returns:
            Tuple of (images, image_masks, lang_tokens, lang_masks).
        """
        images, image_masks = self._prepare_rgb_images(batch)
        proprio = self._combine_proprio(batch)
        if self.discrete_state_input and proprio is None:
            raise ValueError("State is required for Pi05 prompt construction.")
        lang_tokens, lang_masks = self._process_language_proprio_tokens(batch, proprio)
        return images, image_masks, lang_tokens, lang_masks

    def _predict_action(self, batch: BatchedInferenceInputs) -> torch.Tensor:
        """Predict action sequence for the given batch.

        Args:
            batch: Input batch with observations

        Returns:
            Predicted action tensor [B, chunk_size, action_dim]
        """
        images, image_masks, lang_tokens, lang_masks = self._build_inputs_from_batch(
            batch
        )
        actions = self.model.sample_actions(
            images, image_masks, lang_tokens, lang_masks
        )
        actions = actions[:, :, : self.action_dim]  # output pad to max action dim
        return actions

    def _process_subtask_tokens(
        self, batch: BatchedInferenceInputs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract last-timestep subtask tokens, prepend BOS, pad to max_subtask_tokens.

        Args:
            batch: Inference input batch with a SUBTASK_LANGUAGE channel.

        Returns:
            tokens: (B, max_subtask_tokens) int64
            masks: (B, max_subtask_tokens) bool
        """
        if DataType.SUBTASK_LANGUAGE not in batch.inputs:
            raise ValueError(
                "Subtask channel missing from batch.inputs. Pi05Full training "
                "requires SUBTASK_LANGUAGE per batch."
            )
        items = cast(list[BatchedLanguageData], batch.inputs[DataType.SUBTASK_LANGUAGE])
        last = items[-1]
        ids = last.input_ids[:, -1, :]  # (B, L)
        attn = last.attention_mask[:, -1, :].to(dtype=torch.bool)

        bsize = ids.shape[0]
        max_len = self.max_subtask_tokens
        bos = self.prompt_tokenizer.bos_token_id
        if bos is None:
            bos = self.prompt_tokenizer.eos_token_id

        out_ids = torch.full((bsize, max_len), 0, dtype=torch.long)
        out_mask = torch.zeros(bsize, max_len, dtype=torch.bool)

        for i in range(bsize):
            valid = ids[i][attn[i]].detach().cpu().tolist()
            seq = [bos] + valid
            seq = seq[:max_len]
            out_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            out_mask[i, : len(seq)] = True

        return out_ids.to(self.device), out_mask.to(self.device)

    def _build_action_targets_and_fast_tokens(
        self, batch: BatchedTrainingSamples
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Concatenate action targets, normalize, and FAST-tokenize.

        Args:
            batch: Training batch.

        Returns:
            target_actions: (B, T, max_action_dim) float
            fast_tokens: (B, max_fast_tokens) int64
            fast_masks: (B, max_fast_tokens) bool
        """
        action_targets: list[torch.Tensor] = []
        for data_type in self.ordered_output_data_types:
            if data_type == DataType.SUBTASK_LANGUAGE:
                continue
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                joints = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend(j.value for j in joints)
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                grippers = cast(
                    list[BatchedParallelGripperOpenAmountData],
                    batch.outputs[data_type],
                )
                action_targets.extend(g.open_amount for g in grippers)
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        action_data = torch.cat(action_targets, dim=-1)
        target_actions = self.action_normalizer.normalize(data=action_data)
        target_actions = pad_vector(target_actions, self.max_action_dim).to(self.device)

        vocab_size = self.prompt_tokenizer.vocab_size
        fast_ids, fast_mask = fast_tokenize_actions(
            target_actions.detach(),
            tokenizer=self.fast_tokenizer,
            max_tokens=self.max_fast_tokens,
            skip_tokens=self.fast_skip_tokens,
            vocab_size=vocab_size,
        )
        return (
            target_actions,
            fast_ids.to(self.device),
            fast_mask.to(self.device),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_init_description: ModelInitDescription,
        pretrained_name_or_path: str | None = None,
        **kwargs: Any,
    ) -> Pi05Full:
        """Load a pretrained Pi05 model while keeping the Neuracore model interface.

        By default, downloads weights from https://huggingface.co/lerobot/pi05_base
        which contains the π0.5 base model from Physical Intelligence.

        Args:
            model_init_description: Neuracore model initialization config.
            pretrained_name_or_path: HuggingFace repo id (e.g. "lerobot/pi05_base")
                or local path. Defaults to "lerobot/pi05_base".
            **kwargs: Additional arguments passed to PI05FullPolicy.from_pretrained
                (e.g. cache_dir, force_download, token, revision).

        Returns:
            Pi05Full model with loaded pretrained weights.
        """
        model = PI05FullPolicy.from_pretrained(pretrained_name_or_path, **kwargs)
        obj = cls(model_init_description)
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
            Dictionary mapping output data types to lists of batched predictions.
        """
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        if self.compile_model:
            self.model.compile_model_enable()

        images, image_masks, lang_tokens, lang_masks = self._build_inputs_from_batch(
            batch
        )

        bos_id = self.prompt_tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.prompt_tokenizer.eos_token_id
        eos_id = self.prompt_tokenizer.eos_token_id
        loc0_id = self.prompt_tokenizer.convert_tokens_to_ids("<loc0000>")
        if loc0_id == self.prompt_tokenizer.unk_token_id:
            loc0_id = None

        subtask_tokens, subtask_masks = self.model.generate_subtask_tokens(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            bos_token_id=bos_id,
            eos_token_id=eos_id,
            loc_token_id=loc0_id,
        )

        actions = self.model.sample_actions(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
        )
        actions = actions[:, :, : self.action_dim]
        predictions = self.action_normalizer.unnormalize(actions)
        output_tensors: dict[DataType, list[BatchedNCData]] = {}

        for data_type in self.ordered_output_data_types:
            if data_type == DataType.SUBTASK_LANGUAGE:
                # Add a singleton T dimension to match (B, T, L) shape
                ids_3d = subtask_tokens.unsqueeze(1)
                mask_3d = subtask_masks.unsqueeze(1).to(dtype=torch.float32)
                output_tensors[data_type] = [
                    BatchedLanguageData(input_ids=ids_3d, attention_mask=mask_3d)
                ]
                continue
            start_idx, end_idx = self.output_dims[data_type]
            output_width = end_idx - start_idx
            dt_preds = predictions[:, :, start_idx:end_idx]  # (B, T, dt_size)

            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs: list[BatchedNCData] = []
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

        images, image_masks, lang_tokens, lang_masks = self._build_inputs_from_batch(
            inference_sample
        )

        if set(batch.outputs.keys()) != set(self.output_data_types):
            raise ValueError(
                "Batch outputs do not match model output configuration."
                f" Expected {self.output_data_types}, got {list(batch.outputs.keys())}"
            )

        subtask_tokens, subtask_masks = self._process_subtask_tokens(inference_sample)
        target_actions, fast_tokens, fast_masks = (
            self._build_action_targets_and_fast_tokens(batch)
        )

        loss_dict = self.model.forward(
            images,
            image_masks,
            lang_tokens,
            lang_masks,
            subtask_tokens,
            subtask_masks,
            fast_tokens,
            fast_masks,
            target_actions,
        )

        return BatchedTrainingOutputs(
            losses={k: v for k, v in loss_dict.items()},
            metrics={k: v.detach() for k, v in loss_dict.items()},
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
            DataType.SUBTASK_LANGUAGE,
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
            DataType.SUBTASK_LANGUAGE,
        }
