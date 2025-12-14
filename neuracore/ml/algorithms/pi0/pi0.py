"""π0 wrapper that delegates to the reference implementation.

This preserves the Neuracore-facing `Pi0` class but swaps the internal model
for the upstream `PI0Pytorch` from `modules.py`, keeping the API while
matching the maintained implementation.
"""

from __future__ import annotations

import logging
import math
from typing import cast
from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
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
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from neuracore.ml import (
    BatchedInferenceInputs,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)
from neuracore.ml.algorithm_utils.normalizer import MeanStdNormalizer

from .modules import PI0Config, PI0Policy, pad_vector, resize_with_pad_torch

logger = logging.getLogger(__name__)

PROPRIO_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer
ACTION_NORMALIZER = MeanStdNormalizer  # or MinMaxNormalizer

# Normalizers (kept for backward compatibility with the Neuracore pipeline)
JOINT_STATE_NORMALIZER = MeanStdNormalizer
ACTION_NORMALIZER = MeanStdNormalizer


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
        pretrained_name_or_path: Optional[str] = "lerobot/pi0_base",
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
        self.action_dim = self.dataset_description.joint_target_positions.max_len
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

        proprio_dim = current_dim

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
        num_rgbs = 0
        if DataType.RGB_IMAGES in self.input_data_types:
            stats = cast(
                list[CameraDataStats], self.dataset_statistics[DataType.RGB_IMAGES]
            )
            num_rgbs = len(stats)

        self.vlm_max_tokens = num_rgbs * 256 + self.vlm_max_text_tokens

        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            VLM_BACKBONE, dtype=self.dtype, attn_implementation="eager"
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            VLM_BACKBONE, padding_side="right"
        )
        self.vlm_embedding_module = self.vlm.get_input_embeddings()
        assert self.vlm_processor.tokenizer.padding_side == "right"

        # Disable finetuning of the VLM
        for param in self.vlm.parameters():
            param.requires_grad = False

        # Create a mixture of experts (MoE) model consisting of 2 experts:
        # 1. VLM expert
        # 2. Action expert
        # expert_configs = {
        #     "vlm": MoeExpertConfig(
        #         hidden_size=VLM_EXPERT_WIDTH,
        #         intermediate_size=vlm_expert_intermediate_size,
        #         head_dim=vlm_expert_head_dim,
        #         num_attention_heads=vlm_expert_num_heads,
        #         num_key_value_heads=vlm_expert_num_kv_heads,
        #     ),
        #     "action": MoeExpertConfig(
        #         hidden_size=action_expert_width,
        #         intermediate_size=action_expert_intermediate_size,
        #         head_dim=action_expert_head_dim,
        #         num_attention_heads=action_expert_num_heads,
        #         num_key_value_heads=action_expert_num_kv_heads,
        #     ),
        # }
        paligemma_config = get_gemma_config(PALIGEMMA_VARIANT)
        action_expert_config = get_gemma_config(ACTION_EXPERT_VARIANT)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, False],
            precision=self.dtype,
        )
        self.gradient_checkpointing_enabled = False
        # self.moe = GemmaMoE(moe_depth, expert_configs)
        self.action_encoder = ActionEncoder(self.action_dim, action_expert_width)
        self.time_embedding = SinusoidalPosEmb(action_expert_width)
        self.proprio_encoder = nn.Linear(proprio_dim, action_expert_width)
        self.action_decoder = nn.Linear(
            action_expert_width,
            self.action_dim,
        )

        gemma_config = self.vlm.config.text_config
        self.using_pretrained_paligemma = (
            gemma_config.intermediate_size == vlm_expert_intermediate_size
            and gemma_config.hidden_size == VLM_EXPERT_WIDTH
        )

        # Load PaliGemma weights into VLM expert
        if self.using_pretrained_paligemma:
            self._load_pretrained_vlm_weights()

        # Core model from the reference implementation
        if self.use_pretrained_weights and self.pretrained_name_or_path:
            self.model = PI0Policy.from_pretrained(
                self.pretrained_name_or_path, config=self.config
            )
        else:
            self.model = PI0Policy(self.config)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # # disable grads for VLM part of MoE if using pretrained
        # if self.using_pretrained_paligemma:
        #     for param in self.moe.get_parameters("vlm"):
        #         param.requires_grad = False

        # # Delete the language model to save memory (keep only embeddings)
        # # Note: We delete model.language_model (the actual module), not
        # # language_model (the property)
        # del self.vlm.model.language_model

        # Resize the images to 224x224
        self.image_normalizer = torch.nn.Sequential(
            T.Resize((224, 224)),
        )

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
            image = self.image_normalizer(last_frame)
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
        actions = actions[:, :, : self.action_dim]  # output pad to max action dim
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
            torch.Tensor: Mixed attention mask.
        """
        # Calculate sequence lengths for each block
        vlm_len = vlm_seq_len if vlm_seq_len is not None else self.vlm_max_tokens
        state_len = 1
        action_len = self.action_horizon
        total_seq_len = vlm_len + state_len + action_len

        # Create base mask allowing full attention within each block
        mask = torch.zeros(
            (total_seq_len, total_seq_len), device=self.device, dtype=self.dtype
        )

        # (VLM): Can only attend to itself
        mask[:vlm_len, :vlm_len] = 1

        # (State / Action): Can attend to VLM
        mask[vlm_len:, :vlm_len] = 1

        # Proprio can attend to itself and vl
        mask[vlm_len : vlm_len + state_len, : vlm_len + state_len] = 1

        action_start = vlm_len + state_len
        # Actions follow causal pattern
        for i in range(0, action_len):
            # Can attend to proprio and previous actions
            mask[action_start + i, : action_start + i + 1] = 1

        # Add batch dimension and head dimension
        mask = mask.unsqueeze(0).unsqueeze(1)
        mask = mask.expand(batch_size, 1, -1, -1)
        # Convert to attention mask format (0 for attended positions, -inf for masked)
        attention_mask = torch.where(mask == 1, 0.0, torch.finfo(self.dtype).min).to(
            self.dtype
        )
        return attention_mask

    def _create_pi0_position_ids(
        self, batch_size: int, vlm_seq_len: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Create position IDs for the Pi0 model.

        Args:
            batch_size: Size of the batch.
            vlm_seq_len: Actual VLM sequence length.

        Returns:
            dict[str, torch.Tensor]: Position IDs for VLM and action blocks.
        """
        # VLM positions: Use actual sequence length
        vlm_len = vlm_seq_len if vlm_seq_len is not None else self.vlm_max_tokens
        vlm_pos = torch.arange(1, vlm_len + 1, device=self.device).type(self.dtype)
        vlm_pos = vlm_pos.unsqueeze(0).expand(batch_size, -1)

        # State and Action positions: Sequential positions for state and action sequence
        state_action_pos = torch.arange(
            1, 1 + self.action_horizon + 1, device=self.device
        ).type(self.dtype)
        state_action_pos = state_action_pos.unsqueeze(0).expand(batch_size, -1)

        position_ids = {"vlm": vlm_pos, "action": state_action_pos}

        return position_ids

    def _forward_vlm_merged_text_images(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        language_tokens: torch.Tensor,
        language_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for merging text and images in the VLM.

        Generates the mixed image-language embeddings and padding masks.

        Args:
            images: Input images tensor.
            image_masks: Input image masks tensor.
            language_tokens: Input language tokens tensor.
            language_masks: Input language masks tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Merged text and images
                tensor, mixed padding mask.
        """
        embs = []
        pad_masks = []

        # iterate over num_cam images
        for img, img_mask in zip(images, image_masks):
            img_emb = self.vlm.model.get_image_features(img)
            img_emb = img_emb.to(dtype=self.dtype, device=self.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = (
                img_mask[:, None].expand(bsize, num_img_embs).to(device=self.device)
            )

            embs.append(img_emb)
            pad_masks.append(img_mask)

        language_embeddings = self.vlm_embedding_module(language_tokens)
        embs.append(language_embeddings)
        pad_masks.append(language_masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        return embs, pad_masks

    def _sample_fm_time(self, batch_size: int) -> torch.Tensor:
        """Sample flow matching timesteps.

        Args:
            batch_size: Size of the batch.

        Returns:
            torch.Tensor: Sampled timesteps.
        """
        z = self.flow_beta_dist.sample((batch_size,))
        t = (1 - self.flow_sig_min) * (1 - z)
        return t.to(self.device).to(self.dtype)

    def _predict_action(
        self,
        merged_text_images: torch.Tensor,
        proprio_embeds: torch.Tensor,
        action: torch.Tensor,
        t: torch.Tensor,
        vlm_seq_len: int | None = None,
        pad_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict action sequence from observations.

        Args:
            merged_text_images: Merged text and images tensor.
            proprio_embeds: Proprioceptive embeddings tensor.
            action: Action tensor.
            t: Time tensor.
            vlm_seq_len: Actual VLM Embeddings sequence length.
            pad_masks: Padding masks for the merged text and images tensor.

        Returns:
            torch.Tensor: Predicted action tensor.
        """
        batch_size = proprio_embeds.size(0)
        time_cond = self.time_embedding(t)
        # [B, H, E]
        action_embeds = self.action_encoder(action, time_cond)
        # [B, 1 + H, E]
        proprio_embeds = proprio_embeds.unsqueeze(1)  # [B, 1, E]
        proprio_action_tokens = torch.cat([proprio_embeds, action_embeds], dim=1)
        # [B, 1 + H, E]
        proprio_action_embeds = self.moe(
            hidden_states={
                "vlm": merged_text_images,
                "action": proprio_action_tokens,
            },
            expert_attention_masks=self._create_expert_attention_masks(
                batch_size, pad_masks
            ),
            mix_attention_mask=self._create_pi0_mix_attention_mask(
                batch_size, vlm_seq_len
            ),
            position_ids=self._create_pi0_position_ids(batch_size, vlm_seq_len),
        )["action"]
        # [B, H, E]
        action_embeds = proprio_action_embeds[:, 1:]
        return self.action_decoder(action_embeds)

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Forward pass for generating actions.

        Args:
            batch: Batch of inference samples.

        Returns:
            dict[DataType, list[BatchedNCData]]: Model predictions with action sequences
        """
        batch_size = len(batch)

        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("No RGB images available")

        images, image_masks = self._prepare_rgb_images(batch)
        language_tokens, language_masks = self._process_language_tokens(batch)
        merged_text_images, pad_masks = self._forward_vlm_merged_text_images(
            images, image_masks, language_tokens, language_masks
        )
        proprio_states = self._combine_proprio(batch)
        proprio_embeds = self.proprio_encoder(proprio_states)  # (B, E)

        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros(
            batch_size, device=self.device, dtype=proprio_embeds.dtype
        )  # (B,)
        action = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=proprio_embeds.dtype,
        )  # (B, H, A)
        # Get the actual sequence length from the merged embeddings
        actual_seq_len = merged_text_images.shape[1]

        for _ in range(self.num_inference_steps):
            action_vel = self._predict_action(
                merged_text_images, proprio_embeds, action, t, actual_seq_len, pad_masks
            )
            action += delta_t * action_vel
            t += delta_t

        # (B, T, action_dim)
        predictions = self.action_normalizer.unnormalize(action)

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

    # ---------------------------------------------------------------- training
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

        proprios = self._combine_proprio(inference_sample)

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

        t = self._sample_fm_time(len(batch))
        x0 = torch.randn_like(target_actions)
        x1 = target_actions
        # Calculate conditional flow
        _t = t.view(-1, 1, 1)
        psi_t = (1 - (1 - self.flow_sig_min) * _t) * x0 + _t * x1

        if DataType.RGB_IMAGES not in batch.inputs:
            raise ValueError("RGB images are required for training")

        images, image_masks = self._prepare_rgb_images(inference_sample)
        lang_tokens, lang_masks = self._process_language_tokens(inference_sample)
        merged_text_images, pad_masks = self._forward_vlm_merged_text_images(
            images, image_masks, lang_tokens, lang_masks
        )
        proprio_embeds = self.proprio_encoder(proprios)  # (B, E)
        # Get the actual sequence length from the merged embeddings
        actual_seq_len = merged_text_images.shape[1]
        v_psi = self._predict_action(
            merged_text_images, proprio_embeds, psi_t, t, actual_seq_len, pad_masks
        )
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        loss = F.mse_loss(v_psi, d_psi, reduction="none")
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

    # ----------------------------------------------------------- optim/schedule
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
