"""GR00T N1.6: Vision-Language-Action model for general robot control.

This module implements the GR00T N1.6 VLA model from Nvidia adapted for the Neuracore
framework. The architecture consists of:

  1. Cosmos-Reason-2B VLM backbone (Eagle) for visual-language understanding
  2. MLP connector (LayerNorm) between VLM and DiT
  3. 32-layer Diffusion Transformer (DiT) action head
  4. Universal MLPs for state/action encoding (replaces per-embodiment MLPs)

The model uses rectified flow matching for action generation:
  - Training: linear interpolation between noise and target, predict velocity
  - Inference: 4-step Euler integration from noise to clean actions

Actions can be predicted as either state-relative deltas (default) or absolute
positions, controlled by the ``use_relative_deltas`` parameter. When using
deltas, predictions are converted to absolute positions at the output boundary.
Universal MLPs with fixed-dim padding replace
GR00T's per-embodiment MLP registry, allowing any robot to work without
embodiment-specific configuration. Action tokens also receive a learned
action-step positional embedding so the model can distinguish earlier and
later steps within each predicted action chunk.

Reference: NVIDIA GR00T N1.6 technical report. https://research.nvidia.com/labs/gear/gr00t-n1_6/.

"""

from __future__ import annotations

import inspect
import logging
import math
import warnings
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuracore_types import (
    BatchedJointData,
    BatchedLanguageData,
    BatchedNCData,
    BatchedParallelGripperOpenAmountData,
    BatchedRGBData,
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

from .image_processor import SIGLIP2_IMAGE_SIZE, GrootImageProcessor
from .modules import (
    DiTActionHead,
    MLPConnector,
    TinyVLMBackbone,
    VLMBackbone,
    add_action_step_positional_encoding,
)
from .utils import load_pretrained_state_dict

logger = logging.getLogger(__name__)

# Normalizer types for proprioception and actions
PROPRIO_NORMALIZER = MeanStdNormalizer
ACTION_NORMALIZER = MeanStdNormalizer


class Groot(NeuracoreModel):
    """GR00T N1.6 VLA model adapted for Neuracore.

    Architecture: Cosmos-Reason-2B VLM + 32-layer DiT action head.
    Uses rectified flow matching for action generation.
    Supports two action representations via ``use_relative_deltas``:
      - True (default): predict state-relative deltas, convert to absolute
        positions at the output boundary.
      - False: predict and output absolute positions directly.
    Uses fixed-dimension universal MLPs with padding instead of per-embodiment
    MLPs, plus a learned action-step positional embedding over the action
    chunk.

    Fine-tuning is controlled per-component via ``finetune_state_action_projector``,
    ``finetune_action_expert``, ``finetune_vision_encoder``, and
    ``finetune_top_llm_layer``. By default only the projector MLPs and DiT are
    trained; VLM components are frozen.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        model_path: str = "nvidia/GR00T-N1.6-3B",
        num_denoising_steps: int = 4,
        max_state_dim: int = 64,
        max_action_dim: int = 64,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        dit_lr_scale: float = 0.1,
        finetune_vision_encoder: bool = True,
        finetune_state_action_projector: bool = True,
        finetune_action_expert: bool = True,
        finetune_top_llm_layer: int = 0,
        vlm_lr_scale: float = 0.01,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        num_timestep_buckets: int = 1000,
        warmup_ratio: float = 0.05,
        use_pretrained_weights: bool = True,
        use_tiny_vlm: bool = False,
        num_dit_layers: int = 32,
        num_attention_heads: int = 32,
        attention_head_dim: int = 48,
        dit_output_dim: int = 1024,
        backbone_embedding_dim: int = 2048,
        gradient_checkpointing: bool = False,
        use_torch_compile: bool = False,
        torch_compile_mode: str = "default",
        state_dropout_prob: float = 0.0,
        state_noise_scale: float = 0.0,
        load_backbone_bf16: bool = False,
        use_relative_deltas: bool = True,
        add_action_step_positional_encoding: bool = True,
        max_action_pos_embeddings: int = 1024,
        use_alternate_vl_dit: bool = True,
        attend_text_every_n_blocks: int = 2,
    ):
        """Initialize the GR00T N1.6 model.

        Args:
            model_init_description: Neuracore model initialization config
                containing dataset statistics and data type specifications.
            model_path: HuggingFace model ID or local path for the pretrained
                GR00T N1.6 checkpoint.
            num_denoising_steps: Number of Euler steps during inference. More
                steps improve quality but increase latency.
            max_state_dim: Maximum state vector dimension. Robot state is
                zero-padded to this size for the universal MLPs.
            max_action_dim: Maximum action vector dimension. Actions are
                zero-padded to this size for the universal MLPs.
            lr: Base learning rate for universal MLPs.
            weight_decay: AdamW weight decay. Matches Isaac-GR00T's default
                of 1e-5.
            dit_lr_scale: Learning rate multiplier for the pretrained DiT.
                DiT trains at lr * dit_lr_scale.
            finetune_vision_encoder: Unfreeze the VLM visual encoder (vision_model +
                image projector) during training. Recommended only when your
                visual domain differs significantly from pretraining. Increases
                VRAM usage and overfitting risk.
            finetune_state_action_projector: Train the action head projector MLPs
                (proprio_mlp, action_encoder_mlp, action_decoder_mlp,
                mlp_connector). Enabled by default.
            finetune_action_expert: Train the DiT action head. Enabled by
                default.
            finetune_top_llm_layer: Unfreeze the top N LLM transformer layers.
                Set to 0 to keep the LLM fully frozen. Default is 0; NVIDIA
                uses 4 in N1.6 pretraining.
            vlm_lr_scale: Learning rate multiplier for VLM components when
                finetune_vision_encoder or finetune_top_llm_layer are enabled.
                VLM trains at lr * vlm_lr_scale.
            noise_beta_alpha: Alpha parameter for beta distribution time
                sampling during training.
            noise_beta_beta: Beta parameter for beta distribution time sampling
                during training.
            noise_s: Scale factor for sampled time values. Time is sampled as
                (1 - beta_sample) * noise_s.
            num_timestep_buckets: Number of discrete timestep bins for DiT
                conditioning. Continuous time is discretized into this many
                buckets.
            warmup_ratio: Fraction of total training steps used for linear
                warmup of the learning rate.
            use_pretrained_weights: Whether to load pretrained weights from
                model_path. When False, all modules are initialized with
                random weights using the provided architecture hyperparameters.
            use_tiny_vlm: When True (and use_pretrained_weights=False), use a
                lightweight VLM stub instead of the full Eagle architecture.
                Intended for unit testing to avoid HuggingFace downloads.
            num_dit_layers: Number of DiT transformer layers.
            num_attention_heads: Number of attention heads in the DiT.
            attention_head_dim: Dimension per attention head in the DiT.
            dit_output_dim: DiT output projection dimension.
            backbone_embedding_dim: VLM backbone embedding dimension.
            gradient_checkpointing: Enable gradient checkpointing on the DiT
                to reduce VRAM at the cost of ~30% extra compute per step.
                Recommended when training with large batches or limited GPU
                memory.
            use_torch_compile: Compile the DiT with torch.compile() for
                faster iteration. Requires PyTorch >= 2.0. Adds a one-time
                compilation overhead on the first forward pass.
            torch_compile_mode: torch.compile mode string. "default" is safe
                for training; "max-autotune" maximizes kernel fusion for
                inference but has a longer compile time.
            state_dropout_prob: Probability of replacing the entire proprio
                token with a learned mask token during training. Encourages
                the policy to handle missing/noisy observations.
            state_noise_scale: Std of Gaussian noise added to proprio tokens
                during training. Improves robustness to sensor noise.
            load_backbone_bf16: Cast the frozen VLM backbone to bfloat16
                after loading to halve its activation memory. This is
                automatically disabled when any VLM component is trainable,
                matching upstream GR00T fine-tuning behavior.
            use_relative_deltas: When True (default), the model predicts
                state-relative deltas and converts to absolute positions by
                adding the current state at the output. When False, the model
                predicts absolute positions directly without delta conversion.
            add_action_step_positional_encoding: Whether to add a learned
                action-step positional embedding to each action token before
                the DiT. This restores chunk-position information present in
                the original GR00T action head.
            max_action_pos_embeddings: Size of the learned action-step
                positional embedding table. Must be at least the action
                prediction horizon.
            use_alternate_vl_dit: Whether DiT cross-attention blocks should
                alternate between non-image and image VLM tokens, matching
                upstream GR00T N1.6's AlternateVLDiT.
            attend_text_every_n_blocks: Alternation cadence for the
                AlternateVLDiT token schedule.
        """
        super().__init__(model_init_description)
        self.model_path = model_path
        self.num_denoising_steps = num_denoising_steps
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.dit_lr_scale = dit_lr_scale
        self.finetune_vision_encoder = finetune_vision_encoder
        self.finetune_state_action_projector = finetune_state_action_projector
        self.finetune_action_expert = finetune_action_expert
        self.finetune_top_llm_layer = finetune_top_llm_layer
        self.vlm_lr_scale = vlm_lr_scale
        self.noise_beta_alpha = noise_beta_alpha
        self.noise_beta_beta = noise_beta_beta
        self.noise_s = noise_s
        self.num_timestep_buckets = num_timestep_buckets
        self.warmup_ratio = warmup_ratio
        self.use_tiny_vlm = use_tiny_vlm
        self.use_relative_deltas = use_relative_deltas
        self.add_action_step_positional_encoding = add_action_step_positional_encoding
        self.max_action_pos_embeddings = max_action_pos_embeddings
        self.use_alternate_vl_dit = use_alternate_vl_dit
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.load_backbone_bf16 = load_backbone_bf16

        if finetune_vision_encoder or finetune_top_llm_layer > 0:
            parts = []
            if finetune_vision_encoder:
                parts.append("visual encoder")
            if finetune_top_llm_layer > 0:
                parts.append(f"top {finetune_top_llm_layer} LLM layers")
            warnings.warn(
                f"VLM components enabled for finetuning: {', '.join(parts)}. "
                "This increases overfitting risk and VRAM usage. "
                "NVIDIA recommends: (1) at least 500 demonstrations, "
                "(2) stronger image augmentation, (3) state regularization, "
                "and (4) optionally co-training with pretraining data. "
                "Keep VLM frozen if your visual/language domain is similar "
                "to standard robotics scenes.",
                UserWarning,
                stacklevel=2,
            )

        # Build input proprioceptive data
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

        self.actual_state_dim = current_dim
        assert self.actual_state_dim > 0, (
            "GR00T N1.6 requires at least one proprioceptive input data type "
            "(JOINT_POSITIONS, JOINT_VELOCITIES, JOINT_TORQUES, or "
            "PARALLEL_GRIPPER_OPEN_AMOUNTS)."
        )
        assert self.actual_state_dim <= max_state_dim, (
            f"Proprioceptive state dim {self.actual_state_dim} exceeds "
            f"max_state_dim={max_state_dim}"
        )

        # Build output data
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
                continue

            output_stats.append(combined_stats)
            dim = len(combined_stats.mean)
            self.output_dims[data_type] = (current_output_dim, current_output_dim + dim)
            current_output_dim += dim
            self.max_output_size += dim

        self.actual_action_dim = self.max_output_size
        assert (
            self.actual_action_dim > 0
        ), "GR00T N1.6 requires at least one output data type."
        assert self.actual_action_dim <= max_action_dim, (
            f"Action dim {self.actual_action_dim} exceeds "
            f"max_action_dim={max_action_dim}"
        )

        self.proprio_normalizer = PROPRIO_NORMALIZER(
            name="proprioception", statistics=proprio_stats
        )
        self.action_normalizer = ACTION_NORMALIZER(
            name="actions", statistics=output_stats
        )
        if use_pretrained_weights:
            logger.info("Loading pretrained GR00T N1.6 from %s", model_path)
            self.vlm_backbone = VLMBackbone.from_pretrained(model_path)
            self.mlp_connector = MLPConnector.from_pretrained(model_path)
            self.dit = DiTActionHead.from_pretrained(
                model_path,
                use_alternate_vl_dit=use_alternate_vl_dit,
                attend_text_every_n_blocks=attend_text_every_n_blocks,
            )
            self.image_processor = GrootImageProcessor.from_pretrained(model_path)
        else:
            logger.info(
                "Creating GR00T N1.6 with random-init weights: "
                "dit_layers=%d, heads=%d, head_dim=%d, bb_dim=%d",
                num_dit_layers,
                num_attention_heads,
                attention_head_dim,
                backbone_embedding_dim,
            )
            if self.use_tiny_vlm:
                self.vlm_backbone = TinyVLMBackbone(
                    backbone_embedding_dim=backbone_embedding_dim,
                )
            else:
                self.vlm_backbone = VLMBackbone.from_random_init(model_path)
            self.mlp_connector = MLPConnector(embedding_dim=backbone_embedding_dim)
            self.dit = DiTActionHead(
                num_layers=num_dit_layers,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                output_dim=dit_output_dim,
                cross_attention_dim=backbone_embedding_dim,
                dropout=0.0,
                num_timestep_buckets=num_timestep_buckets,
                use_alternate_vl_dit=use_alternate_vl_dit,
                attend_text_every_n_blocks=attend_text_every_n_blocks,
            )
            self.image_processor = GrootImageProcessor()

        if gradient_checkpointing:
            self.dit.gradient_checkpointing = True
            if (
                not self.use_tiny_vlm
                and hasattr(self.vlm_backbone, "model")
                and self.vlm_backbone.model is not None
                and hasattr(self.vlm_backbone.model, "gradient_checkpointing_enable")
            ):
                self.vlm_backbone.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        if use_torch_compile:
            self.dit = torch.compile(self.dit, mode=torch_compile_mode)

        if (
            not self.use_tiny_vlm
            and hasattr(self.vlm_backbone, "model")
            and self.vlm_backbone.model is not None
        ):
            if self.load_backbone_bf16:
                self.vlm_backbone.model.to(torch.bfloat16)
                logger.info("VLM backbone cast to bfloat16")
            else:
                # Pretrained weights may already be bf16 on disk; cast the
                # entire backbone to fp32 so that vision encoder outputs and
                # LLM embeddings share the same dtype.
                self.vlm_backbone.model.to(torch.float32)
        # Let cuDNN auto-tune convolution algorithms for fixed input shapes.
        # Gives a small free speedup for the SigLip2 vision encoder.
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        # Create universal MLP adaptors for state and action. This is used for
        # cross-embodiment training and inference. With this, we can project
        # different robot states and actions to the same embedding space.
        input_embedding_dim = (
            self.dit.input_dim
        )  # (num_heads * head_dim, e.g., 32 * 48 = 1536)
        dit_output_dim = self.dit.output_dim  # DiT's output projection dim (e.g., 1024)

        if (
            self.add_action_step_positional_encoding
            and self.output_prediction_horizon > self.max_action_pos_embeddings
        ):
            raise ValueError(
                "output_prediction_horizon="
                f"{self.output_prediction_horizon} exceeds "
                f"max_action_pos_embeddings={self.max_action_pos_embeddings}"
            )

        # State encoder: (B, max_state_dim) -> (B, input_embedding_dim)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(max_state_dim, input_embedding_dim),
            nn.GELU(),
            nn.Linear(input_embedding_dim, input_embedding_dim),
        )

        # Action encoder: (B, H, max_action_dim) -> (B, H, input_embedding_dim)
        self.action_encoder_mlp = nn.Sequential(
            nn.Linear(max_action_dim, input_embedding_dim),
            nn.GELU(),
            nn.Linear(input_embedding_dim, input_embedding_dim),
        )
        if self.add_action_step_positional_encoding:
            self.action_position_embedding = nn.Embedding(
                self.max_action_pos_embeddings, input_embedding_dim
            )
        else:
            self.action_position_embedding = None

        # Action decoder: (B, H, dit_output_dim) -> (B, H, max_action_dim)
        self.action_decoder_mlp = nn.Sequential(
            nn.Linear(dit_output_dim, dit_output_dim),
            nn.GELU(),
            nn.Linear(dit_output_dim, max_action_dim),
        )
        if use_pretrained_weights and self.action_position_embedding is not None:
            self._load_action_position_embedding(model_path)

        # Beta distribution for time sampling during training
        self._beta_dist = torch.distributions.Beta(noise_beta_alpha, noise_beta_beta)

        # State augmentation (training only)
        self.state_dropout_prob = state_dropout_prob
        self.state_noise_scale = state_noise_scale
        if state_dropout_prob > 0.0:
            self.state_mask_token = nn.Parameter(
                0.02 * torch.randn(1, input_embedding_dim)
            )
        else:
            self.state_mask_token = None

    def _augment_proprio_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply state dropout and noise to proprio tokens (training only).

        Args:
            tokens: Proprio embedding (B, input_embedding_dim).

        Returns:
            Augmented tokens with the same shape.
        """
        if self.state_dropout_prob > 0.0 and self.state_mask_token is not None:
            # Randomly replace entire proprio token with learned mask token
            do_drop = (
                (
                    torch.rand(tokens.shape[0], device=tokens.device)
                    < self.state_dropout_prob
                )
                .float()
                .unsqueeze(1)
            )  # (B, 1)
            tokens = tokens * (1 - do_drop) + self.state_mask_token * do_drop

        if self.state_noise_scale > 0.0:
            tokens = tokens + torch.randn_like(tokens) * self.state_noise_scale

        return tokens

    def _load_action_position_embedding(self, model_path: str) -> None:
        """Load GR00T's learned action-step positional embedding if available."""
        if self.action_position_embedding is None:
            return

        full_state_dict = load_pretrained_state_dict(model_path)
        checkpoint_key = "action_head.position_embedding.weight"
        checkpoint_weight = full_state_dict.get(checkpoint_key)
        if checkpoint_weight is None:
            logger.warning(
                "No pretrained action-step positional embedding found at %s; "
                "using random initialization.",
                checkpoint_key,
            )
            return

        target_weight = self.action_position_embedding.weight
        if checkpoint_weight.shape[1] != target_weight.shape[1]:
            logger.warning(
                "Skipping pretrained action-step positional embedding due to "
                "shape mismatch: checkpoint=%s model=%s",
                tuple(checkpoint_weight.shape),
                tuple(target_weight.shape),
            )
            return

        rows_to_copy = min(checkpoint_weight.shape[0], target_weight.shape[0])
        with torch.no_grad():
            target_weight[:rows_to_copy].copy_(
                checkpoint_weight[:rows_to_copy].to(
                    device=target_weight.device,
                    dtype=target_weight.dtype,
                )
            )

        if rows_to_copy < target_weight.shape[0]:
            logger.info(
                "Loaded %d/%d action-step positional embedding rows from the "
                "checkpoint; kept the remaining rows randomly initialized.",
                rows_to_copy,
                target_weight.shape[0],
            )

    def _set_frozen_modules_eval(self) -> None:
        """Force frozen sub-modules into eval mode.

        The training loop calls model.train() before each step, which puts
        every sub-module into train mode — including frozen ones. This enables
        dropout inside the frozen VLM, which corrupts its outputs. Calling
        this method at the start of every forward/training_step restores eval
        mode for all modules that have no trainable parameters.
        """
        if not self.training:
            return
        # Freeze visual encoder sub-modules when not being tuned
        if (
            not self.finetune_vision_encoder
            and not self.use_tiny_vlm
            and hasattr(self.vlm_backbone, "model")
            and self.vlm_backbone.model is not None
        ):
            self.vlm_backbone.model.vision_model.eval()
            if hasattr(self.vlm_backbone.model, "mlp1"):
                self.vlm_backbone.model.mlp1.eval()
        # Freeze LLM when not tuning any top layers
        if (
            self.finetune_top_llm_layer == 0
            and not self.use_tiny_vlm
            and hasattr(self.vlm_backbone, "model")
            and self.vlm_backbone.model is not None
        ):
            self.vlm_backbone.model.language_model.eval()
        if not self.finetune_state_action_projector:
            self.mlp_connector.eval()

    def _combine_proprio(self, batch: BatchedInferenceInputs) -> torch.Tensor:
        """Combine, normalize, and pad proprioceptive state.

        Args:
            batch: Input batch containing proprioceptive observations.

        Returns:
            Normalized and padded state tensor (B, max_state_dim).
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
            mask = batch.inputs_mask[data_type]
            if data_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched = cast(
                    list[BatchedParallelGripperOpenAmountData],
                    batch.inputs[data_type],
                )
                data = torch.cat([b.open_amount for b in batched], dim=-1)
            else:
                batched = cast(list[BatchedJointData], batch.inputs[data_type])
                data = torch.cat([b.value for b in batched], dim=-1)
            proprio_list.append(data[:, -1, :] * mask)  # last timestep

        all_proprio = torch.cat(proprio_list, dim=-1)  # (B, actual_state_dim)
        normalized = self.proprio_normalizer.normalize(all_proprio)
        return self._pad_state(normalized)  # (B, max_state_dim)

    def _prepare_vlm_inputs(
        self, batch: BatchedInferenceInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare pixel_values, input_ids, and attention_mask for the VLM.

        Args:
            batch: Input batch containing RGB_IMAGES and optionally LANGUAGE.

        Returns:
            Tuple of (pixel_values, input_ids, attention_mask).
        """
        if DataType.RGB_IMAGES in batch.inputs:
            rgb_data = cast(list[BatchedRGBData], batch.inputs[DataType.RGB_IMAGES])
            num_cameras = len(rgb_data)
            pixel_values = self.image_processor.preprocess_images(rgb_data)
        else:
            # No camera input: substitute a single black image so the VLM/DiT
            # pipeline keeps consistent shapes. Conditioning then comes purely
            # from language + proprio. Value -1.0 matches SigLip2 normalization
            # of a zero pixel: (0 - 0.5) / 0.5 = -1.
            num_cameras = 1
            pixel_values = torch.full(
                (len(batch) * num_cameras, 3, SIGLIP2_IMAGE_SIZE, SIGLIP2_IMAGE_SIZE),
                -1.0,
                device=self.device,
            )

        language_data = None
        if DataType.LANGUAGE in batch.inputs:
            language_data = cast(
                list[BatchedLanguageData], batch.inputs[DataType.LANGUAGE]
            )

        pixel_values, input_ids, attn_mask = self.image_processor.construct_vlm_inputs(
            pixel_values=pixel_values,
            language_data=language_data,
            batch_size=len(batch),
            num_cameras=num_cameras,
            device=self.device,
        )

        return pixel_values, input_ids, attn_mask

    def _build_current_output_state(
        self, batch: BatchedInferenceInputs
    ) -> torch.Tensor:
        """Extract the current value for each output type from the inputs.

        Used as the base for state-relative delta computation. For each
        output data type, the corresponding input type provides the current
        value (e.g. JOINT_TARGET_POSITIONS uses JOINT_POSITIONS as base).

        Args:
            batch: Input batch.

        Returns:
            Current output state tensor (B, actual_action_dim).
        """
        # Map each output type to its corresponding input type
        output_to_input: dict[DataType, DataType] = {
            DataType.JOINT_TARGET_POSITIONS: DataType.JOINT_POSITIONS,
            DataType.JOINT_POSITIONS: DataType.JOINT_POSITIONS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS: (
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS
            ),
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS: (
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS
            ),
        }

        current_list = []
        for data_type in self.ordered_output_data_types:
            input_type = output_to_input[data_type]
            # The input may carry more joints than the output targets (e.g.
            # input JOINT_POSITIONS has 16 dims but output JOINT_TARGET_POSITIONS
            # only has 14). Use output_dims to determine how many dims the output
            # expects, then slice the input current state to match. Please make sure
            # that the input and output data indices are matching.
            start_idx, end_idx = self.output_dims[data_type]
            expected_dim = end_idx - start_idx
            if input_type == DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS:
                batched = cast(
                    list[BatchedParallelGripperOpenAmountData],
                    batch.inputs[input_type],
                )
                data = torch.cat([b.open_amount for b in batched], dim=-1)
            else:
                batched = cast(list[BatchedJointData], batch.inputs[input_type])
                data = torch.cat([b.value for b in batched], dim=-1)
            # Slice to match output dim (input may have more joints than output)
            current_list.append(data[:, -1, :expected_dim])  # (B, dim)

        return torch.cat(current_list, dim=-1)  # (B, actual_action_dim)

    def forward(
        self, batch: BatchedInferenceInputs
    ) -> dict[DataType, list[BatchedNCData]]:
        """Perform inference to predict action sequence.

        Runs the full GR00T N1.6 pipeline:
          1. Combine and normalize proprioceptive state
          2. Preprocess images and construct VLM inputs
          3. Run VLM backbone (once) to get visual-language context
          4. Run Euler denoising with the DiT
          5. Convert predicted deltas to absolute positions

        Args:
            batch: Input batch containing proprioceptive data, RGB_IMAGES,
                and optionally LANGUAGE observations.

        Returns:
            Dictionary mapping output data types to lists of batched predictions.
        """
        self._set_frozen_modules_eval()

        B = len(batch)

        # -- 1. Proprio state --
        proprio_padded = self._combine_proprio(batch)  # (B, max_state_dim)
        proprio_tokens = self.proprio_mlp(proprio_padded)  # (B, input_embedding_dim)

        # Current output state for delta -> absolute conversion
        if self.use_relative_deltas:
            q_current = self._build_current_output_state(
                batch
            )  # (B, actual_action_dim)

        # -- 2. VLM inputs --
        pixel_values, input_ids, attn_mask = self._prepare_vlm_inputs(batch)

        # -- 3. VLM forward (run once, reuse for all denoising steps) --
        vl_hidden, vl_attn_mask, image_mask = self.vlm_backbone(
            pixel_values, input_ids, attn_mask
        )
        # Cast to connector dtype in case backbone runs in bf16
        connector_dtype = self.mlp_connector.layer_norm.weight.dtype
        vl_context = self.mlp_connector(vl_hidden.to(connector_dtype))

        # -- 4. Denoising loop --
        action_mask_expanded = self._action_mask(self.device)

        x_t = torch.randn(
            B, self.output_prediction_horizon, self.max_action_dim, device=self.device
        )
        x_t = x_t * action_mask_expanded

        dt = 1.0 / self.num_denoising_steps

        for i in range(self.num_denoising_steps):
            t_cont = i / self.num_denoising_steps
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full(
                (B,), t_discretized, dtype=torch.long, device=self.device
            )

            action_tokens = self.action_encoder_mlp(x_t)
            action_tokens = add_action_step_positional_encoding(
                action_tokens=action_tokens,
                action_position_embedding=self.action_position_embedding,
                max_action_pos_embeddings=self.max_action_pos_embeddings,
            )
            sa_tokens = torch.cat([proprio_tokens.unsqueeze(1), action_tokens], dim=1)

            dit_output = self.dit(
                hidden_states=sa_tokens,
                timestep=timesteps,
                encoder_hidden_states=vl_context,
                encoder_attention_mask=vl_attn_mask,
                image_mask=image_mask,
            )

            v_pred = self.action_decoder_mlp(dit_output[:, 1:, :])
            x_t = x_t + dt * v_pred
            x_t = x_t * action_mask_expanded

        # -- 5. Decode: slice actual dims -> unnormalize -> convert to absolute --
        decoded_actual = self._slice_action(x_t)  # (B, H, actual_action_dim)
        if self.use_relative_deltas:
            # Model predicted deltas in normalized space. Recover absolute by
            # adding the normalized current state, then unnormalize.
            # delta_norm = norm(target) - norm(current)
            # => norm(target) = delta_norm + norm(current)
            normalized_current = self.action_normalizer.normalize(
                q_current.unsqueeze(1)
            )  # (B, 1, actual_action_dim)
            absolute_actions = self.action_normalizer.unnormalize(
                decoded_actual + normalized_current
            )
        else:
            absolute_actions = self.action_normalizer.unnormalize(decoded_actual)

        # -- 6. Pack into Neuracore output format --
        output: dict[DataType, list[BatchedNCData]] = {}
        for data_type in self.ordered_output_data_types:
            start_idx, end_idx = self.output_dims[data_type]
            dt_preds = absolute_actions[:, :, start_idx:end_idx]

            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched_outputs: list[BatchedNCData] = []
                for i, _ in enumerate(self.output_dataset_statistics[data_type]):
                    batched_outputs.append(
                        BatchedJointData(value=dt_preds[:, :, i : i + 1])
                    )
                output[data_type] = batched_outputs
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                batched_outputs = []
                for i, _ in enumerate(self.output_dataset_statistics[data_type]):
                    batched_outputs.append(
                        BatchedParallelGripperOpenAmountData(
                            open_amount=dt_preds[:, :, i : i + 1]
                        )
                    )
                output[data_type] = batched_outputs
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        return output

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step using flow matching.

        Flow matching objective:
            t ~ Beta(alpha, beta)
            epsilon ~ N(0, I)
            x_t = (1 - t) * epsilon + t * target_gt
            v_target = target_gt - epsilon
            v_pred = DiT(x_t, t, proprio, vl_context)
            loss = MSE(v_pred, v_target)  (masked to real dims only)

        When use_relative_deltas=True, target_gt is the state-relative delta
        (absolute_target - current_state). When False, target_gt is the
        absolute target position directly.

        Args:
            batch: Training batch with inputs and target outputs.

        Returns:
            BatchedTrainingOutputs containing flow_matching_loss.
        """
        self._set_frozen_modules_eval()

        B = batch.batch_size
        inference_sample = BatchedInferenceInputs(
            inputs=batch.inputs,
            inputs_mask=batch.inputs_mask,
            batch_size=B,
        )

        # -- 1. Proprio state --
        proprio_padded = self._combine_proprio(inference_sample)
        proprio_tokens = self.proprio_mlp(proprio_padded)
        proprio_tokens = self._augment_proprio_tokens(proprio_tokens)

        # Current output state for delta conversion (only needed for relative deltas)
        if self.use_relative_deltas:
            q_current = self._build_current_output_state(
                inference_sample
            )  # (B, actual_action_dim)

        # -- 2. VLM forward --
        pixel_values, input_ids, attn_mask = self._prepare_vlm_inputs(inference_sample)
        vl_hidden, vl_attn_mask, image_mask = self.vlm_backbone(
            pixel_values, input_ids, attn_mask
        )
        # Cast to connector dtype in case backbone runs in bf16
        connector_dtype = self.mlp_connector.layer_norm.weight.dtype
        vl_context = self.mlp_connector(vl_hidden.to(connector_dtype))

        # -- 3. Build targets (deltas or absolute) -> normalize -> pad --
        action_targets = []
        for data_type in self.ordered_output_data_types:
            if data_type in [DataType.JOINT_TARGET_POSITIONS, DataType.JOINT_POSITIONS]:
                batched = cast(list[BatchedJointData], batch.outputs[data_type])
                action_targets.extend([b.value for b in batched])
            elif data_type in [
                DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
                DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            ]:
                batched = cast(
                    list[BatchedParallelGripperOpenAmountData],
                    batch.outputs[data_type],
                )
                action_targets.extend([b.open_amount for b in batched])
            else:
                raise ValueError(f"Unsupported output data type: {data_type}")

        absolute_targets = torch.cat(
            action_targets, dim=-1
        )  # (B, H, actual_action_dim)
        if self.use_relative_deltas:
            # Compute deltas in normalized space: norm(target) - norm(current).
            # This correctly scales deltas by std (mean cancels out), so the model
            # trains on zero-centered, unit-scaled targets without needing separate
            # delta statistics.
            normalized_targets = self.action_normalizer.normalize(
                absolute_targets
            ) - self.action_normalizer.normalize(q_current.unsqueeze(1))
        else:
            normalized_targets = self.action_normalizer.normalize(absolute_targets)
        padded_targets = self._pad_action(normalized_targets)  # (B, H, max_action_dim)

        # -- 4. Flow matching: sample noise and interpolate --
        action_mask_expanded = self._action_mask(padded_targets.device)

        t_beta = self._beta_dist.sample([B]).to(
            device=self.device, dtype=padded_targets.dtype
        )
        t = (1 - t_beta) * self.noise_s
        t_expanded = t[:, None, None]

        noise = torch.randn_like(padded_targets) * action_mask_expanded
        x_t = (1 - t_expanded) * noise + t_expanded * padded_targets

        # -- 5. DiT predicts velocity --
        t_discretized = (t * self.num_timestep_buckets).long()
        action_tokens = self.action_encoder_mlp(x_t)
        action_tokens = add_action_step_positional_encoding(
            action_tokens=action_tokens,
            action_position_embedding=self.action_position_embedding,
            max_action_pos_embeddings=self.max_action_pos_embeddings,
        )
        sa_tokens = torch.cat([proprio_tokens.unsqueeze(1), action_tokens], dim=1)

        dit_output = self.dit(
            hidden_states=sa_tokens,
            timestep=t_discretized,
            encoder_hidden_states=vl_context,
            encoder_attention_mask=vl_attn_mask,
            image_mask=image_mask,
        )
        v_pred = self.action_decoder_mlp(dit_output[:, 1:, :])

        # -- 6. Flow matching loss (only on real dims) --
        v_target = padded_targets - noise
        loss = F.mse_loss(
            v_pred * action_mask_expanded,
            v_target * action_mask_expanded,
        )

        losses: dict[str, Any] = {"flow_matching_loss": loss}
        metrics: dict[str, Any] = {"flow_matching_loss": loss}
        return BatchedTrainingOutputs(losses=losses, metrics=metrics)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer with per-component learning rates.

        Learning rate strategy:
          - Projector MLPs (proprio, action enc/dec, connector): full lr
            when finetune_state_action_projector=True (default).
          - DiT action head: lr * dit_lr_scale when finetune_action_expert=True
            (default).
          - Visual encoder: lr * vlm_lr_scale when finetune_vision_encoder=True.
          - Top LLM layers: lr * vlm_lr_scale for top finetune_top_llm_layer
            layers (0 = fully frozen).

        All other parameters are frozen.

        Returns:
            List containing a single AdamW optimizer with parameter groups.
        """
        for param in self.parameters():
            param.requires_grad = False

        trainable_params: list[dict] = []

        if self.finetune_state_action_projector:
            trainable_params += [
                {"params": list(self.proprio_mlp.parameters()), "lr": self.lr},
                {
                    "params": list(self.action_encoder_mlp.parameters()),
                    "lr": self.lr,
                },
                {
                    "params": list(self.action_decoder_mlp.parameters()),
                    "lr": self.lr,
                },
                {
                    "params": list(self.mlp_connector.parameters()),
                    "lr": self.lr * self.dit_lr_scale,
                },
            ]
            if self.action_position_embedding is not None:
                trainable_params.append({
                    "params": list(self.action_position_embedding.parameters()),
                    "lr": self.lr,
                })

        if self.finetune_action_expert:
            trainable_params.append({
                "params": list(self.dit.parameters()),
                "lr": self.lr * self.dit_lr_scale,
            })

        if self.finetune_vision_encoder and not self.use_tiny_vlm:
            if (
                hasattr(self.vlm_backbone, "model")
                and self.vlm_backbone.model is not None
            ):
                visual_params = list(self.vlm_backbone.model.vision_model.parameters())
                if hasattr(self.vlm_backbone.model, "mlp1"):
                    visual_params += list(self.vlm_backbone.model.mlp1.parameters())
                trainable_params.append({
                    "params": visual_params,
                    "lr": self.lr * self.vlm_lr_scale,
                })

        if self.finetune_top_llm_layer > 0:
            for layer in self.vlm_backbone.llm_layers[-self.finetune_top_llm_layer :]:
                trainable_params.append({
                    "params": list(layer.parameters()),
                    "lr": self.lr * self.vlm_lr_scale,
                })

        for group in trainable_params:
            for param in cast(list[nn.Parameter], group["params"]):
                param.requires_grad = True

        logger.info(
            "Trainable components — projector: %s, DiT: %s, visual: %s, "
            "top_llm_layers: %d",
            self.finetune_state_action_projector,
            self.finetune_action_expert,
            self.finetune_vision_encoder,
            self.finetune_top_llm_layer,
        )

        # Use fused AdamW when CUDA is available (PyTorch 2.0+) for ~10% faster
        # optimizer step by fusing the update kernels.
        use_fused = (
            torch.cuda.is_available()
            and "fused" in inspect.signature(torch.optim.AdamW).parameters
        )
        return [
            torch.optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                fused=use_fused,
            )
        ]

    def configure_schedulers(
        self,
        optimizers: list[torch.optim.Optimizer],
        num_training_steps: int,
    ) -> list[LambdaLR]:
        """Configure cosine annealing LR scheduler with linear warmup.

        Matches the GR00T N1.6 training recipe: 5% linear warmup followed
        by cosine decay to zero.

        Args:
            optimizers: List of optimizers from configure_optimizers.
            num_training_steps: Total number of training steps.

        Returns:
            List of LambdaLR schedulers, one per optimizer.
        """
        warmup_steps = int(num_training_steps * self.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(
                1, num_training_steps - warmup_steps
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load checkpoints while staying compatible with older GR00T ports."""
        patched_state_dict = dict(state_dict)
        key = "action_position_embedding.weight"

        if self.action_position_embedding is not None and key not in patched_state_dict:
            logger.warning(
                "Checkpoint is missing %s; keeping the current initialized "
                "action-step positional embedding.",
                key,
            )
            patched_state_dict[key] = (
                self.action_position_embedding.weight.detach().clone()
            )
        elif self.action_position_embedding is None and key in patched_state_dict:
            logger.warning(
                "Dropping %s from checkpoint because action-step positional "
                "encoding is disabled in this model instance.",
                key,
            )
            patched_state_dict.pop(key)

        return super().load_state_dict(patched_state_dict, strict=strict)

    def _action_mask(self, device: torch.device) -> torch.Tensor:
        """Build action mask on the given device, shaped (1, 1, max_action_dim)."""
        mask = torch.zeros(self.max_action_dim, device=device)
        mask[: self.actual_action_dim] = 1.0
        return mask.unsqueeze(0).unsqueeze(0)

    def _pad_state(self, x: torch.Tensor) -> torch.Tensor:
        """Pad state vector to max_state_dim with zeros."""
        return F.pad(x, (0, self.max_state_dim - self.actual_state_dim), value=0.0)

    def _pad_action(self, x: torch.Tensor) -> torch.Tensor:
        """Pad action tensor to max_action_dim with zeros."""
        return F.pad(x, (0, self.max_action_dim - self.actual_action_dim), value=0.0)

    def _slice_action(self, x: torch.Tensor) -> torch.Tensor:
        """Slice padded action tensor to actual action dimensions."""
        return x[..., : self.actual_action_dim]

    @staticmethod
    def get_supported_input_data_types() -> set[DataType]:
        """Get the input data types supported by GR00T N1.6."""
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
        """Get the output data types supported by GR00T N1.6."""
        return {
            DataType.JOINT_POSITIONS,
            DataType.JOINT_TARGET_POSITIONS,
            DataType.PARALLEL_GRIPPER_OPEN_AMOUNTS,
            DataType.PARALLEL_GRIPPER_TARGET_OPEN_AMOUNTS,
        }
