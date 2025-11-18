"""π0: A Vision-Language-Action Flow Model for General Robot Control.

This module implements the π0 (Pi0) model from the Physical Intelligence
paper. π0 is a vision-language-action model that has a VLM from the pretrained
PaliGemma model and a flow matching action expert. The model uses a mixture
of experts (MoE) to process the input and predict the action sequence.

Reference: Black, Kevin, et al. "π0: A Vision-Language-Action Flow Model
for General Robot Control." arXiv preprint https://arxiv.org/abs/2410.24164.
"""

import logging
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from neuracore_types import DataType, ModelInitDescription, ModelPrediction
from transformers import AutoProcessor, AutoTokenizer, PaliGemmaForConditionalGeneration

from neuracore.ml import (
    BatchedInferenceSamples,
    BatchedTrainingOutputs,
    BatchedTrainingSamples,
    NeuracoreModel,
)

from .modules import ActionEncoder, GemmaMoE, MoeExpertConfig, SinusoidalPosEmb

logging.getLogger("transformers").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Global tokenizer for static method
_tokenizer = None
LANGUAGE_MODEL_NAME = "google/paligemma-3b-pt-224"

VLM_BACKBONE = "google/paligemma-3b-pt-224"
VLM_EXPERT_WIDTH = 2048  # Width of the VLM expert, matches PaliGemma's hidden size


class Pi0(NeuracoreModel):
    """Implementation of Pi0 model from the Physical Intelligence paper.

    Currently only supports finetuning the action expert. The model combines
    vision-language understanding with action prediction through a mixture of
    experts architecture.
    """

    def __init__(
        self,
        model_init_description: ModelInitDescription,
        vlm_expert_intermediate_size: int = 16384,
        vlm_expert_num_heads: int = 8,
        vlm_expert_num_kv_heads: int = 1,
        vlm_expert_head_dim: int = 256,
        vlm_max_text_tokens: int = 128,
        action_expert_width: int = 1024,
        action_expert_intermediate_size: int = 4096,
        action_expert_num_heads: int = 8,
        action_expert_num_kv_heads: int = 1,
        action_expert_head_dim: int = 256,
        moe_depth: int = 18,
        num_inference_steps: int = 10,
        flow_sig_min: float = 0.001,
        flow_alpha: float = 1.5,
        flow_beta: float = 1.0,
        lr: float = 5e-5,
        weight_decay: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the Pi0 model.

        Args:
            model_init_description: Model initialization configuration.
            vlm_expert_intermediate_size: Intermediate size of the VLM expert.
            vlm_expert_num_heads: Number of attention heads in the VLM expert.
            vlm_expert_num_kv_heads: Number of key-value heads in the VLM expert.
            vlm_expert_head_dim: Dimension of each attention head in the VLM expert.
            vlm_max_text_tokens: Maximum number of text tokens for the VLM.
            action_expert_width: Width of the action expert.
            action_expert_intermediate_size: Intermediate size of the action expert.
            action_expert_num_heads: Number of attention heads in the action expert.
            action_expert_num_kv_heads: Number of key-value heads in the action expert.
            action_expert_head_dim: Dimension of each attention head in action expert.
            moe_depth: Depth of the mixture of experts.
            num_inference_steps: Number of inference steps.
            flow_sig_min: Minimum value for the flow sigma.
            flow_alpha: Alpha parameter for the flow beta distribution.
            flow_beta: Beta parameter for the flow beta distribution.
            lr: Learning rate for the model.
            weight_decay: Weight decay for the model.
            dtype: Data type for model parameters and computations.
        """
        super().__init__(model_init_description)

        if not os.environ.get("HF_TOKEN"):
            raise ValueError(
                "Hugging Face token not found. "
                "Please set the HF_TOKEN environment variable."
            )

        self.action_dim = self.dataset_description.joint_target_positions.max_len
        self.action_horizon = self.output_prediction_horizon
        self.vlm_max_text_tokens = vlm_max_text_tokens
        num_rgbs = model_init_description.dataset_description.rgb_images.max_len
        self.vlm_max_tokens = num_rgbs * 256 + self.vlm_max_text_tokens
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)
        self.lr = lr
        self.weight_decay = weight_decay
        proprio_dim = (
            self.dataset_description.joint_positions.max_len
            + self.dataset_description.joint_velocities.max_len
            + self.dataset_description.joint_torques.max_len
        )
        self.dtype = dtype

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
        expert_configs = {
            "vlm": MoeExpertConfig(
                hidden_size=VLM_EXPERT_WIDTH,
                intermediate_size=vlm_expert_intermediate_size,
                head_dim=vlm_expert_head_dim,
                num_attention_heads=vlm_expert_num_heads,
                num_key_value_heads=vlm_expert_num_kv_heads,
            ),
            "action": MoeExpertConfig(
                hidden_size=action_expert_width,
                intermediate_size=action_expert_intermediate_size,
                head_dim=action_expert_head_dim,
                num_attention_heads=action_expert_num_heads,
                num_key_value_heads=action_expert_num_kv_heads,
            ),
        }
        self.moe = GemmaMoE(moe_depth, expert_configs)
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
        else:
            logger.warning("Using custom VLM weights, not pretrained PaliGemma")

        # disable grads for VLM part of MoE if using pretrained
        if self.using_pretrained_paligemma:
            for param in self.moe.get_parameters("vlm"):
                param.requires_grad = False

        # Delete the language model to save memory (keep only embeddings)
        # Note: We delete model.language_model (the actual module), not
        # language_model (the property)
        del self.vlm.model.language_model

        # Resize the images to 224x224
        self.image_normalizer = torch.nn.Sequential(
            T.Resize((224, 224)),
        )

        # Normalization statistics
        self._setup_normalization_stats()

    def _setup_normalization_stats(self) -> None:
        """Setup normalization statistics for different data types."""
        # Joint state normalization
        state_means = []
        state_stds = []

        if DataType.JOINT_POSITIONS in self.model_init_description.input_data_types:
            state_means.extend(self.dataset_description.joint_positions.mean)
            state_stds.extend(self.dataset_description.joint_positions.std)
        if DataType.JOINT_VELOCITIES in self.model_init_description.input_data_types:
            state_means.extend(self.dataset_description.joint_velocities.mean)
            state_stds.extend(self.dataset_description.joint_velocities.std)
        if DataType.JOINT_TORQUES in self.model_init_description.input_data_types:
            state_means.extend(self.dataset_description.joint_torques.mean)
            state_stds.extend(self.dataset_description.joint_torques.std)

        if state_means:
            # Register as buffers so they move with the model
            self.register_buffer(
                "joint_state_mean", self._to_torch_float_tensor(state_means)
            )
            self.register_buffer(
                "joint_state_std", self._to_torch_float_tensor(state_stds)
            )
        else:
            self.joint_state_mean = None
            self.joint_state_std = None
        self.register_buffer(
            "joint_target_mean",
            self._to_torch_float_tensor(
                self.dataset_description.joint_target_positions.mean
            ),
        )
        self.register_buffer(
            "joint_target_std",
            self._to_torch_float_tensor(
                self.dataset_description.joint_target_positions.std
            ),
        )

    def _to_torch_float_tensor(self, data: list[float]) -> torch.Tensor:
        """Convert list of floats to torch tensor on the correct device."""
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def _preprocess_joint_state(
        self,
        joint_state: torch.Tensor,
        joint_state_mean: torch.Tensor,
        joint_state_std: torch.Tensor,
    ) -> torch.Tensor:
        """Preprocess the states."""
        return (joint_state - joint_state_mean) / joint_state_std

    def _unnormalize_actions(self, predicted_actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize the actions."""
        return (predicted_actions * self.joint_target_std) + self.joint_target_mean

    def _combine_normalized_joint_states(
        self, batch: BatchedInferenceSamples
    ) -> torch.Tensor:
        """Combine joint states."""
        state_inputs = []
        if batch.joint_positions:
            state_inputs.append(batch.joint_positions.data * batch.joint_positions.mask)
        if batch.joint_velocities:
            state_inputs.append(
                batch.joint_velocities.data * batch.joint_velocities.mask
            )
        if batch.joint_torques:
            state_inputs.append(batch.joint_torques.data * batch.joint_torques.mask)

        if state_inputs:
            joint_states = torch.cat(state_inputs, dim=-1)
            joint_states = self._preprocess_joint_state(
                joint_states, self.joint_state_mean, self.joint_state_std
            )
            return joint_states
        else:
            # Return zero tensor if no joint states available
            raise ValueError("No joint states available")

    def _prepare_rgb_images(self, batch: BatchedInferenceSamples) -> torch.Tensor:
        """Prepare the RGB images and masks.

        First resize to 224x224 and then normalize values to [-1,1]. And transform
        the image dimension to (num_cams, B, C, H, W).

        Args:
            batch: Batch of inference samples.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: List of images and masks.
        """
        if batch.rgb_images is None:
            raise ValueError("RGB images are required but not provided")
        images = []
        image_masks = []
        for cam_id in range(self.dataset_description.rgb_images.max_len):
            image = self.image_normalizer(batch.rgb_images.data[:, cam_id])
            # Normalize from range [0,1] to [-1,1] as expected by siglip
            image = image * 2.0 - 1.0
            images.append(image)
            image_masks.append(batch.rgb_images.mask[:, cam_id])
        return images, image_masks

    def _process_language_tokens(
        self,
        batch: BatchedInferenceSamples,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the language tokens.

        Args:
            batch_size: Size of the batch.
            language_tokens: Language tokens tensor.
            language_mask: Language mask tensor.

        Returns:
            torch.Tensor: Language tokens tensor.
            torch.Tensor: Language mask tensor.
        """
        batch_size = len(batch)
        if batch.language_tokens is None:
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
            language_tokens = batch.language_tokens.data
            language_mask = batch.language_tokens.mask
        return language_tokens, language_mask

    def _load_pretrained_vlm_weights(self) -> None:
        """Load pretrained PaliGemma weights into the VLM expert of the MoE."""
        logger.info("Loading pretrained PaliGemma weights into VLM expert...")
        vlm_state_dict = self.vlm.model.language_model.state_dict()
        moe_state_dict = self.moe.state_dict()
        new_state_dict = {}
        for moe_key, moe_param in moe_state_dict.items():
            # Check if this is a VLM expert parameter
            if "experts.vlm" in moe_key:
                # Convert MoE key format to PaliGemma key format
                vlm_key = moe_key.replace("experts.vlm.", "")

                # If this key exists in the VLM state dict, copy it
                if vlm_key not in vlm_state_dict:
                    raise ValueError(f"VLM key not found: {vlm_key}")
                new_state_dict[moe_key] = vlm_state_dict[vlm_key]
            else:
                # Keep non-VLM parameters as is
                new_state_dict[moe_key] = moe_param

        # Load the combined state dict
        missing_keys, unexpected_keys = self.moe.load_state_dict(
            new_state_dict, strict=True
        )

        # Log any mismatches for debugging
        if missing_keys:
            raise ValueError(f"Missing keys when loading VLM weights: {missing_keys}")
        if unexpected_keys:
            raise ValueError(
                f"Unexpected keys when loading VLM weights: {unexpected_keys}"
            )

        logger.info("Successfully loaded pretrained PaliGemma weights into VLM expert.")

    def _create_expert_attention_masks(
        self, batch_size: int, pad_masks: torch.Tensor = None
    ) -> dict[str, torch.Tensor]:
        """Create attention masks for the experts.

        Args:
            batch_size: Size of the batch.
            pad_masks: Padding masks for the merged text and images tensor.

        Returns:
            dict[str, torch.Tensor]: Attention masks for the experts.
        """
        # generate 2d padding mask from 1d padding mask
        # pad_masks has shape [batch_size, seq_len]
        # Create attention mask: [batch_size, seq_len, seq_len]
        vlm_mask = pad_masks.unsqueeze(1) * pad_masks.unsqueeze(2)
        # Convert to attention mask format (0 for attended positions, -inf for masked)
        vlm_mask = torch.where(vlm_mask == 1, 0.0, torch.finfo(self.dtype).min).to(
            self.dtype
        )
        state_len = 1
        action_len = self.action_horizon

        stat_act_len = state_len + action_len  # proprio + actions
        state_action_mask = torch.zeros(
            (stat_act_len, stat_act_len), device=self.device, dtype=self.dtype
        )

        # Proprio can only attend to itself
        state_action_mask[0, 0] = 1

        # Each action can attend to proprio and previous actions
        for i in range(1, stat_act_len):  # i starts at 1 (first action)
            # Can attend to proprio
            state_action_mask[i, 0] = 1
            # Can attend to self and previous actions
            state_action_mask[i, 1 : i + 1] = 1

        # Convert to attention mask format (0 for attended positions, -inf for masked)
        state_action_mask = torch.where(
            state_action_mask == 1, 0.0, torch.finfo(self.dtype).min
        ).to(self.dtype)

        # Add head dimension: [batch_size, 1, seq_len, seq_len]
        vlm_mask = vlm_mask.unsqueeze(1)
        state_action_mask = (
            state_action_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1, -1)
        )

        return {"vlm": vlm_mask, "action": state_action_mask}

    def _create_pi0_mix_attention_mask(
        self, batch_size: int, vlm_seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Create the mixed attention mask for the Pi0 model.

        Args:
            batch_size: Size of the batch.
            vlm_seq_len: Actual VLM sequence length.

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
        self, batch_size: int, vlm_seq_len: Optional[int] = None
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
        images: torch.Tensor,
        image_masks: torch.Tensor,
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
        vlm_seq_len: Optional[int] = None,
        pad_masks: Optional[torch.Tensor] = None,
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

    def forward(self, batch: BatchedInferenceSamples) -> ModelPrediction:
        """Forward pass for generating actions.

        Args:
            batch: Batch of inference samples.

        Returns:
            torch.Tensor: Generated actions tensor.
        """
        t_start = time.time()
        batch_size = len(batch)
        if self.dataset_description.rgb_images.max_len > 0:
            # (B, Predict_Horizon, VLM_EMBED_DIM)
            if batch.rgb_images is None:
                raise ValueError("RGB images are required")
            images, image_masks = self._prepare_rgb_images(batch)

        else:
            raise ValueError("No RGB images available")
        language_tokens, language_masks = self._process_language_tokens(batch)
        merged_text_images, pad_masks = self._forward_vlm_merged_text_images(
            images, image_masks, language_tokens, language_masks
        )
        proprio_states = self._combine_normalized_joint_states(batch)
        proprio_embeds = self.proprio_encoder(proprio_states)  # (B, 1, E)

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
        prediction_time = time.time() - t_start
        predictions = self._unnormalize_actions(action)
        predictions = predictions.detach().cpu().float().numpy()
        return ModelPrediction(
            outputs={DataType.JOINT_TARGET_POSITIONS: predictions},
            prediction_time=prediction_time,
        )

    def training_step(self, batch: BatchedTrainingSamples) -> BatchedTrainingOutputs:
        """Perform a single training step.

        Args:
            batch: Training batch with inputs and targets

        Returns:
            BatchedTrainingOutputs: Training outputs with losses and metrics
        """
        inference_sample = BatchedInferenceSamples(
            joint_positions=batch.inputs.joint_positions,
            joint_velocities=batch.inputs.joint_velocities,
            joint_torques=batch.inputs.joint_torques,
            rgb_images=batch.inputs.rgb_images,
            language_tokens=batch.inputs.language_tokens,
            joint_target_positions=batch.outputs.joint_target_positions,
        )
        proprios = self._combine_normalized_joint_states(inference_sample)
        if batch.outputs.joint_target_positions is None:
            raise ValueError("Joint target positions are required")
        target_actions = self._preprocess_joint_state(
            batch.outputs.joint_target_positions.data,
            self.joint_target_mean,
            self.joint_target_std,
        )
        target_actions = target_actions * batch.outputs.joint_target_positions.mask
        t = self._sample_fm_time(len(batch))
        x0 = torch.randn_like(target_actions)
        x1 = target_actions
        # Calculate conditional flow
        _t = t.view(-1, 1, 1)
        psi_t = (1 - (1 - self.flow_sig_min) * _t) * x0 + _t * x1
        if self.dataset_description.rgb_images.max_len > 0:
            # (B, Predict_Horizon, VLM_EMBED_DIM)
            if (
                inference_sample.rgb_images is None
                or inference_sample.language_tokens is None
            ):
                raise ValueError(
                    "RGB images and language tokens are required for training"
                )
            images, image_masks = self._prepare_rgb_images(inference_sample)
        else:
            raise ValueError("No RGB images available")
        lang_tokens, lang_masks = self._process_language_tokens(inference_sample)
        merged_text_images, pad_masks = self._forward_vlm_merged_text_images(
            images, image_masks, lang_tokens, lang_masks
        )
        proprio_embeds = self.proprio_encoder(proprios)  # (B, 1, E)
        # Get the actual sequence length from the merged embeddings
        actual_seq_len = merged_text_images.shape[1]
        v_psi = self._predict_action(
            merged_text_images, proprio_embeds, psi_t, t, actual_seq_len, pad_masks
        )
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        loss = F.mse_loss(v_psi, d_psi, reduction="none")
        mask = batch.outputs.joint_target_positions.mask
        loss = (loss * mask).mean()
        losses = {
            "mse_loss": loss,
        }
        metrics = {
            "mse_loss": loss,
        }
        return BatchedTrainingOutputs(
            output_predictions=v_psi,
            losses=losses,
            metrics=metrics,
        )

    def _get_action_expert_parameters(self) -> list[torch.nn.Parameter]:
        """Get parameters of the action expert.

        Returns:
            list: List of action expert parameters.
        """
        return (
            list(self.action_encoder.parameters())
            + list(self.action_decoder.parameters())
            + list(self.proprio_encoder.parameters())
            + list(self.moe.get_parameters("action"))
        )

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        """Configure optimizer with different learning rates for different components.

        Uses separate learning rates for image encoder backbone (typically lower)
        and other model parameters to account for pre-trained vision components.

        Returns:
            list[torch.optim.Optimizer]: List containing the configured optimizer
        """
        if self.using_pretrained_paligemma:
            # Only train action expert parameters when using pretrained VLM
            trainable_params = self._get_action_expert_parameters()
        else:
            # Train all parameters when not using pretrained weights
            trainable_params = [p for p in self.parameters() if p.requires_grad]
        param_groups = [
            {"params": trainable_params, "lr": self.lr},
        ]

        return [torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)]

    @staticmethod
    def get_supported_input_data_types() -> list[DataType]:
        """Get the input data types supported by this model.

        Returns:
            list[DataType]: List of supported input data types
        """
        return [
            DataType.JOINT_POSITIONS,
            DataType.JOINT_VELOCITIES,
            DataType.JOINT_TORQUES,
            DataType.RGB_IMAGE,
            DataType.LANGUAGE,
        ]

    @staticmethod
    def get_supported_output_data_types() -> list[DataType]:
        """Get the output data types supported by this model.

        Returns:
            list[DataType]: List of supported output data types
        """
        return [DataType.JOINT_TARGET_POSITIONS]

    @staticmethod
    def tokenize_text(text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text using the pretrained tokenizer.

        Args:
            text: List of text strings to tokenize

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Input IDs and attention masks
        """
        text = [text if text.endswith("\n") else f"{text}\n" for text in text]

        global _tokenizer
        if _tokenizer is None:
            # Only PaliGemma-3B supported for now
            _tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME)

        # Tokenize the text
        tokens = _tokenizer(
            text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        # Extract token ids and attention mask
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return input_ids, attention_mask
