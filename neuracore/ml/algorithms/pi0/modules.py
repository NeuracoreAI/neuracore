"""Core PyTorch modules for the PI0 algorithm.

This module implements the PI0Policy model that combines a PaliGemma
vision-language model with a Gemma action expert for robot manipulation.
The model uses flow matching to denoise action sequences conditioned on
visual observations and proprioceptive state.
"""

# cspell:ignore OPENPI adarms layernorm silu huggingface openpi denoised

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)
from transformers.utils import cached_file

from neuracore.ml.algorithms.pi0.gemma_pytorch import (
    PaliGemmaWithExpertModel,
    get_gemma_config,
)
from neuracore.ml.algorithms.pi0.utils import (
    OPENPI_ATTENTION_MASK_VALUE,
    PI0Config,
    _align_mask_length,
    _create_sinusoidal_pos_embedding,
    _make_att_2d_masks,
    _sample_beta,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class PI0Policy(nn.Module):
    """Core PI0 model combining PaliGemma VLM with Gemma action expert.

    This model processes visual observations and language through PaliGemma,
    then uses a separate Gemma model as the action expert to predict
    denoised action sequences via flow matching.

    The architecture supports gradient checkpointing and torch.compile
    optimization for efficient training and inference.
    """

    def __init__(self, config: PI0Config):
        """Initialize the PI0 model.

        Args:
            config: Model configuration specifying architecture and hyperparameters
        """
        super().__init__()
        self.config = config

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=config.use_adarms,
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(
            config.max_action_dim, action_expert_config.width
        )
        self.action_out_proj = nn.Linear(
            action_expert_config.width, config.max_action_dim
        )

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(
            2 * action_expert_config.width, action_expert_config.width
        )
        self.action_time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        self.gradient_checkpointing_enabled = False

        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
        if config.device is not None:
            self.to(config.device)

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on all submodules."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            True
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on all submodules."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def compile_model_enable(self) -> None:
        """Enable model compilation."""
        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(  # type: ignore[method-assign]
            self.sample_actions, mode=self.config.compile_mode
        )
        self.forward = torch.compile(  # type: ignore[method-assign]
            self.forward, mode=self.config.compile_mode
        )
        logging.info("Enabled model compilation for PI0Pytorch model")

    def _apply_checkpoint(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Apply gradient checkpointing to a function if enabled.

        Args:
            func: Function to potentially checkpoint
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Function output, computed with or without checkpointing.
        """
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        """Expand 2D attention masks to 4D format for transformer layers.

        Args:
            att_2d_masks: 2D attention mask [B, seq_len, seq_len]

        Returns:
            4D attention mask [B, 1, seq_len, seq_len] with fill values applied.
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def _sample_noise(
        self, shape: torch.Size | tuple[int, ...], device: torch.device
    ) -> Tensor:
        """Sample standard normal noise for flow matching.

        Args:
            shape: Shape of the noise tensor
            device: Target device

        Returns:
            Tensor of standard normal noise.
        """
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def _sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Sample diffusion time steps from beta distribution.

        Args:
            bsize: Batch size
            device: Target device

        Returns:
            Tensor of time values [bsize] in range [offset, offset + scale].
        """
        time_beta = _sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = (
            time_beta * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        return time.to(dtype=torch.float32, device=device)

    def _embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed image and language inputs for the prefix sequence.

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of image masks [B] per camera
            lang_tokens: Language token IDs [B, L]
            lang_masks: Language attention mask [B, L]

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks).
        """
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1).to(dtype=torch.bool)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks_t = _align_mask_length(att_masks_t, pad_masks.shape[1])
        bsize = pad_masks.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, att_masks_t.shape[0])
        return embs, pad_masks, att_masks_t

    def _embed_suffix(
        self, state: Tensor, noisy_actions: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        """Embed state, noisy actions, and timestep for the action expert.

        Args:
            state: Proprioceptive state [B, state_dim]
            noisy_actions: Noisy action sequence [B, chunk_size, action_dim]
            timestep: Diffusion timestep [B]

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks, adarms_cond).
        """
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state: Tensor) -> Tensor:
            return self.state_proj(state)

        state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        time_emb = _create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions: Tensor) -> Tensor:
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb: Tensor) -> Tensor:
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
        adarms_cond = None

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks_t = _align_mask_length(att_masks_t, pad_masks.shape[1])
        att_masks_t = att_masks_t[None, :].expand(bsize, att_masks_t.shape[0])

        return embs, pad_masks, att_masks_t, adarms_cond

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Compute flow matching loss for training.

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of image masks [B] per camera
            lang_tokens: Language token IDs [B, L]
            lang_masks: Language attention mask [B, L]
            state: Proprioceptive state [B, state_dim]
            actions: Target action sequence [B, chunk_size, action_dim]
            noise: Optional pre-sampled noise
            time: Optional pre-sampled diffusion time

        Returns:
            Per-element MSE loss [B, chunk_size, action_dim].
        """
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device)
        if time is None:
            time = self._sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self._embed_suffix(state, x_t, time)
        )

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs: Tensor,
            suffix_embs: Tensor,
            att_2d_masks_4d: Tensor,
            position_ids: Tensor,
            adarms_cond: Tensor | None,
        ) -> Tensor:
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out: Tensor) -> Tensor:
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """Sample action sequence via Euler integration.

        From pure noise to actions using the flow matching ODE.

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of image masks [B] per camera
            lang_tokens: Language token IDs [B, L]
            lang_masks: Language attention mask [B, L]
            state: Proprioceptive state [B, state_dim]
            noise: Optional initial noise
            num_steps: Number of Euler steps (default: config.num_inference_steps)

        Returns:
            Sampled action sequence [B, chunk_size, action_dim].
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self._sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        paligemma_lm_config = self.paligemma_with_expert.paligemma.language_model.config
        paligemma_lm_config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self._denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def _denoise_step(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        past_key_values: list[torch.FloatTensor] | None,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Compute velocity field for a single Euler denoising step.

        Args:
            state: Proprioceptive state [B, state_dim]
            prefix_pad_masks: Padding masks from prefix embedding
            past_key_values: Cached key-values from prefix forward pass
            x_t: Current noisy actions [B, chunk_size, action_dim]
            timestep: Current diffusion time [B]

        Returns:
            Predicted velocity [B, chunk_size, action_dim].
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self._embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = _make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        gemma_config = self.paligemma_with_expert.gemma_expert.model.config
        gemma_config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        assert suffix_out is not None
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path | None = None,
        *,
        config: PI0Config | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> PI0Policy:
        """Load a pretrained PI0 model from HuggingFace Hub or local path.

        Args:
            pretrained_name_or_path: HuggingFace repo id or local path
            config: Model configuration (default: PI0Config())
            strict: Whether to strictly enforce state dict loading
            **kwargs: Additional arguments (cache_dir, force_download, etc.)

        Returns:
            PI0Policy model with loaded weights.
        """
        if pretrained_name_or_path is None:
            pretrained_name_or_path = "lerobot/pi0_base"
            logging.warning(
                "No pretrained model path provided; using default pi0_base model"
            )
        if config is None:
            config = PI0Config()

        model = cls(config, **kwargs)

        if cached_file is None or load_file is None:
            logging.warning(
                "transformers/safetensors not available; loading weights skipped"
            )
            return model

        try:
            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                resume_download=kwargs.get("resume_download"),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token") or kwargs.get("use_auth_token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            original_state_dict = load_file(resolved_file)
            logging.info("Loaded state dict from %s", resolved_file)
        except Exception as exc:
            logging.warning(
                "Could not load state dict from %s: %s", pretrained_name_or_path, exc
            )
            return model

        fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict)

        missing_keys, unexpected_keys = model.load_state_dict(
            fixed_state_dict, strict=False
        )
        if missing_keys:
            logging.warning("Missing keys when loading state dict: %s", missing_keys)
        if unexpected_keys:
            logging.warning(
                "Unexpected keys when loading state dict: %s", unexpected_keys
            )

        tie_key = (
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        )
        if tie_key in missing_keys:
            paligemma = model.paligemma_with_expert.paligemma
            if model._tie_or_copy_language_embeddings(paligemma):
                logging.info("Tied language embeddings to lm_head weight")
                missing_keys = [key for key in missing_keys if key != tie_key]
        logging.warning(
            "Missing keys after tying language embeddings: %s", missing_keys
        )
        logging.info(
            "Successfully loaded pretrained PI0 weights from %s",
            pretrained_name_or_path,
        )
        return model

    def _tie_or_copy_language_embeddings(
        self, paligemma: PaliGemmaForConditionalGeneration
    ) -> bool:
        """Tie or copy language embeddings to lm_head weight.

        Args:
            paligemma: PaliGemma model instance

        Returns:
            True if embeddings were successfully tied, False otherwise.
        """
        language_model = getattr(
            getattr(paligemma, "model", None), "language_model", None
        )
        lm_head = getattr(paligemma, "lm_head", None)
        if language_model is None or lm_head is None:
            return False

        embed_tokens = getattr(language_model, "embed_tokens", None)
        lm_head_weight = getattr(lm_head, "weight", None)
        if embed_tokens is None or lm_head_weight is None:
            return False

        embed_weight = getattr(embed_tokens, "weight", None)
        if embed_weight is None or embed_weight.shape != lm_head_weight.shape:
            return False

        with torch.no_grad():
            embed_weight.copy_(lm_head_weight)

        if hasattr(paligemma, "tie_weights"):
            paligemma.tie_weights()

        tied_embed = getattr(language_model.embed_tokens, "weight", None)
        return (
            tied_embed is not None
            and tied_embed.data_ptr() == lm_head_weight.data_ptr()
        )

    def _fix_pytorch_state_dict_keys(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fix state dict keys to match current model architecture.

        Handles key remapping and filtering for compatibility with
        different checkpoint formats (e.g., OpenPI vs current).

        Args:
            state_dict: Original state dict from checkpoint

        Returns:
            Fixed state dict with compatible keys.
        """
        import re

        fixed_state_dict: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            new_key = key

            if re.match(
                (
                    r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
                    r"(input_layernorm|post_attention_layernorm)\.weight"
                ),
                key,
            ):
                expert_uses_adarms = getattr(
                    self.paligemma_with_expert.gemma_expert.config,
                    "use_adarms",
                    False,
                )
                if expert_uses_adarms:
                    logging.warning(
                        "Skipping layer norm key (adaRMS mismatch): %s", key
                    )
                    continue

            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key
            ):
                expert_uses_adarms = getattr(
                    self.paligemma_with_expert.gemma_expert.config,
                    "use_adarms",
                    False,
                )
                if expert_uses_adarms:
                    logging.warning("Skipping norm key (adaRMS mismatch): %s", key)
                    continue

            if key.startswith("time_mlp_in."):
                new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
            elif key.startswith("time_mlp_out."):
                new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

            if "patch_embedding" in key:
                logging.warning("Vision embedding key might need handling: %s", key)

            fixed_state_dict[new_key] = value

        return fixed_state_dict
