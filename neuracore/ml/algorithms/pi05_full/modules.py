"""Core PyTorch modules for the Pi05Full algorithm.

This module implements the PI05FullPolicy model that combines a PaliGemma
vision-language model with a Gemma action expert for robot manipulation.
The model uses flow matching to denoise action sequences conditioned on
visual observations.
"""

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

from .gemma_pytorch import PaliGemmaWithExpertModel, get_gemma_config
from .utils import (
    OPENPI_ATTENTION_MASK_VALUE,
    PI05FullConfig,
    _align_mask_length,
    _create_sinusoidal_pos_embedding,
    _make_att_2d_masks,
    _sample_beta,
)

T = TypeVar("T")

logger = logging.getLogger(__name__)


class PI05FullPolicy(nn.Module):
    """Core Pi05 model combining PaliGemma VLM with Gemma action expert.

    This model processes visual observations and language through PaliGemma and
    uses a separate Gemma model as the action expert to predict denoised action
    sequences via flow matching.

    The architecture supports gradient checkpointing and torch.compile
    optimization for efficient training and inference.
    """

    def __init__(self, config: PI05FullConfig):
        """Initialize the Pi05 model.

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
            use_adarms=(False, True),
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(
            config.max_action_dim, action_expert_config.width
        )
        self.action_out_proj = nn.Linear(
            action_expert_config.width, config.max_action_dim
        )

        self.time_mlp_in = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )
        self.time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        self.gradient_checkpointing_enabled = False
        self.compile_enabled = False

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
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on all submodules."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = (
            False
        )
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def compile_model_enable(self) -> None:
        """Enable model compilation."""
        if self.compile_enabled:
            return
        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(  # type: ignore[method-assign]
            self.sample_actions, mode=self.config.compile_mode
        )
        self.compile_enabled = True
        logging.info("Enabled model compilation for PI05Pytorch model")

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
        subtask_tokens: Tensor | None = None,
        subtask_masks: Tensor | None = None,
        fast_tokens: Tensor | None = None,
        fast_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, slice]]:
        """Embed image, language, subtask, and FAST tokens into one prefix sequence.

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of image masks [B] per camera
            lang_tokens: Language token IDs [B, L]
            lang_masks: Language attention mask [B, L]
            subtask_tokens: Optional subtask token IDs [B, L_sub]
            subtask_masks: Optional subtask attention mask [B, L_sub]
            fast_tokens: Optional FAST action token IDs [B, L_fast]
            fast_masks: Optional FAST attention mask [B, L_fast]

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks, segments)
            where segments maps each segment name to the slice it occupies in
            the prefix sequence.
        """
        embs: list[Tensor] = []
        pad_masks: list[Tensor] = []
        att_masks: list[int] = []
        segments: dict[str, slice] = {}

        cursor = 0
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img: Tensor) -> Tensor:
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
            cursor += num_img_embs

        def lang_embed_func(lang_tokens: Tensor) -> Tensor:
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        cursor += lang_emb.shape[1]

        if subtask_tokens is not None:
            assert subtask_masks is not None
            subtask_emb = self._apply_checkpoint(lang_embed_func, subtask_tokens)
            embs.append(subtask_emb)
            pad_masks.append(subtask_masks)
            seg_len = subtask_emb.shape[1]
            # Each subtask token advances the cumsum so within-segment attention
            # is causal — required for the LM-style subtask CE loss to avoid
            # leaking the target token via bidirectional attention.
            att_masks += [1] * seg_len
            segments["subtask"] = slice(cursor, cursor + seg_len)
            cursor += seg_len

        if fast_tokens is not None:
            assert fast_masks is not None
            fast_emb = self._apply_checkpoint(lang_embed_func, fast_tokens)
            embs.append(fast_emb)
            pad_masks.append(fast_masks)
            seg_len = fast_emb.shape[1]
            # Same causal-within-segment rule for FAST: the FAST CE loss
            # would learn a degenerate "echo" solution under bidirectional
            # attention.
            att_masks += [1] * seg_len
            segments["fast"] = slice(cursor, cursor + seg_len)
            cursor += seg_len

        embs_t = torch.cat(embs, dim=1)
        pad_masks_t = torch.cat(pad_masks, dim=1).to(dtype=torch.bool)
        att_masks_t = torch.tensor(
            att_masks, dtype=torch.bool, device=pad_masks_t.device
        )
        att_masks_t = _align_mask_length(att_masks_t, pad_masks_t.shape[1])
        bsize = pad_masks_t.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, att_masks_t.shape[0])
        return embs_t, pad_masks_t, att_masks_t, segments

    def _embed_suffix(
        self, noisy_actions: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embed noisy actions and timestep for the action expert.

        Args:
            noisy_actions: Noisy action sequence [B, chunk_size, action_dim]
            timestep: Diffusion timestep [B]

        Returns:
            Tuple of (embeddings, padding_masks, attention_masks, adarms_cond).
        """
        embs = []
        pad_masks = []
        att_masks = []

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

        def time_mlp_func(time_emb: Tensor) -> Tensor:
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

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
        subtask_tokens: Tensor,
        subtask_masks: Tensor,
        fast_tokens: Tensor,
        fast_masks: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the three losses for pi05_full training.

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of image masks [B] per camera
            lang_tokens: Language token IDs [B, L]
            lang_masks: Language attention mask [B, L]
            subtask_tokens: Subtask token IDs [B, L_sub]
            subtask_masks: Subtask attention mask [B, L_sub]
            fast_tokens: FAST action token IDs [B, L_fast]
            fast_masks: FAST attention mask [B, L_fast]
            actions: Target action sequence [B, chunk_size, action_dim]
            noise: Optional pre-sampled noise
            time: Optional pre-sampled diffusion time

        Returns:
            Dict with keys flow_mse_loss, subtask_ce_loss, fast_ce_loss, loss.
            All values are scalar tensors.
        """
        if noise is None:
            noise = self._sample_noise(actions.shape, actions.device)
        if time is None:
            time = self._sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, segments = self._embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
            fast_tokens=fast_tokens,
            fast_masks=fast_masks,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self._embed_suffix(x_t, time)
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

        # Suffix MUST NOT attend to FAST tokens. Zero out the attention from
        # suffix positions to fast-segment positions in the 2D mask.
        att_2d_masks = _make_att_2d_masks(pad_masks, att_masks)
        if "fast" in segments:
            fast_slice = segments["fast"]
            suffix_start = pad_masks.shape[1] - suffix_pad_masks.shape[1]
            att_2d_masks[:, suffix_start:, fast_slice] = False

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs: Tensor,
            suffix_embs: Tensor,
            att_2d_masks_4d: Tensor,
            position_ids: Tensor,
            adarms_cond: Tensor | None,
        ) -> tuple[Tensor, Tensor]:
            outs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
                knowledge_insulation=self.config.knowledge_insulation,
            )
            return outs[0], outs[1]

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)

        # Flow matching MSE
        def action_out_proj_func(suffix_out: Tensor) -> Tensor:
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        flow_mse_loss = F.mse_loss(u_t, v_t, reduction="none").mean()

        # Subtask CE via tied LM head
        lm_head = self.paligemma_with_expert.paligemma.lm_head
        subtask_slice = segments["subtask"]
        subtask_hidden = prefix_out[:, subtask_slice, :].to(dtype=torch.float32)
        subtask_logits = lm_head(subtask_hidden)
        subtask_ce_loss = self._token_ce(subtask_logits, subtask_tokens, subtask_masks)

        # FAST CE via tied LM head
        fast_slice = segments["fast"]
        fast_hidden = prefix_out[:, fast_slice, :].to(dtype=torch.float32)
        fast_logits = lm_head(fast_hidden)
        fast_ce_loss = self._token_ce(fast_logits, fast_tokens, fast_masks)

        loss = (
            self.config.flow_matching_loss_weight * flow_mse_loss
            + self.config.subtask_loss_weight * subtask_ce_loss
            + self.config.fast_token_loss_weight * fast_ce_loss
        )
        return {
            "flow_mse_loss": flow_mse_loss,
            "subtask_ce_loss": subtask_ce_loss,
            "fast_ce_loss": fast_ce_loss,
            "loss": loss,
        }

    @staticmethod
    def _token_ce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Shifted cross-entropy with right-padding mask, computed in float32.

        Predicts targets[:, 1:] from logits[:, :-1] (causal shift).

        Args:
            logits: Predicted logits [B, L, vocab_size]
            targets: Target token IDs [B, L]
            mask: Boolean mask [B, L] where True marks valid (non-pad) tokens

        Returns:
            Scalar loss averaged over the masked target positions.
        """
        if logits.shape[1] < 2:
            return torch.zeros((), device=logits.device, dtype=torch.float32)
        logits_for_pred = logits[:, :-1, :].contiguous()
        targets_for_pred = targets[:, 1:].contiguous()
        mask_for_pred = mask[:, 1:].contiguous().to(dtype=torch.float32)
        per_token = (
            F.cross_entropy(
                logits_for_pred.view(-1, logits_for_pred.shape[-1]),
                targets_for_pred.view(-1),
                reduction="none",
            )
            .view_as(targets_for_pred)
            .to(dtype=torch.float32)
        )
        weighted = per_token * mask_for_pred
        denom = mask_for_pred.sum().clamp(min=1.0)
        return weighted.sum() / denom

    @torch.no_grad()
    def generate_subtask_tokens(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        bos_token_id: int,
        eos_token_id: int | None = None,
        loc_token_id: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Autoregressively generate subtask tokens.

        Args:
            images: List of image tensors [B, C, H, W] per camera
            img_masks: List of image masks [B] per camera
            lang_tokens: Language token IDs [B, L]
            lang_masks: Language attention mask [B, L]
            bos_token_id: PaliGemma BOS token to seed generation.
            eos_token_id: Optional EOS to halt early per-batch-item.
            loc_token_id: Optional first <loc####> id; all token ids >= this are
                masked out before sampling so we never emit visual-grounding tokens.

        Returns:
            Tuple of:
            - generated_tokens: (B, L) int64 — starts with BOS, ends at EOS or cap.
            - masks: (B, L) bool — True for valid (non-pad) tokens.
        """
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        max_steps = self.config.max_decoding_steps
        temperature = self.config.subtask_temperature

        # Phase A: prefill cache with [images, language, BOS]
        bos_col = torch.full((bsize, 1), bos_token_id, dtype=torch.long, device=device)
        bos_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            subtask_tokens=bos_col,
            subtask_masks=bos_mask,
        )
        att_2d_masks = _make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        paligemma_lm_config = self.paligemma_with_expert.paligemma.language_model.config
        paligemma_lm_config._attn_implementation = "eager"

        outs, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        prefix_out = outs[0]
        assert prefix_out is not None

        lm_head = self.paligemma_with_expert.paligemma.lm_head

        def _sample(logits: Tensor) -> Tensor:
            if loc_token_id is not None:
                logits[:, loc_token_id:] = float("-inf")
            if temperature == 0.0:
                return torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)

        first_logits = lm_head(prefix_out[:, -1:, :].to(dtype=torch.float32))[:, -1]
        next_tok = _sample(first_logits)

        generated = [bos_col, next_tok[:, None]]
        finished = torch.zeros(bsize, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            finished |= next_tok == eos_token_id

        # Phase B: autoregressive decode. Keep a running pad mask so position_ids
        # stay correct.
        running_pad = torch.cat([prefix_pad_masks, bos_mask], dim=1)
        for _ in range(max_steps - 1):
            if finished.all():
                break
            tok_emb = self.paligemma_with_expert.embed_language_tokens(
                next_tok[:, None]
            )
            tok_emb = tok_emb * math.sqrt(tok_emb.shape[-1])

            running_pad = torch.cat(
                [
                    running_pad,
                    torch.ones(bsize, 1, dtype=torch.bool, device=device),
                ],
                dim=1,
            )
            position_ids = (running_pad.long().cumsum(dim=1) - 1)[:, -1:]
            # Single-token attention mask: 2D shape (B, 1, total_len); every
            # position is allowed (the kv_padding gating from running_pad is
            # already implicitly true since we pass real tokens only).
            step_pad_2d = running_pad[:, None, :].expand(bsize, 1, running_pad.shape[1])
            step_pad_4d = self._prepare_attention_masks_4d(step_pad_2d)

            outs, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=step_pad_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[tok_emb, None],
                use_cache=True,
            )
            step_out = outs[0]
            assert step_out is not None
            step_logits = lm_head(step_out[:, -1:, :].to(dtype=torch.float32))[:, -1]
            next_tok = _sample(step_logits)
            # Force finished sequences to emit pad (id=0)
            next_tok = torch.where(finished, torch.zeros_like(next_tok), next_tok)
            if eos_token_id is not None:
                finished |= next_tok == eos_token_id
            generated.append(next_tok[:, None])

        generated_tokens = torch.cat(generated, dim=1)
        masks = generated_tokens != 0
        masks[:, 0] = True  # always keep BOS valid
        return generated_tokens, masks

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        subtask_tokens: Tensor | None = None,
        subtask_masks: Tensor | None = None,
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
            subtask_tokens: Optional subtask token IDs [B, L_sub] to condition on
            subtask_masks: Optional subtask attention mask [B, L_sub]
            noise: Optional initial noise
            num_steps: Number of Euler steps (default: config.num_inference_steps)

        Returns:
            Sampled action sequence [B, chunk_size, action_dim].
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self._sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self._embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            subtask_tokens=subtask_tokens,
            subtask_masks=subtask_masks,
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
        prefix_pad_masks: Tensor,
        past_key_values: list[torch.FloatTensor] | None,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Compute velocity field for a single Euler denoising step.

        Args:
            prefix_pad_masks: Padding masks from prefix embedding
            past_key_values: Cached key-values from prefix forward pass
            x_t: Current noisy actions [B, chunk_size, action_dim]
            timestep: Current diffusion time [B]

        Returns:
            Predicted velocity [B, chunk_size, action_dim].
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self._embed_suffix(x_t, timestep)
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
        config: PI05FullConfig | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> PI05FullPolicy:
        """Load a pretrained Pi05 model from HuggingFace Hub or local path.

        Args:
            pretrained_name_or_path: HuggingFace repo id or local path
            config: Model configuration (default: PI05FullConfig())
            strict: Whether to strictly enforce state dict loading
            **kwargs: Additional arguments (cache_dir, force_download, etc.)

        Returns:
            PI05FullPolicy model with loaded weights.
        """
        if pretrained_name_or_path is None:
            pretrained_name_or_path = "lerobot/pi05_base"
            logging.warning(
                "No pretrained model path provided; using default pi05_base model"
            )
        if config is None:
            config = PI05FullConfig()

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
            "Successfully loaded pretrained Pi05 weights from %s",
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

            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            if key.startswith("state_proj."):
                logging.warning("Skipping state_proj key in pi05 mode: %s", key)
                continue

            if "patch_embedding" in key:
                logging.warning("Vision embedding key might need handling: %s", key)

            fixed_state_dict[new_key] = value

        return fixed_state_dict
