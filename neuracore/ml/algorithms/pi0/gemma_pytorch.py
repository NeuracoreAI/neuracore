"""Minimal Gemma/PaliGemma helpers for PI0."""

# cspell:ignore adarms layernorm
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)


def _unpack_norm_output(
    norm_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize RMSNorm return values across patched/unpatched transformers."""
    if isinstance(norm_output, tuple):
        return norm_output
    return norm_output, None


def compute_shared_attention_layer(
    layer_idx: int,
    inputs_embeds: list[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    adarms_cond: list[torch.Tensor | None],
    paligemma: PaliGemmaForConditionalGeneration,
    gemma_expert: GemmaForCausalLM,
) -> list[torch.Tensor]:
    """Run a single transformer layer jointly across prefix/suffix branches.

    This function performs shared attention computation between the PaliGemma
    vision-language model and the Gemma action expert model. It concatenates
    the query, key, and value states from both models, applies rotary positional
    embeddings, computes attention, and then separates the outputs back to each
    model's branch.

    Args:
        layer_idx: Index of the transformer layer to process.
        inputs_embeds: List of two tensors containing hidden states for
            [prefix (PaliGemma), suffix (Gemma expert)] branches.
        attention_mask: Attention mask tensor of shape (batch_size, 1, seq_len, seq_len)
            for masking padded or future tokens.
        position_ids: Position indices tensor of shape (batch_size, seq_len) for
            rotary positional embeddings.
        adarms_cond: List of two optional conditioning tensors for adaptive RMS
            normalization, one per branch. None values indicate no conditioning.
        paligemma: The PaliGemma vision-language model instance.
        gemma_expert: The Gemma action expert model instance.

    Returns:
        List of two tensors containing the output hidden states for each branch
        after attention and feed-forward processing.
    """
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        if adarms_cond[i] is None:
            norm_output = layer.input_layernorm(hidden_states)
        else:
            norm_output = layer.input_layernorm(hidden_states, cond=adarms_cond[i])
        hidden_states, gate = _unpack_norm_output(norm_output)  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = (
            layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        key_state = (
            layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        value_state = (
            layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    num_heads = paligemma.language_model.layers[
        layer_idx
    ].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, num_heads * head_dim)
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        if adarms_cond[i] is None:
            norm_output = layer.post_attention_layernorm(out_emb)
        else:
            norm_output = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        out_emb, gate = _unpack_norm_output(norm_output)
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


@dataclass(slots=True)
class GemmaConfig:
    """Configuration for Gemma model variants.

    Attributes:
        variant: Model variant name (e.g., "gemma_300m", "gemma_2b", "gemma_tiny").
        width: Hidden size of the model (embedding dimension).
        depth: Number of transformer decoder layers.
        mlp_dim: Intermediate size of the feed-forward network.
        num_heads: Number of attention heads.
        num_kv_heads: Number of key-value heads for grouped query attention.
        head_dim: Dimension of each attention head.
    """

    variant: str
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_config(variant: str) -> GemmaConfig:
    """Return the GemmaConfig for a known Gemma model variant.

    Provides predefined configurations for different Gemma model sizes used
    in the PI0 architecture.

    Args:
        variant: The model variant name. Supported values are:
            - "gemma_300m": Small 300M parameter model with width=1024, depth=18.
            - "gemma_2b": Larger 2B parameter model with width=2048, depth=18.
            - "gemma_tiny": Tiny model for testing with width=16, depth=4.

    Returns:
        A GemmaConfig dataclass containing the model hyperparameters.

    Raises:
        ValueError: If the variant name is not recognized.
    """
    if variant == "gemma_300m":
        return GemmaConfig(
            variant="gemma_300m",
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaConfig(
            variant="gemma_2b",
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_tiny":
        return GemmaConfig(
            variant="gemma_tiny",
            width=16,
            depth=4,
            mlp_dim=64,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
        )
    raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for PI0."""

    def __init__(
        self,
        vlm_config: GemmaConfig,
        action_expert_config: GemmaConfig,
        use_adarms: tuple[bool, bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ) -> None:
        """Initialize the joint vision-language and action expert model.

        Creates a PaliGemma vision-language model alongside a separate Gemma
        action expert model. These models share attention computation during
        forward passes for efficient multi-modal reasoning in the PI0 architecture.

        Args:
            vlm_config: Configuration for the vision-language model (PaliGemma),
                specifying dimensions, depth, and attention parameters.
            action_expert_config: Configuration for the action expert model (Gemma),
                which processes action-related tokens.
            use_adarms: Optional tuple of two booleans indicating whether to use
                adaptive RMS normalization for (VLM, action expert) respectively.
                Defaults to (False, False) if not provided.
            precision: Model precision, either "bfloat16" for mixed precision
                training/inference or "float32" for full precision. Defaults to
                "bfloat16".
        """
        if use_adarms is None:
            use_adarms = (False, False)
        super().__init__()

        paligemma_config = CONFIG_MAPPING["paligemma"]()
        paligemma_config._vocab_size = 257152  # noqa: SLF001
        paligemma_config.image_token_index = 257152
        paligemma_config.text_config.hidden_size = vlm_config.width
        paligemma_config.text_config.intermediate_size = vlm_config.mlp_dim
        paligemma_config.text_config.num_attention_heads = vlm_config.num_heads
        paligemma_config.text_config.head_dim = vlm_config.head_dim
        paligemma_config.text_config.num_hidden_layers = vlm_config.depth
        paligemma_config.text_config.num_key_value_heads = vlm_config.num_kv_heads
        paligemma_config.text_config.hidden_activation = "gelu_pytorch_tanh"
        paligemma_config.text_config.torch_dtype = "float32"
        paligemma_config.text_config.vocab_size = 257152
        paligemma_config.text_config.use_adarms = use_adarms[0]
        paligemma_config.text_config.adarms_cond_dim = (
            vlm_config.width if use_adarms[0] else None
        )
        paligemma_config.vision_config.intermediate_size = 4304
        # Keep pretrained-compatible projection size for real models, but make
        # tiny variant align vision/text dims to avoid concat shape mismatch.
        if vlm_config.variant == "gemma_tiny":
            paligemma_config.vision_config.projection_dim = vlm_config.width
        else:
            paligemma_config.vision_config.projection_dim = 2048
        paligemma_config.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config.vision_config.torch_dtype = "float32"

        action_expert_config_gemma = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_gemma)
        self.gemma_expert.model.embed_tokens = None
        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ) -> None:
        """Move parameters to bfloat16, keeping sensitive ones in float32.

        Converts most model parameters to bfloat16 for memory efficiency while
        preserving numerical stability by keeping certain sensitive parameters
        in float32. Parameters kept in float32 include vision embeddings
        (patch and position) and layer normalization weights.

        Args:
            precision: Target precision for the model. If "bfloat16", converts
                most parameters while keeping sensitive ones in float32. If
                "float32", keeps all parameters in full precision.

        Raises:
            ValueError: If precision is not "bfloat16" or "float32".
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed an input image using the vision tower.

        Processes raw image pixels through the PaliGemma vision encoder to
        produce visual embeddings that can be concatenated with language tokens.

        Args:
            image: Input image tensor of shape (batch_size, channels, height, width),
                typically normalized to the expected range for the vision model.

        Returns:
            Image embeddings tensor of shape (batch_size, num_patches, embed_dim)
            representing the encoded visual features.
        """
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens with the language model embedding layer.

        Converts discrete token IDs into continuous embeddings using the
        PaliGemma language model's embedding table.

        Args:
            tokens: Token IDs tensor of shape (batch_size, seq_len) containing
                integer indices into the vocabulary.

        Returns:
            Token embeddings tensor of shape (batch_size, seq_len, embed_dim)
            representing the embedded language tokens.
        """
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor | None] | None = None,
    ) -> tuple[list[torch.Tensor | None], list[torch.FloatTensor] | None]:
        """Forward pass for prefix (vision/lang) and suffix (action) branches.

        Processes input embeddings through the joint PaliGemma and Gemma expert
        architecture. Supports three modes of operation:
        1. Prefix-only: When suffix embeddings are None, runs only PaliGemma.
        2. Suffix-only: When prefix embeddings are None, runs only Gemma expert.
        3. Joint: When both are provided, runs shared attention across both models.

        Args:
            attention_mask: Optional mask tensor of shape
                (batch_size, 1, seq_len, seq_len) to prevent attention to
                certain positions (e.g., padding, future tokens).
            position_ids: Optional position indices of shape (batch_size, seq_len) for
                rotary positional embeddings.
            past_key_values: Optional cached key-value states from previous forward
                passes for efficient autoregressive generation.
            inputs_embeds: List of two tensors [prefix_embeds, suffix_embeds] where
                each is of shape (batch_size, seq_len, embed_dim). One can be None
                to run only the other branch.
            use_cache: Whether to return updated key-value cache for generation.
            adarms_cond: Optional list of two conditioning tensors for adaptive RMS
                normalization, one per branch [VLM_cond, expert_cond].

        Returns:
            A tuple containing:
            - List of two output tensors [prefix_output, suffix_output], where each
              is the final hidden states of shape (batch_size, seq_len, embed_dim).
              One may be None if that branch was not computed.
            - Updated past_key_values cache if use_cache=True and running prefix-only,
              otherwise None.

        Raises:
            ValueError: If inputs_embeds is None.
        """
        adarms: list[torch.Tensor | None]
        if adarms_cond is None:
            adarms = [None, None]
        else:
            adarms = adarms_cond
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms[0],
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms[1],
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_shared_attention_layer,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_shared_attention_layer(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            def compute_final_norms(
                inputs_embeds: list[torch.Tensor],
                adarms_cond: list[torch.Tensor | None],
            ) -> list[torch.Tensor]:
                """Apply final layer normalization to both model branches.

                Performs the final RMS normalization step after all transformer
                layers have been processed. Supports optional adaptive conditioning.

                Args:
                    inputs_embeds: List of hidden state tensors for each branch,
                        each of shape (batch_size, seq_len, embed_dim).
                    adarms_cond: List of optional conditioning tensors for adaptive
                        RMS normalization, one per branch.

                Returns:
                    List of normalized output tensors, one per branch.
                """
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
