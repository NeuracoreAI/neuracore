"""Minimal Gemma/PaliGemma helpers for PI0."""

# cspell:ignore adarms layernorm
from typing import Literal

import torch
import torch.nn as nn
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
from transformers.models.paligemma.modeling_paligemma import (
    PaliGemmaForConditionalGeneration,
)

from neuracore.ml.algorithms.pi0.modules import compute_layer_complete


class GemmaConfig:
    """Configuration for Gemma model variants."""

    def __init__(
        self,
        width: int,
        depth: int,
        mlp_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize a Gemma configuration."""
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:
    """Return the GemmaConfig for a known variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
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
        """Initialize the joint vision-language and action expert model."""
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
        """Move parameters to bfloat16, keeping sensitive ones in float32."""
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
        """Embed an input image using the vision tower."""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens with the language model tokenizer."""
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
        """Forward pass for prefix (vision/lang) and suffix (action) branches."""
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
                        compute_layer_complete,
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
                    inputs_embeds = compute_layer_complete(
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
