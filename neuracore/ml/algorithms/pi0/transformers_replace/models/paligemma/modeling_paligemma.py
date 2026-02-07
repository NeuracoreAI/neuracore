"""PyTorch PaliGemma model implementation.

This module implements the PaliGemma vision-language model adapted for the
Neuracore PI0 algorithm. PaliGemma combines a vision encoder (SigLIP) with a
language model (Gemma) to enable vision-language understanding and generation.

The implementation includes:
- PaliGemmaMultiModalProjector: Projects vision features to language model space
- PaliGemmaModel: Base model with vision and language backbones
- PaliGemmaForConditionalGeneration: Full model with language modeling head
- Support for prefix attention, causal masking, and efficient caching

This file started life as auto-generated code from the upstream transformers
PaliGemma model and is now maintained here and adapted for the Neuracore PI0
implementation.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.utils.checkpoint
from torch import nn

from ...cache_utils import Cache, HybridCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    LossKwargs,
    ModelOutput,
    auto_docstring,
    can_return_tuple,
    is_torchdynamo_compiling,
    logging,
)
from ..auto import AutoModel
from .configuration_paligemma import PaliGemmaConfig

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Paligemma outputs, with hidden states and attentions.
    """
)
class PaligemmaModelOutputWithPast(BaseModelOutputWithPast):
    """Base class for Paligemma outputs with past key values.

    Args:
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned
            when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`,
            with each tuple having 2 tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`. Contains pre-computed
            hidden-states (key and values in the self-attention blocks) that can
            be used (see `past_key_values` input) to speed up sequential
            decoding.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images,
            sequence_length, hidden_size)`. Image hidden states of the model
            produced by the vision encoder and after projecting the last hidden
            state.
    """

    image_hidden_states: torch.FloatTensor | None = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for PaliGemma causal language model (or autoregressive) outputs.
    """
)
class PaliGemmaCausalLMOutputWithPast(ModelOutput):
    """Outputs for Paligemma causal language modeling.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when
            `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,
            config.text_config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each
            vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned
            when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`,
            with each tuple having 2 tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`. Contains pre-computed
            hidden-states (key and values in the self-attention blocks) that can
            be used (see `past_key_values` input) to speed up sequential
            decoding.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images,
            sequence_length, hidden_size)`. Image hidden states of the model
            produced by the vision encoder after projecting last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class PaliGemmaMultiModalProjector(nn.Module):
    """Multi-modal projector for aligning vision and language features.

    This module projects vision encoder features to the language model's
    embedding space, enabling the language model to process visual information.

    Args:
        config: PaliGemma configuration containing vision and projection parameters
    """

    def __init__(self, config: PaliGemmaConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project image features to language model embedding space.

        Args:
            image_features: Vision encoder features of shape
                [batch_size, num_patches, vision_hidden_size]

        Returns:
            Projected features of shape [batch_size, num_patches, projection_dim]
        """
        hidden_states = self.linear(image_features)

        return hidden_states


@auto_docstring
class PaliGemmaPreTrainedModel(PreTrainedModel):
    """Base class for PaliGemma models.

    This class provides common functionality for all PaliGemma model variants,
    including weight initialization and support for various attention backends
    and optimization features. Note that this ported version is intended for
    inference and fine-tuning, not training from scratch.
    """

    config_class = PaliGemmaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["PaliGemmaMultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for different module types.

        Note: This ported version of PaliGemma isn't meant for training from
        scratch - only inference and fine-tuning.

        Args:
            module: PyTorch module to initialize
        """
        # important: this ported version of PaliGemma isn't meant for training
        # from scratch - only inference and fine-tuning
        std = getattr(
            self.config,
            "initializer_range",
            self.config.get_text_config().initializer_range,
        )

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()


@auto_docstring(
    custom_intro="""
    The base Paligemma model which consists of a vision backbone and a language
    model without a language modeling head.
    """
)
class PaliGemmaModel(PaliGemmaPreTrainedModel):
    """PaliGemma model with vision and language backbones."""

    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}
    # We are filtering the logits/labels so we shouldn't divide the loss based
    # on num_items_in_batch.
    accepts_loss_kwargs = False

    def __init__(self, config: PaliGemmaConfig) -> None:
        """Initialize the PaliGemma model.

        Args:
            config: PaliGemma configuration containing vision and text model
                hyperparameters
        """
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        language_model = AutoModel.from_config(config=config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return the language model input embeddings.

        Returns:
            Embedding layer for input tokens
        """
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the language model input embeddings.

        Args:
            value: New embedding layer to use for input tokens
        """
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder: nn.Module) -> None:
        """Set the language model decoder.

        Args:
            decoder: Language model decoder module
        """
        self.language_model = decoder

    def get_decoder(self) -> nn.Module:
        """Return the language model decoder.

        Returns:
            The underlying language model decoder
        """
        return self.language_model

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        input_tensor: torch.Tensor | None = None,
        is_training: bool | None = None,
    ) -> torch.Tensor | None:
        """Update causal attention mask for prefix attention.

        This method creates or updates a causal attention mask that supports:
        - Prefix attention: allows attending to image tokens during training
        - Causal masking: prevents attending to future tokens
        - Padding masking: masks padding tokens
        - Cache-aware masking: handles static and dynamic caches

        Args:
            attention_mask: Optional 2D or 4D attention mask
            token_type_ids: Optional token type IDs for prefix attention
                (required during training)
            past_key_values: Optional cache for key/value states
            cache_position: Optional position indices for cache updates
            input_tensor: Optional input tensor to determine sequence length
            is_training: Optional flag indicating training mode

        Returns:
            Updated 4D causal attention mask of shape
            [batch_size, 1, seq_len, target_len] or None if flash attention
            is used and mask is not needed
        """
        if self.config.text_config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        is_training = is_training if is_training is not None else self.training
        using_static_cache = isinstance(past_key_values, StaticCache)
        if using_static_cache:
            assert past_key_values is not None
        min_dtype = torch.finfo(self.dtype).min
        if input_tensor is None:
            input_tensor = attention_mask
        assert input_tensor is not None

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
            if past_key_values is None:
                raise ValueError("past_key_values must be provided for static cache.")
            static_cache = cast(StaticCache, past_key_values)
            target_length = static_cache.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            hybrid_cache = cast(HybridCache, past_key_values)
            target_length = hybrid_cache.get_max_cache_shape()
        else:
            if isinstance(attention_mask, torch.Tensor):
                target_length = attention_mask.shape[-1]
            else:
                if cache_position is None:
                    raise ValueError(
                        "cache_position must be provided without attention_mask."
                    )
                target_length = cache_position[0] + sequence_length + 1

        if attention_mask is not None and attention_mask.dim() == 4:
            # The mask comes already in inverted form and requires no inversion
            # or slicing.
            return attention_mask

        if cache_position is None:
            raise ValueError("cache_position must be provided to build a causal mask.")
        cache_position_tensor = cache_position
        causal_mask: torch.Tensor
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=self.dtype,
            device=cache_position_tensor.device,
        )
        # Causal diagonal mask only if training, otherwise attend to the whole
        # prefix. Training-specific attention for prefix is handled below.
        if sequence_length != 1:
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(
            target_length, device=cache_position_tensor.device
        ) > cache_position_tensor.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = (
                causal_mask.clone()
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # First unmask prefix tokens during training
            if is_training:
                if token_type_ids is None:
                    raise ValueError("Token type ids must be provided during training")
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                :, None, None, :
            ].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[
                :, :, :, :mask_length
            ].masked_fill(padding_mask, min_dtype)

        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Obtain image features from the vision tower.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, channels,
                height, width)`):
                The tensors corresponding to the input images.

        Returns:
            image_features (`torch.Tensor`):
                Image feature tensor of shape `(num_images, image_length,
                embed_dim)`.
        """
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | PaligemmaModelOutputWithPast:
        """Run the forward pass for the multi-modal model.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs.
            pixel_values (`torch.FloatTensor`, *optional*):
                Image pixel values.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for input tokens.
            position_ids (`torch.LongTensor`, *optional*):
                Position indices for the input tokens.
            past_key_values (`Cache`, *optional*):
                Cached key/value states for faster decoding.
            token_type_ids (`torch.LongTensor`, *optional*):
                Token type ids for prefix attention.
            cache_position (`torch.LongTensor`, *optional*):
                Cache positions for decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Precomputed input embeddings.
            labels (`torch.LongTensor`, *optional*):
                Labels for computing the masked language modeling loss. Indices
                should either be in `[0, ..., config.text_config.vocab_size]` or
                -100 (see `input_ids` docstring). Tokens with indices set to
                `-100` are ignored (masked), so the loss is computed only for
                tokens with labels in `[0, ..., config.text_config.vocab_size]`.
            use_cache (`bool`, *optional*):
                Whether to return key/value cache.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            return_dict (`bool`, *optional*):
                Whether to return a ModelOutput dict.
            **kwargs:
                Additional attention-related keyword arguments.

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor
            >>> from transformers import PaliGemmaForConditionalGeneration

            >>> model = PaliGemmaForConditionalGeneration.from_pretrained(
            ...     "google/paligemma2-3b-mix-224"
            ... )
            >>> processor = AutoProcessor.from_pretrained(
            ...     "google/paligemma2-3b-mix-224"
            ... )

            >>> prompt = "Where is the cat standing?"
            >>> url = (
            ...     "https://huggingface.co/datasets/huggingface/"
            ...     "documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            ... )
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(**inputs)
            >>> processor.batch_decode(
            ...     generate_ids,
            ...     skip_special_tokens=True,
            ...     clean_up_tokenization_spaces=False,
            ... )[0]
            "Where is the cat standing? snow"
            ```
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        is_training = token_type_ids is not None and labels is not None

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = (
                cache_position.unsqueeze(0) + 1
            )  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.image_token_id,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
            else:
                special_image_mask = (
                    input_ids == self.config.image_token_id
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )

            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    "Number of images does not match number of special image "
                    "tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but "
                    f"{image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            inputs_embeds,
            is_training,
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return PaligemmaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs):
    """Type alias for keyword arguments accepted by PaliGemmaForConditionalGeneration.

    Combines flash attention kwargs and loss kwargs for type checking.
    """


@auto_docstring(
    custom_intro="""
    The base Paligemma model which consists of a vision backbone and a language
    model without language modeling head.
    """
)
class PaliGemmaForConditionalGeneration(PaliGemmaPreTrainedModel, GenerationMixin):
    """PaliGemma model with a conditional generation head."""

    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PaliGemmaConfig) -> None:
        """Initialize the conditional generation model.

        Args:
            config: PaliGemma configuration containing model hyperparameters
        """
        super().__init__(config)
        self.model = PaliGemmaModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return the input token embeddings.

        Returns:
            Embedding layer for input tokens
        """
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set the input token embeddings.

        Args:
            value: New embedding layer to use for input tokens
        """
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Linear:
        """Return the output token embeddings.

        Returns:
            Linear layer mapping hidden states to vocabulary logits
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        """Set the output token embeddings.

        Args:
            new_embeddings: New linear layer for output embeddings
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        """Set the language model decoder.

        Args:
            decoder: Language model decoder module
        """
        self.model.set_decoder(decoder)

    def get_decoder(self) -> nn.Module:
        """Return the language model decoder.

        Returns:
            The underlying language model decoder
        """
        return self.model.get_decoder()

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """Return image features from the vision tower.

        Args:
            pixel_values: Image pixel values of shape
                [batch_size, channels, height, width]

        Returns:
            Image feature tensor of shape [num_images, image_length, embed_dim]
        """
        return self.model.get_image_features(pixel_values)

    # Make modules available through conditional class for BC
    @property
    def language_model(self) -> nn.Module:
        """Return the language model module.

        Returns:
            The underlying language model (Gemma) decoder
        """
        return self.model.language_model

    @property
    def vision_tower(self) -> nn.Module:
        """Return the vision tower module.

        Returns:
            The vision encoder (SigLIP) model
        """
        return self.model.vision_tower

    @property
    def multi_modal_projector(self) -> nn.Module:
        """Return the multimodal projector module.

        Returns:
            The projector that maps vision features to language model space
        """
        return self.model.multi_modal_projector

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> tuple | PaliGemmaCausalLMOutputWithPast:
        """Run the forward pass for conditional generation.

        Args:
            input_ids (`torch.LongTensor`, *optional*):
                Input token IDs.
            pixel_values (`torch.FloatTensor`, *optional*):
                Image pixel values.
            attention_mask (`torch.Tensor`, *optional*):
                Attention mask for input tokens.
            position_ids (`torch.LongTensor`, *optional*):
                Position indices for the input tokens.
            past_key_values (`Cache`, *optional*):
                Cached key/value states for faster decoding.
            token_type_ids (`torch.LongTensor`, *optional*):
                Token type ids for prefix attention.
            cache_position (`torch.LongTensor`, *optional*):
                Cache positions for decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Precomputed input embeddings.
            labels (`torch.LongTensor`, *optional*):
                Labels for computing the masked language modeling loss. Indices
                should either be in `[0, ..., config.text_config.vocab_size]` or
                -100 (see `input_ids` docstring). Tokens with indices set to
                `-100` are ignored (masked), so the loss is computed only for
                tokens with labels in `[0, ..., config.text_config.vocab_size]`.
            use_cache (`bool`, *optional*):
                Whether to return key/value cache.
            output_attentions (`bool`, *optional*):
                Whether to return attention weights.
            output_hidden_states (`bool`, *optional*):
                Whether to return hidden states.
            return_dict (`bool`, *optional*):
                Whether to return a ModelOutput dict.
            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                Number of logits to compute or indices to keep.
            **kwargs:
                Additional attention or loss-related keyword arguments.

        Example:
            ```python
            >>> from PIL import Image
            >>> import requests
            >>> from transformers import AutoProcessor
            >>> from transformers import PaliGemmaForConditionalGeneration

            >>> model = PaliGemmaForConditionalGeneration.from_pretrained(
            ...     "google/paligemma2-3b-mix-224"
            ... )
            >>> processor = AutoProcessor.from_pretrained(
            ...     "google/paligemma2-3b-mix-224"
            ... )

            >>> prompt = "Where is the cat standing?"
            >>> url = (
            ...     "https://huggingface.co/datasets/huggingface/"
            ...     "documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
            ... )
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(**inputs)
            >>> processor.batch_decode(
            ...     generate_ids,
            ...     skip_special_tokens=True,
            ...     clean_up_tokenization_spaces=False,
            ... )[0]
            "Where is the cat standing? snow"
            ```
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we
        # are not computing the loss.
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return PaliGemmaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        use_cache: bool = True,
        logits_to_keep: int | torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for generation with image-aware defaults.

        This method handles special requirements for PaliGemma generation:
        - Position IDs are 1-indexed (unlike standard transformers)
        - Pixel values are only needed at the first cache position
        - Causal mask is updated for hybrid cache scenarios

        Args:
            input_ids: Input token IDs
            past_key_values: Optional cache for key/value states
            inputs_embeds: Optional precomputed input embeddings
            cache_position: Optional position indices for cache updates
            position_ids: Optional position indices (will be adjusted to 1-indexed)
            pixel_values: Optional image pixel values (only used at cache start)
            attention_mask: Optional attention mask
            token_type_ids: Optional token type IDs for prefix attention
            use_cache: Whether to use caching
            logits_to_keep: Optional number of logits to compute
            labels: Optional labels for training
            **kwargs: Additional generation arguments

        Returns:
            Dictionary of prepared inputs for the model forward pass
        """
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        if cache_position is None:
            return model_inputs
        # If we're in cached decoding stage, pixel values should be None because
        # input ids do not contain special image token anymore.
        # Otherwise we need pixel values to be passed to model. NOTE:
        # use_cache=False needs pixel_values always.
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self.model._update_causal_mask(
                attention_mask,
                token_type_ids,
                past_key_values,
                cache_position,
                input_tensor,
                is_training,
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Create a 4D causal attention mask.

        Creates a causal 4D mask of shape `(batch_size, 1, query_length,
        key_value_length)` from a 2D mask of shape `(batch_size,
        key_value_length)`, or if the input `attention_mask` is already 4D, does
        nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or
                a 4D attention mask of shape `(batch_size, 1, query_length,
                key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length. When generating with static cache, the mask
                should be as long as the static cache to account for the 0
                padding and the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in
                the sequence.
            batch_size (`int`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # The mask comes already in inverted form and requires no inversion
            # or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=cache_position.device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
                    :, None, None, :
                ].to(causal_mask.device)
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)

        return causal_mask


__all__ = [
    "PaliGemmaForConditionalGeneration",
    "PaliGemmaPreTrainedModel",
    "PaliGemmaModel",
]
