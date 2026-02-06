"""Gemma configuration for the PI0 transformers library modification.

This file started life as auto-generated code from the upstream transformers
Gemma configuration and is now maintained here and adapted for the Neuracore PI0
implementation.
"""

from collections.abc import Callable
from typing import Any

from ...configuration_utils import PretrainedConfig


class GemmaConfig(PretrainedConfig):
    """Configuration for [`GemmaModel`].

    This stores the configuration of a [`GemmaModel`] and is used to
    instantiate a model according to the specified arguments, defining the
    model architecture. Instantiating a configuration with the defaults yields
    a similar configuration to Gemma-7B, e.g.
    [google/gemma-7b](https://huggingface.co/google/gemma-7b).
    Configuration objects inherit from [`PretrainedConfig`] and can be used to
    control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 256000):
            Vocabulary size of the Gemma model. Defines the number of different
            tokens that can be represented by the `inputs_ids` passed when
            calling [`GemmaModel`].
        hidden_size (`int`, *optional*, defaults to 3072):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the
            Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key/value heads used to implement Grouped Query Attention.
            If `num_key_value_heads=num_attention_heads`, the model will use
            Multi Head Attention (MHA). If `num_key_value_heads=1` the model will
            use Multi Query Attention (MQA), otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group
            key and value head should be constructed by meanpooling all the
            original heads within that group. For more details, see
            [this paper](https://huggingface.co/papers/2305.13245). If it is not
            specified, defaults to `num_attention_heads`.
        head_dim (`int`, *optional*, defaults to 256):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The legacy activation function. It is overwritten by the
            `hidden_activation`.
        hidden_activation (`str` or `function`, *optional*):
            The non-linear activation function (function or string) in the
            decoder. Defaults to `"gelu_pytorch_tanh"` if not specified.
            `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"`
            activation function.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/value attentions
            (not used by all models). Only relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        bos_token_id (`int`, *optional*, defaults to 2):
            Beginning of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie weight embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection
            layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_adarms (`bool`, *optional*, defaults to `False`):
            Whether to use ADARMS.
        adarms_cond_dim (`int`, *optional*, defaults to `None`):
            The dimension of the ADARMS condition.
    ```python
    >>> from transformers import GemmaModel, GemmaConfig
    >>> # Initializing a Gemma gemma-7b style configuration
    >>> configuration = GemmaConfig()
    >>> # Initializing a model from the gemma-7b style configuration
    >>> model = GemmaModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "gemma"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 256000,
        hidden_size: int = 3072,
        intermediate_size: int = 24576,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        head_dim: int = 256,
        hidden_act: str | Callable = "gelu_pytorch_tanh",
        hidden_activation: str | Callable | None = None,
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        bos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_adarms: bool = False,
        adarms_cond_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the configuration."""
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.hidden_activation = hidden_activation
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_adarms = use_adarms
        self.adarms_cond_dim = adarms_cond_dim

        # Set default for adarms_cond_dim if use_adarms is True
        if self.use_adarms and self.adarms_cond_dim is None:
            self.adarms_cond_dim = self.hidden_size

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["GemmaConfig"]
