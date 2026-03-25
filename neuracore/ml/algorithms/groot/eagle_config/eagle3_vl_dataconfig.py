"""Dataclass representations of the Eagle3-VL config.json.

These dataclasses serve as the single source of truth for the default
Eagle3-VL configuration values. They provide type safety, IDE support,
and validation while remaining compatible with HuggingFace's
``AutoConfig.from_pretrained`` pipeline.

Usage::

    from .eagle3_vl_dataconfig import Eagle3VLDataConfig

    # Use all defaults
    cfg = Eagle3VLDataConfig()

    # Override specific fields
    cfg = Eagle3VLDataConfig(select_layer=-2, downsample_ratio=0.25)

    # Convert to dict (for Eagle3_VLConfig or JSON serialisation)
    d = cfg.to_dict()

    # Write config.json
    cfg.write_json("path/to/eagle_config/config.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Siglip2VisionDataConfig:
    """Vision encoder (SigLIP-2) configuration."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    num_patches: int = 256
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    window_size: int = 14
    full_attention_indexes: list[int] = field(default_factory=lambda: [7, 14, 21, 26])
    use_rope: bool = False
    use_windows_attn: bool = False
    model_type: str = "siglip2_vision_model"
    torch_dtype: str = "bfloat16"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for ``Siglip2VisionConfig(**d)``."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Qwen3TextDataConfig:
    """Text decoder (Qwen3) configuration."""

    _name_or_path: str = "Qwen/Qwen3-1.7B"
    architectures: list[str] = field(default_factory=lambda: ["Qwen3ForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 6144
    max_position_embeddings: int = 40960
    max_window_layers: int = 28
    model_type: str = "qwen3"
    num_attention_heads: int = 16
    num_hidden_layers: int = 28
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_scaling: Any = None
    rope_theta: int = 1000000
    sliding_window: Any = None
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = False
    use_sliding_window: bool = False
    vocab_size: int = 151680

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for ``Qwen3Config(**d)``."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Eagle3VLDataConfig:
    """Top-level Eagle3-VL model configuration.

    This dataclass mirrors the ``config.json`` shipped with the bundled
    Eagle-Block2A-2B-v2 weights and acts as the canonical source of truth
    for default values.
    """

    # HuggingFace auto-map (needed for ``AutoConfig`` / ``AutoModel`` resolution)
    auto_map: dict[str, str] = field(
        default_factory=lambda: {
            "AutoConfig": "configuration_eagle3_vl.Eagle3_VLConfig",
            "AutoModel": "modeling_eagle3_vl.Eagle3_VLForConditionalGeneration",
            "AutoModelForCausalLM": (
                "modeling_eagle3_vl.Eagle3_VLForConditionalGeneration"
            ),
        }
    )

    # Architecture identifiers
    architectures: list[str] = field(
        default_factory=lambda: ["Eagle3_VLForConditionalGeneration"]
    )
    model_type: str = "eagle_3_vl"

    # Vision / text sub-configs
    vision_config: Siglip2VisionDataConfig = field(
        default_factory=Siglip2VisionDataConfig
    )
    text_config: Qwen3TextDataConfig = field(default_factory=Qwen3TextDataConfig)

    # Model hyper-parameters
    downsample_ratio: float = 0.5
    dynamic_image_size: bool = False
    image_token_index: int = 151669
    loss_version: str = "efficient_v2_cp_head"
    max_dynamic_tiles: int = 12
    min_dynamic_tiles: int = 1
    mlp_checkpoint: bool = False
    mlp_connector_layers: int = 2
    pad2square: bool = False
    select_layer: int = -1
    use_backbone_lora: int = 0
    use_llm_lora: int = 0
    use_pixel_shuffle: bool = True
    use_thumbnail: bool = False

    # Attention
    _attn_implementation: str = "sdpa"

    # Serialisation metadata
    torch_dtype: str = "bfloat16"
    transformers_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a nested dict matching the ``config.json`` schema."""
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (Siglip2VisionDataConfig, Qwen3TextDataConfig)):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def write_json(self, path: str | Path) -> None:
        """Write the configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Eagle3VLDataConfig:
        """Construct from a plain dict (e.g. parsed JSON).

        Unknown keys in the top-level dict or sub-config dicts are
        silently ignored so that HuggingFace-generated JSON (which may
        contain extra fields like ``_attn_implementation_autoset``) can
        be loaded without error.
        """
        vision = data.pop("vision_config", {})
        text = data.pop("text_config", {})
        vision_fields = {f for f in Siglip2VisionDataConfig.__dataclass_fields__}
        text_fields = {f for f in Qwen3TextDataConfig.__dataclass_fields__}
        return cls(
            vision_config=Siglip2VisionDataConfig(
                **{k: v for k, v in vision.items() if k in vision_fields}
            ),
            text_config=Qwen3TextDataConfig(
                **{k: v for k, v in text.items() if k in text_fields}
            ),
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Eagle3VLDataConfig:
        """Load from a ``config.json`` file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
