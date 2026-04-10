"""Configuration for the Eagle3-VL model family."""

import copy
from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.utils import logging

from .eagle3_vl_dataconfig import Eagle3VLDataConfig
from .modeling_siglip2 import Siglip2VisionConfig

logger = logging.get_logger(__name__)

# Canonical defaults — single source of truth
_DEFAULTS = Eagle3VLDataConfig()

try:
    from transformers import InternVisionConfig  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    InternVisionConfig = None  # type: ignore[assignment]

try:
    from transformers import RADIOConfig  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    RADIOConfig = None  # type: ignore[assignment]


class Eagle3_VLConfig(PretrainedConfig):
    """Configuration class for the Eagle3-VL model."""

    model_type = "eagle_3_vl"
    is_composition = True
    sub_configs = {"vision_config": SiglipVisionConfig, "text_config": Qwen2Config}

    def __init__(
        self,
        vision_config: Any = None,
        text_config: Any = None,
        use_backbone_lora: int = _DEFAULTS.use_backbone_lora,
        use_llm_lora: int = _DEFAULTS.use_llm_lora,
        pad2square: bool = _DEFAULTS.pad2square,
        select_layer: int = _DEFAULTS.select_layer,
        downsample_ratio: float = _DEFAULTS.downsample_ratio,
        template: Any = None,
        loss_version: str = _DEFAULTS.loss_version,
        mlp_checkpoint: bool = _DEFAULTS.mlp_checkpoint,
        image_token_index: int = _DEFAULTS.image_token_index,
        **kwargs: Any,
    ) -> None:
        """Initialize Eagle3_VLConfig."""
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = _DEFAULTS.vision_config.to_dict()
            logger.info(
                "vision_config is None. "
                "Initializing the vision config with default values."
            )

        if text_config is None:
            text_config = _DEFAULTS.text_config.to_dict()
            logger.info(
                "text_config is None. Initializing the text config with default values."
            )

        if vision_config["model_type"] == "siglip_vision_model":
            self.vision_config = SiglipVisionConfig(**vision_config)
        elif vision_config["model_type"] == "siglip2_vision_model":
            self.vision_config = Siglip2VisionConfig(**vision_config)
        elif vision_config["model_type"] == "intern_vit_6b":
            if InternVisionConfig is None:
                raise ImportError(
                    "InternVisionConfig is not available in this environment. "
                    "Install the required dependency or use a supported "
                    "`vision_config.model_type`."
                )
            self.vision_config = InternVisionConfig(**vision_config)
        elif vision_config["model_type"] == "radio":
            if RADIOConfig is None:
                raise ImportError(
                    "RADIOConfig is not available in this environment. "
                    "Install the required dependency or use a supported "
                    "`vision_config.model_type`."
                )
            self.vision_config = RADIOConfig(**vision_config)
        else:
            raise ValueError(
                "Unsupported model_type: {}".format(vision_config["model_type"])
            )

        if text_config["architectures"][0] == "LlamaForCausalLM":
            self.text_config = LlamaConfig(**text_config)
        elif text_config["architectures"][0] == "Phi3ForCausalLM":
            self.text_config = Phi3Config(**text_config)
        elif text_config["architectures"][0] == "Qwen2ForCausalLM":
            self.text_config = Qwen2Config(**text_config)
        elif text_config["architectures"][0] == "Qwen3ForCausalLM":
            self.text_config = Qwen3Config(**text_config)
        else:
            raise ValueError(
                "Unsupported architecture: {}".format(text_config["architectures"][0])
            )
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary.

        Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["select_layer"] = self.select_layer
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["image_token_index"] = self.image_token_index
        output["_attn_implementation"] = self._attn_implementation
        output["_attn_implementation_autoset"] = self._attn_implementation_autoset
        return output
