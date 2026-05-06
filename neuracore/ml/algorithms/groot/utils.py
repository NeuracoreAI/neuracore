"""Utility helpers for the GR00T N1.6 algorithm.

This module groups together the environment plumbing used by
``modules.py`` so the architecture file can stay focused on
``nn.Module`` definitions:

  - A monkey-patch for a ``transformers`` 4.53.x docstring crash on
    Python 3.10+ union types.
  - Hardware-specific attention backend selection (Spark sm12.1
    workaround and ``flash_attn`` detection).
  - Pretrained checkpoint loading from local paths or HuggingFace Hub,
    including sharded safetensors support.
  - Eagle3-VL config construction from the bundled Python dataclass.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .eagle_config.eagle3_vl_dataconfig import Eagle3VLDataConfig

logger = logging.getLogger(__name__)


def _patch_transformers_union_type_bug() -> None:
    """Patch transformers 4.53.x auto_docstring crash on Python 3.10+ union types.

    ``transformers.utils.args_doc._process_parameter_type`` accesses
    ``param.annotation.__name__`` which does not exist on ``types.UnionType``
    (the ``X | Y`` syntax).  This monkey-patches the function so that union
    type annotations are converted to their string representation first.
    The patch is idempotent.

    Can be removed once transformers is upgraded to >=5.0.0.
    """
    import types as _types

    from transformers.utils import args_doc as _args_doc

    if getattr(_args_doc, "_union_type_patched", False):
        return
    _orig = _args_doc._process_parameter_type

    def _safe_process_parameter_type(param: Any, param_name: str, func: Any) -> Any:
        if isinstance(param.annotation, _types.UnionType):
            param = param.replace(annotation=str(param.annotation))
        return _orig(param, param_name, func)

    _args_doc._process_parameter_type = _safe_process_parameter_type
    _args_doc._union_type_patched = True


_patch_transformers_union_type_bug()


### Hardware-specific attention backend selection
def _is_spark_sm121() -> bool:
    """Detect NVIDIA Spark (compute capability 12.1) GPUs."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (12, 1)


def _should_force_math_sdpa() -> bool:
    """Return True if we must use the math SDPA backend.

    Spark sm12.1 hits a broken PyTorch mem-efficient SDPA kernel dispatch.
    Override via env var GR00T_DIT_SDPA_MODE=math|default.
    """
    override = os.environ.get("GR00T_DIT_SDPA_MODE")
    if override == "math":
        return True
    if override == "default":
        return False
    return _is_spark_sm121()


def get_attn_implementation() -> str:
    """Return the best available attention implementation.

    Prefers ``flash_attention_2`` when the ``flash_attn`` package is installed,
    otherwise falls back to PyTorch's built-in ``sdpa``.
    """
    try:
        import flash_attn  # noqa: F401

        logger.info("flash_attn detected – using flash_attention_2")
        return "flash_attention_2"
    except ImportError:
        logger.info("flash_attn not found – falling back to sdpa")
        return "sdpa"


def sdpa_context() -> AbstractContextManager[None]:
    """Return a context manager that selects the correct SDPA backend.

    On most hardware this is a no-op. On Spark sm12.1 it forces the safe
    math backend to avoid numerical noise from the mem-efficient kernel.
    """
    if not _should_force_math_sdpa():
        return nullcontext()
    return sdpa_kernel(SDPBackend.MATH)


### Pretrained weight loading utilities
def load_pretrained_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Download and load a GR00T N1.6 state dict from a checkpoint.

    Supports both local paths and HuggingFace Hub model identifiers.
    Handles both single-file and sharded safetensors checkpoints.

    Args:
        model_path: Local directory path or HuggingFace model ID
            (e.g., "nvidia/GR00T-N1.6-3B").

    Returns:
        Full state dict mapping parameter names to tensors.

    Raises:
        RuntimeError: If the checkpoint cannot be loaded from the given path.
    """
    from safetensors.torch import load_file

    # --- Try local single-file checkpoint ---
    local_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(local_file):
        logger.info("Loading checkpoint from local path: %s", local_file)
        return load_file(local_file)

    # --- Try local sharded checkpoint ---
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info("Loading sharded checkpoint from: %s", model_path)
        return load_sharded_safetensors(model_path, index_file)

    # --- Fall back to HuggingFace Hub download ---
    try:
        from huggingface_hub import hf_hub_download

        # Try single-file first
        try:
            weight_path = hf_hub_download(model_path, "model.safetensors")
            logger.info("Downloaded single-file checkpoint from HF Hub.")
            return load_file(weight_path)
        except Exception:
            pass

        # Try sharded checkpoint
        index_path = hf_hub_download(model_path, "model.safetensors.index.json")
        logger.info("Downloaded sharded checkpoint index from HF Hub.")
        return load_sharded_safetensors_from_hub(model_path, index_path)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from '{model_path}'. "
            f"Ensure the path exists locally or is a valid HuggingFace model ID. "
            f"Error: {e}"
        ) from e


def load_sharded_safetensors(
    model_dir: str, index_file: str
) -> dict[str, torch.Tensor]:
    """Load a sharded safetensors checkpoint from a local directory.

    Args:
        model_dir: Directory containing the shard files.
        index_file: Path to the model.safetensors.index.json file.

    Returns:
        Combined state dict from all shards.
    """
    from safetensors.torch import load_file

    with open(index_file) as f:
        index = json.load(f)

    # Collect unique shard filenames
    shard_files = set(index["weight_map"].values())
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(shard_files):
        shard_path = os.path.join(model_dir, shard_file)
        state_dict.update(load_file(shard_path))
    return state_dict


def load_sharded_safetensors_from_hub(
    model_path: str, index_path: str
) -> dict[str, torch.Tensor]:
    """Load a sharded safetensors checkpoint from HuggingFace Hub.

    Args:
        model_path: HuggingFace model identifier.
        index_path: Local path to the downloaded index JSON file.

    Returns:
        Combined state dict from all downloaded shards.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    with open(index_path) as f:
        index = json.load(f)

    shard_files = set(index["weight_map"].values())
    state_dict: dict[str, torch.Tensor] = {}
    for shard_file in sorted(shard_files):
        local_path = hf_hub_download(model_path, shard_file)
        state_dict.update(load_file(local_path))
    return state_dict


def load_config_json(model_path: str) -> dict:
    """Load config.json from a GR00T N1.6 checkpoint.

    Args:
        model_path: Local directory path or HuggingFace model ID.

    Returns:
        Parsed config dictionary. Returns empty dict if not found.
    """
    # Try local first
    local_file = os.path.join(model_path, "config.json")
    if os.path.exists(local_file):
        with open(local_file) as f:
            return json.load(f)

    # Try HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, "config.json")
        with open(config_path) as f:
            return json.load(f)
    except Exception:
        logger.warning("Could not load config.json from %s", model_path)
        return {}


def build_eagle_config_from_dataclass(
    dataconfig: Eagle3VLDataConfig | None = None,
) -> Any:
    """Build an ``Eagle3_VLConfig`` from a dataclass (no JSON on disk needed).

    Args:
        dataconfig: Dataclass holding Eagle3-VL configuration values.
            If *None*, the default ``Eagle3VLDataConfig()`` is used.

    Returns:
        An ``Eagle3_VLConfig`` instance ready for ``AutoModel.from_config``.
    """
    from .eagle_config.configuration_eagle3_vl import Eagle3_VLConfig

    if dataconfig is None:
        dataconfig = Eagle3VLDataConfig()

    cfg_dict = dataconfig.to_dict()
    # Remove fields that Eagle3_VLConfig doesn't accept directly
    for key in (
        "auto_map",
        "architectures",
        "model_type",
        "dynamic_image_size",
        "max_dynamic_tiles",
        "min_dynamic_tiles",
        "mlp_connector_layers",
        "use_pixel_shuffle",
        "use_thumbnail",
        "transformers_version",
    ):
        cfg_dict.pop(key, None)
    return Eagle3_VLConfig(**cfg_dict)
