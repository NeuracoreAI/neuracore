"""PI0 algorithm with transformers patching.

Automatically patches the installed transformers library with custom modifications
required by PI0. This eliminates the need to manually copy files into the transformers
installation directory.

The patching includes:
- Gemma model with Adaptive RMSNorm support
- Gated residual connections for Gemma modeling
- Custom PaliGemma and SigLIP modifications
- Python 3.10 UnionType annotation support for transformers docs
"""

import importlib

# cspell:ignore adarms
import inspect
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_whether_transformers_replace_is_installed_correctly() -> bool:
    """Check whether transformers has been patched with PI0 modifications.

    Verifies that the installed `transformers` library has been patched by checking
    runtime symbols and call signatures that are added by PI0.

    Returns:
        True if patches are detected, False otherwise.
    """
    try:
        from transformers.models.gemma import modeling_gemma
        from transformers.models.gemma.configuration_gemma import GemmaConfig
        from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer

        cfg_init_params = inspect.signature(GemmaConfig.__init__).parameters
        if "use_adarms" not in cfg_init_params:
            return False
        if "adarms_cond_dim" not in cfg_init_params:
            return False

        cfg = GemmaConfig(use_adarms=True)
        if not getattr(cfg, "use_adarms", False):
            return False
        if getattr(cfg, "adarms_cond_dim", None) is None:
            return False

        if not callable(getattr(modeling_gemma, "_gated_residual", None)):
            return False

        decoder_forward_params = inspect.signature(GemmaDecoderLayer.forward).parameters
        if "adarms_cond" not in decoder_forward_params:
            return False
        return True
    except Exception:
        return False


def _patch_transformers_args_doc() -> None:
    """Patch transformers args_doc to handle Python 3.10 UnionType annotations.

    Fixes documentation generation errors caused by UnionType syntax
    (e.g., `int | str`). The patch is applied once and marked to prevent
    re-patching.
    """
    try:
        import inspect
        import re
        import types
        from collections.abc import Callable
        from typing import Any, get_args

        from transformers.utils import args_doc

        if getattr(args_doc, "_UNIONTYPE_PATCHED", False):
            return

        original = args_doc._process_parameter_type

        def _process_parameter_type(
            param: inspect.Parameter, param_name: str, func: Callable[..., Any]
        ) -> tuple[str, bool]:
            if param.annotation != inspect.Parameter.empty and isinstance(
                param.annotation, types.UnionType
            ):
                param_type = str(param.annotation).replace("transformers.", "~")
                optional = any(arg is type(None) for arg in get_args(param.annotation))
                if "ForwardRef" in param_type:
                    param_type = re.sub(r"ForwardRef\('([\w.]+)'\)", r"\1", param_type)
                if "Optional" in param_type:
                    param_type = re.sub(r"Optional\[(.*?)\]", r"\1", param_type)
                    optional = True
                return param_type, optional
            return original(param, param_name, func)

        args_doc._process_parameter_type = _process_parameter_type
        args_doc._UNIONTYPE_PATCHED = True
    except Exception:
        return


def _patch_transformers() -> None:
    """Automatically patch transformers with custom modifications.

    Checks if patching is needed, then copies files from transformers_replace/
    to the installed transformers library. The process is idempotent and works
    across different installation methods.

    Raises:
        ValueError: If patching fails due to permission issues.
    """
    if check_whether_transformers_replace_is_installed_correctly():
        _reload_transformers_modules()
        return  # Already patched
    else:
        logger.info("Transformers not patched; attempting to patch now.")

    try:
        import transformers

        src = Path(__file__).parent / "transformers_replace"
        dst = Path(transformers.__file__).parent
        if src.exists():
            for f in src.rglob("*.py"):
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
            _reload_transformers_modules()
    except Exception:
        raise ValueError("Failed to patch transformers because of permission issues")


def _reload_transformers_modules() -> None:
    """Reload patched transformers modules if they were already imported.

    `check_whether_transformers_replace_is_installed_correctly()` imports Gemma
    modules before patching. If patching is needed, those stale module objects
    remain in `sys.modules` and hide copied updates unless reloaded.
    """
    importlib.invalidate_caches()
    module_names = [
        "transformers.models.gemma.configuration_gemma",
        "transformers.models.gemma.modeling_gemma",
        "transformers.models.paligemma.modeling_paligemma",
        "transformers.models.siglip.modeling_siglip",
    ]
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None:
            importlib.reload(module)


_patch_transformers()
_patch_transformers_args_doc()
