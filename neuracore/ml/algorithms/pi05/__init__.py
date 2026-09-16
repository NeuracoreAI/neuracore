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
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy `src` over `dst` without ever exposing a partially written file.

    Every patcher writes the same bytes, so renaming into place leaves a
    concurrent reader with either the old file or the new one, never a
    truncated one.
    """
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _patch_transformers() -> None:
    """Copy transformers_replace/ into the installed transformers library.

    Patching is unconditional. A symbol check cannot distinguish a fully
    patched install from one where only some of the replaced modules landed,
    so the files are copied and the affected modules reloaded on every import.

    Raises:
        ValueError: If patching/reloading transformers fails.
    """
    # Must be applied before importing/reloading Gemma modules on Python 3.10.
    _patch_transformers_args_doc()

    try:
        import transformers

        src = Path(__file__).parent / "transformers_replace"
        dst = Path(transformers.__file__).parent
        if src.exists():
            logger.debug("Patching transformers at %s", dst)
            for f in src.rglob("*.py"):
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                _atomic_copy(f, target)
            _reload_transformers_modules()
    except Exception as e:
        raise ValueError(f"Failed to patch/reload transformers: {e}") from e


def _reload_transformers_modules() -> None:
    """Reload patched transformers modules if they were already imported.

    Modules imported before the copy hold the stock source, so they are
    reloaded to pick up the replacements.
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
