"""PI0 algorithm with transformers patching."""

import shutil
from pathlib import Path


def _patch_transformers_args_doc() -> None:
    """Patch transformers args_doc for Python 3.10 UnionType annotations."""
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
    """Auto-patch transformers on first import."""
    # Check if already patched using local check module
    from .transformers_replace.models.siglip import check

    if check.check_whether_transformers_replace_is_installed_correctly():
        return  # Already patched
    else:
        print("Transformers not patched, trying to patch now")

    try:
        import transformers

        src = Path(__file__).parent / "transformers_replace"
        dst = Path(transformers.__file__).parent
        if src.exists():
            for f in src.rglob("*.py"):
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    except Exception:
        raise ValueError("Failed to patch transformers because of permission issues")


_patch_transformers()
_patch_transformers_args_doc()
