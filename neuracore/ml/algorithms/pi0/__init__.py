"""PI0 algorithm with transformers patching."""

import shutil
from pathlib import Path


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

        breakpoint()

        if src.exists():
            for f in src.rglob("*.py"):
                target = dst / f.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
    except Exception:
        raise ValueError("Failed to patch transformers because of permission issues")


_patch_transformers()
