"""ML package initialisation.

Kept lightweight to avoid importing heavy optional dependencies (e.g., torch)
unless the ML symbols are actually accessed.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "NeuracoreModel",
    "BatchedInferenceInputs",
    "BatchedTrainingSamples",
    "BatchedTrainingOutputs",
]

if TYPE_CHECKING:
    from .core.ml_types import (  # pragma: no cover
        BatchedInferenceInputs,
        BatchedTrainingOutputs,
        BatchedTrainingSamples,
    )
    from .core.neuracore_model import NeuracoreModel  # pragma: no cover


def __getattr__(name: str) -> Any:
    """Lazily import ML symbols to avoid eager heavy dependencies."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if name == "NeuracoreModel":
        module = import_module(".core.neuracore_model", __name__)
    else:
        module = import_module(".core.ml_types", __name__)

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
