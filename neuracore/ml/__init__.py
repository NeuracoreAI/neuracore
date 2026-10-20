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
    "RTCConfig",
    "TemporalEnsembleConfig",
]

# Which submodule each lazily-exported name comes from.
_SOURCES = {
    "NeuracoreModel": ".core.neuracore_model",
    "BatchedInferenceInputs": ".core.ml_types",
    "BatchedTrainingSamples": ".core.ml_types",
    "BatchedTrainingOutputs": ".core.ml_types",
    "RTCConfig": ".utils.real_time_chunking",
    "TemporalEnsembleConfig": ".utils.temporal_ensemble",
}

if TYPE_CHECKING:
    from .core.ml_types import (  # pragma: no cover
        BatchedInferenceInputs,
        BatchedTrainingOutputs,
        BatchedTrainingSamples,
    )
    from .core.neuracore_model import NeuracoreModel  # pragma: no cover
    from .utils.real_time_chunking import RTCConfig  # pragma: no cover
    from .utils.temporal_ensemble import TemporalEnsembleConfig  # pragma: no cover


def __getattr__(name: str) -> Any:
    """Lazily import ML symbols to avoid eager heavy dependencies."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(_SOURCES[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
