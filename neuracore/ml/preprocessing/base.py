"""Base class for preprocessing runtime methods."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from neuracore_types import DataType

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData


class PreprocessingStage(Enum):
    """Where in the pipeline a preprocessing method runs.

    This is a placement decision, not a change to what the pipeline does: a
    configuration serialises and replays identically regardless of the stages
    its methods declare.
    """

    # In the dataset, per sample, inside the DataLoader worker. Required for
    # anything that has to happen before collation -- notably resizing, since
    # recordings differ in resolution and cannot be stacked until they match.
    WORKER = "worker"

    # In the trainer, per batch, after the batch reaches the device. Much
    # cheaper for elementwise work: it runs batched on the GPU instead of
    # per frame on a contended worker CPU.
    DEVICE = "device"


class PreprocessingMethod(ABC):
    """Base interface for preprocessing implementations."""

    # Conservative default: running in the worker is always correct, just
    # potentially slow. Methods opt in to DEVICE.
    stage: ClassVar[PreprocessingStage] = PreprocessingStage.WORKER

    @staticmethod
    @abstractmethod
    def allowed_data_types() -> frozenset[DataType]:
        """Return all data types the method supports."""

    @abstractmethod
    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        """Apply preprocessing and return transformed data."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the preprocessing method to a OmegaConf-style dictionary."""
        target_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        init_signature = inspect.signature(type(self).__init__)
        params = {}
        for param_name in init_signature.parameters:
            if param_name == "self":
                continue
            if param_name == "kwargs" or param_name == "args":
                continue
            params[param_name] = getattr(self, param_name, None)

        return {"_target_": target_name, **params}

    def __str__(self) -> str:
        """Return a human-readable representation of the preprocessing method."""
        params = {k: v for k, v in self.to_dict().items() if k != "_target_"}
        param_str = ", ".join(f"{name}={value!r}" for name, value in params.items())
        return f"{self.__class__.__name__}({param_str})"

    def __repr__(self) -> str:
        """Return a human-readable representation for debugging."""
        return self.__str__()


class PreprocessingConfiguration(dict[DataType, list[PreprocessingMethod]]):
    """Runtime preprocessing pipeline keyed by data type."""

    def for_stage(self, stage: PreprocessingStage) -> PreprocessingConfiguration:
        """Return the subset of this pipeline that runs at ``stage``.

        Order within each data type is preserved. Splitting by stage only
        reorders work across the device boundary when methods of different
        stages are interleaved, so a pipeline that depends on strict ordering
        should keep its methods in one stage.

        Args:
            stage: The stage to select.

        Returns:
            A configuration containing only that stage's methods, omitting data
            types left with nothing to do.
        """
        selected = PreprocessingConfiguration()
        for data_type, methods in self.items():
            staged = [method for method in methods if method.stage is stage]
            if staged:
                selected[data_type] = staged
        return selected

    def __str__(self) -> str:
        """Return a human-readable representation of the preprocessing pipeline."""
        if not self:
            return "PreprocessingConfiguration({})"
        lines = []
        for data_type in sorted(self, key=lambda dt: dt.value):
            methods = self[data_type]
            method_strs = ", ".join(str(method) for method in methods)
            lines.append(f"  {data_type.value}: [{method_strs}]")
        return "PreprocessingConfiguration({\n" + "\n".join(lines) + "\n})"

    def __repr__(self) -> str:
        """Return a human-readable representation for debugging."""
        return self.__str__()
