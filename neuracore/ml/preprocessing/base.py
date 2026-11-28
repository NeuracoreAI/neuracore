"""Base class for preprocessing runtime methods."""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from neuracore_types import DataType

if TYPE_CHECKING:
    from neuracore_types import BatchedNCData


def class_name_to_method_name(class_name: str) -> str:
    """Convert a CamelCase class name to its snake_case method name."""
    words_split_after_acronyms = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", class_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words_split_after_acronyms).lower()


class PreprocessingType(str, Enum):
    """Pipeline group a preprocessing method runs in."""

    GEOMETRIC = "geometric"
    PHOTOMETRIC = "photometric"


class PreprocessingMethod(ABC):
    """Base interface for preprocessing implementations."""

    # Key that names this method in a preprocessing config, derived from the
    # class name: ColorJitter becomes color_jitter.
    name: ClassVar[str]

    # Where in the pipeline this method runs.
    #
    # geometric runs it in the dataset, per sample, inside a DataLoader worker.
    #
    # photometric runs it in the trainer, per batch, once the batch has reached
    # the training device.
    # A photometric method must handle a batched input: it sees a whole
    # batch rather than one sample, so anything random has to draw per sample
    # explicitly or the entire batch gets one draw.
    preprocessing_type: ClassVar[PreprocessingType] = PreprocessingType.GEOMETRIC

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set the config name of each subclass from its class name."""
        super().__init_subclass__(**kwargs)
        cls.name = class_name_to_method_name(cls.__name__)

    @staticmethod
    @abstractmethod
    def allowed_data_types() -> frozenset[DataType]:
        """Return all data types the method supports."""

    @abstractmethod
    def __call__(self, data: BatchedNCData) -> BatchedNCData:
        """Apply preprocessing and return transformed data."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the preprocessing method to a one-key name to params mapping."""
        init_signature = inspect.signature(type(self).__init__)
        params = {}
        for param_name in init_signature.parameters:
            if param_name == "self":
                continue
            if param_name == "kwargs" or param_name == "args":
                continue
            params[param_name] = getattr(self, param_name, None)

        return {self.name: params}

    def __str__(self) -> str:
        """Return a human-readable representation of the preprocessing method."""
        params = self.to_dict()[self.name]
        param_str = ", ".join(f"{name}={value!r}" for name, value in params.items())
        return f"{self.__class__.__name__}({param_str})"

    def __repr__(self) -> str:
        """Return a human-readable representation for debugging."""
        return self.__str__()


class PreprocessingConfiguration(dict[DataType, list[PreprocessingMethod]]):
    """Runtime preprocessing pipeline keyed by data type."""

    def split_by_stage(
        self,
    ) -> tuple[PreprocessingConfiguration, PreprocessingConfiguration]:
        """Partition into the geometric and photometric groups.

        Order within each data type is preserved.

        Returns:
            ``(geometric, photometric)``, each omitting data types left with
            nothing to do.
        """
        geometric = PreprocessingConfiguration()
        photometric = PreprocessingConfiguration()
        for data_type, methods in self.items():
            for method in methods:
                target = (
                    geometric
                    if method.preprocessing_type is PreprocessingType.GEOMETRIC
                    else photometric
                )
                target.setdefault(data_type, []).append(method)
        return geometric, photometric

    def to_serializable(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Serialize to the data type to geometric and photometric lists shape."""
        serialized: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for data_type, methods in self.items():
            groups = serialized.setdefault(
                data_type.value,
                {
                    preprocessing_type.value: []
                    for preprocessing_type in PreprocessingType
                },
            )
            for method in methods:
                groups[method.preprocessing_type.value].append(method.to_dict())
        return serialized

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
