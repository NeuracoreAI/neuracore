"""Base class for preprocessing runtime methods."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from neuracore_types import BatchedNCData, DataType


class PreprocessingMethod(ABC):
    """Base interface for preprocessing implementations."""

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
