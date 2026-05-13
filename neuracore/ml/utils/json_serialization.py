"""Helpers for converting config objects before JSON serialization."""

from collections.abc import Mapping
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

from omegaconf import DictConfig, ListConfig, OmegaConf

JsonKey: TypeAlias = str | int | float | bool | None
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[JsonKey, "JsonValue"]


@runtime_checkable
class SupportsModelDump(Protocol):
    """Protocol for Pydantic-like objects that can dump JSON-compatible data."""

    def model_dump(self, *, mode: str) -> object:
        """Return a serializable representation of the object."""


def _to_json_key(key: object) -> JsonKey:
    if isinstance(key, (str, int, float, bool)) or key is None:
        return key
    if isinstance(key, Enum):
        return _to_json_key(key.value)
    return str(key)


def to_json_serializable(value: object) -> JsonValue:
    """Convert OmegaConf and Pydantic-style objects into JSON-safe containers."""
    if isinstance(value, SupportsModelDump):
        return to_json_serializable(value.model_dump(mode="json"))

    if isinstance(value, (DictConfig, ListConfig)):
        return to_json_serializable(OmegaConf.to_container(value, resolve=True))

    if isinstance(value, Mapping):
        return {
            _to_json_key(key): to_json_serializable(item) for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_json_serializable(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Enum):
        return to_json_serializable(value.value)

    return str(value)
