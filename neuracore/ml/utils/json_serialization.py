"""Helpers for converting config objects before JSON serialization."""

from collections.abc import Mapping
from enum import Enum
from typing import Protocol, TypeAlias, runtime_checkable

from neuracore_types import CrossEmbodimentDescription, CrossEmbodimentUnion
from omegaconf import DictConfig, ListConfig, OmegaConf

JsonKey: TypeAlias = str | int | float | bool | None
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[JsonKey, "JsonValue"]


@runtime_checkable
class SupportsModelDump(Protocol):
    """Protocol for Pydantic-like objects that can dump JSON-compatible data."""

    def model_dump(self, *, mode: str) -> object:
        """Return a serializable representation of the object."""


def _serialize_cross_embodiment_description(
    cross_embodiment_description: CrossEmbodimentDescription,
) -> dict[str, dict[str, list[str]]]:
    """Convert indexed robot data specs to JSON-serializable ordered name lists."""
    serializable: dict[str, dict[str, list[str]]] = {}
    for robot_id, data_types in cross_embodiment_description.items():
        serializable[robot_id] = {}
        for data_type, indexed_names in data_types.items():
            key = data_type.name if hasattr(data_type, "name") else str(data_type)
            serializable[robot_id][key] = [
                indexed_names[index] for index in sorted(indexed_names)
            ]
    return serializable


def _serialize_cross_embodiment_union(
    cross_embodiment_union: CrossEmbodimentUnion,
) -> dict[str, dict[str, list[str]]]:
    """Convert merged robot data specs to JSON-serializable form."""
    serializable: dict[str, dict[str, list[str]]] = {}
    for robot_id, data_types in cross_embodiment_union.items():
        serializable[robot_id] = {}
        for data_type, names in data_types.items():
            key = data_type.name if hasattr(data_type, "name") else str(data_type)
            serializable[robot_id][key] = list(names)
    return serializable


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
