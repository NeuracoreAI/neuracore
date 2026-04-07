"""Minimal preprocessing runtime for batched neuracore data."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from neuracore_types import BatchedNCData, DataType
from neuracore_types.preprocessing import MethodSpec, PreProcessingMethod

from .methods import resize_pad

# Single source of truth for available runtime methods and compatibility.
_METHOD_REGISTRY: dict[str, MethodSpec] = {
    "resize_pad": MethodSpec(
        handler=resize_pad,
        allowed_data_types={DataType.RGB_IMAGES, DataType.DEPTH_IMAGES},
    ),
}


def _resolve_custom_callable(custom_callable: str) -> Callable[..., BatchedNCData]:
    """Resolve a dotted callable path like 'pkg.module.function'."""
    if "." not in custom_callable:
        raise ValueError(
            "custom_callable must be a dotted path in the form"
            "'module_path.function_name'."
        )
    module_path, function_name = custom_callable.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to import custom callable module '{module_path}'."
        ) from exc
    handler = getattr(module, function_name, None)
    if handler is None or not callable(handler):
        raise ValueError(
            f"custom_callable '{custom_callable}' is not a callable object."
        )
    return handler


def apply_methods_for_slot(
    data_type: DataType,
    batched_data: BatchedNCData,
    methods: list[PreProcessingMethod],
) -> BatchedNCData:
    """Apply configured preprocessing methods for one data-type/slot item."""
    for method in methods:
        method_key = method.name
        if method_key.startswith("custom"):
            if method.custom_callable is None:
                raise ValueError(
                    f"Preprocessing method '{method_key}' requires `custom_callable`."
                )
            handler = _resolve_custom_callable(method.custom_callable)
            batched_data = handler(batched_data, **method.args)
            continue
        method_spec = _METHOD_REGISTRY.get(method_key)
        if method_spec is None:
            raise ValueError(
                f"Unsupported preprocessing method '{method_key}'. "
                f"Expected a built-in registry key or a custom "
                "name starting with 'custom' plus `custom_callable`."
            )
        allowed_types = method_spec.allowed_data_types
        if data_type not in allowed_types:
            allowed_list = ", ".join(sorted(dt.value for dt in allowed_types))
            raise ValueError(
                f"Preprocessing method '{method_key}' is not allowed for data type "
                f"{data_type.value}. Allowed data types: [{allowed_list}]"
            )
        handler = method_spec.handler
        batched_data = handler(batched_data, **method.args)
    return batched_data
