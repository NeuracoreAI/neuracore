"""Utility functions."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(reason: str = "") -> Callable[[F], F]:
    """Decorator to mark functions as deprecated using warnings."""

    def decorator(func: F) -> Any:
        @functools.wraps(func)
        def wrapper(*args: tuple, **kwargs: dict[str, Any]) -> Any:
            warnings.warn(
                f"Function {func.__name__} is deprecated. {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
