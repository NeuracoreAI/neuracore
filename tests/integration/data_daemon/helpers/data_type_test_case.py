"""Test case definition for data type integration tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DataTypeTestCase:
    """Test case for a specific data type."""

    name: str
    data_type: str
    log_func: Callable[[float], None]
    timestamp: float = 1234567890.123
    marks: tuple = ()
