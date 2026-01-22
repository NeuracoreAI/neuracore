from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MockDaemonConfig:
    """Mock version of DaemonConfig for testing."""

    storage_limit: int | None = None
    bandwidth_limit: int | None = None
    path_to_store_record: str | None = None
    num_threads: int | None = None
    keep_wakelock_while_upload: bool | None = None
    offline: bool | None = None
    api_key: str | None = None
    current_org_id: str | None = None

    @classmethod
    def with_defaults(cls) -> MockDaemonConfig:
        """Return a MockDaemonConfig with sensible defaults for testing."""
        return cls(
            storage_limit=None,
            bandwidth_limit=None,
            path_to_store_record=None,
            num_threads=1,
            keep_wakelock_while_upload=False,
            offline=False,
            api_key=None,
            current_org_id=None,
        )


class MockConfigManager:
    """Mock version of ConfigManager for testing.

    Mirrors real ConfigManager by merging overrides onto defaults.
    """

    def __init__(self, config: MockDaemonConfig | None = None, **kwargs: Any) -> None:
        if config is not None:
            self._overrides = config
        else:
            self._overrides = MockDaemonConfig(**kwargs)

    def resolve_effective_config(self, *args: Any, **kwargs: Any) -> MockDaemonConfig:
        """Merge overrides onto defaults, returning effective config."""
        defaults = MockDaemonConfig.with_defaults()
        merged = {
            field: (
                getattr(self._overrides, field)
                if getattr(self._overrides, field) is not None
                else getattr(defaults, field)
            )
            for field in defaults.__dataclass_fields__
        }
        return MockDaemonConfig(**merged)
