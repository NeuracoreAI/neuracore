from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MockConfigManager:
    storage_limit: int | None = None
    bandwidth_limit: int | None = None
    path_to_store_record: str | None = None
    num_threads: int | None = None
    keep_wakelock_while_upload: bool | None = None
    offline: bool | None = None
    api_key: str | None = None
    current_org_id: str | None = None

    def path_to_store_record_from(self, path: Path) -> MockConfigManager:
        return replace(self, path_to_store_record=str(path))

    def resolve_effective_config(self, *args: Any, **kwargs: Any) -> MockConfigManager:
        return self
