"""Internal shared-slot transport runtime components."""

from __future__ import annotations

from .communications_manager import CommunicationsManager
from .registry import SharedSlotRegistry, SharedSlotTimeout

__all__ = [
    "SharedSlotRegistry",
    "SharedSlotTimeout",
    "CommunicationsManager",
    "SharedSlotVideoTransport",
]


def __getattr__(name: str) -> object:
    """Lazily expose heavier transport types to avoid package import cycles."""
    if name == "SharedSlotVideoTransport":
        from .shared_slot_transport import SharedSlotVideoTransport

        return SharedSlotVideoTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
