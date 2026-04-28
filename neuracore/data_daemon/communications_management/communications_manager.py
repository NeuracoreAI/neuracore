"""Backward-compatible import shim for the shared transport manager."""

from __future__ import annotations

from .shared_transport.communications_manager import CommunicationsManager

__all__ = ["CommunicationsManager"]
