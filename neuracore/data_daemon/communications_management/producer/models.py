"""Producer-side transport message models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from neuracore.data_daemon.models import MessageEnvelope


@dataclass
class QueuedEnvelope:
    """A socket message plus optional failure callback."""

    envelope: MessageEnvelope
    on_sent: Callable[[], None] | None = None
    on_failed_send: Callable[[], None] | None = None
