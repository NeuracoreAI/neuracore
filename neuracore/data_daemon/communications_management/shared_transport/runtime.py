"""Internal runtime helpers for shared-slot transport."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from neuracore.data_daemon.const import ACK_BASE_DIR


def create_control_socket_path(base_dir: Path = ACK_BASE_DIR) -> Path:
    """Create a unique filesystem path for the shared-slot control socket."""
    base_dir.mkdir(parents=True, exist_ok=True)
    socket_path = base_dir / f"slot_control_{os.getpid()}_{uuid.uuid4().hex}.ipc"
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    return socket_path
