"""Internal runtime helpers for shared-slot transport."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from neuracore.data_daemon.const import ACK_BASE_DIR


def _create_ack_socket_path(*, prefix: str, base_dir: Path = ACK_BASE_DIR) -> Path:
    """Create a unique filesystem path for a transient IPC socket."""
    base_dir.mkdir(parents=True, exist_ok=True)
    socket_path = base_dir / f"{prefix}_{os.getpid()}_{uuid.uuid4().hex}.ipc"
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    return socket_path


def create_control_socket_path(base_dir: Path = ACK_BASE_DIR) -> Path:
    """Create a unique filesystem path for the shared-slot control socket."""
    return _create_ack_socket_path(prefix="slot_control", base_dir=base_dir)


def create_stop_ack_socket_path(base_dir: Path = ACK_BASE_DIR) -> Path:
    """Create a unique filesystem path for recording-stop acknowledgements."""
    return _create_ack_socket_path(prefix="recording_stop_ack", base_dir=base_dir)
