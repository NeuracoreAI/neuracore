from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zmq

from neuracore.data_daemon.communications_management import (
    communications_manager as comms_module,
)
from neuracore.data_daemon.communications_management.communications_manager import (
    CommunicationsManager,
)


class FakeContext:
    def __init__(self, socket_obj: MagicMock) -> None:
        self._socket_obj = socket_obj

    def socket(self, _socket_type):
        return self._socket_obj


def test_cleanup_daemon_removes_socket_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "ndd"
    socket_path = base_dir / "management.sock"
    events_path = base_dir / "recording_events.sock"
    base_dir.mkdir(parents=True)
    socket_path.write_text("stale")
    events_path.write_text("stale")

    mpsa = monkeypatch.setattr

    mpsa(comms_module, "BASE_DIR", base_dir)
    mpsa(comms_module, "SOCKET_PATH", socket_path)
    mpsa(comms_module, "RECORDING_EVENTS_SOCKET_PATH", events_path)

    comm = CommunicationsManager(context=MagicMock())
    consumer = MagicMock()
    publisher = MagicMock()
    comm.consumer_socket = consumer
    comm.publisher_socket = publisher

    comm.cleanup_daemon()

    assert not socket_path.exists()
    assert not events_path.exists()
    consumer.close.assert_called_once_with(0)
    publisher.close.assert_called_once_with(0)


def test_start_consumer_removes_stale_socket_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_dir = tmp_path / "ndd"
    socket_path = base_dir / "management.sock"
    base_dir.mkdir(parents=True)
    socket_path.write_text("stale")

    monkeypatch.setattr(comms_module, "BASE_DIR", base_dir)
    monkeypatch.setattr(comms_module, "SOCKET_PATH", socket_path)

    mock_socket = MagicMock()
    mock_socket.bind.side_effect = [
        zmq.error.ZMQError(zmq.EADDRINUSE),
        None,
    ]

    comm = CommunicationsManager(context=FakeContext(mock_socket))
    comm.start_consumer()

    assert mock_socket.bind.call_count == 2
    assert not socket_path.exists()
