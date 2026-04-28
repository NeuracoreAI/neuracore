"""Utilities for managing communication with the neuracore data daemon."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import zmq

from neuracore.data_daemon.const import BASE_DIR, SOCKET_PATH
from neuracore.data_daemon.models import MessageEnvelope

logger = logging.getLogger(__name__)


def _build_endpoint(target: Path | str) -> str:
    """Return a ZMQ endpoint from a path or full endpoint string."""
    if isinstance(target, str) and "://" in target:
        return target
    return f"ipc://{target}"


class CommunicationsManager:
    """Low-level ZeroMQ IPC manager for the data daemon.

    - Daemon uses `start_consumer()` and `receive_message()`.
    - Producers use `create_producer_socket()` and `send_message()`.
    """

    def __init__(self, context: zmq.Context | None = None) -> None:
        """Initialize the CommunicationsManager."""
        self._owns_context = context is None
        # ProducerChannel instances may coexist within one process. Using the
        # global singleton context here lets one channel's cleanup terminate the
        # sockets for every other channel. Give each owned manager its own
        # context unless a shared one was explicitly passed in.
        self._context = context or zmq.Context()

        self._consumer_socket: zmq.Socket | None = None

        self._producer_socket: zmq.Socket | None = None

    def _endpoint(self, socket_path: Path | str) -> str:
        """Build a ZMQ endpoint from a socket path or address."""
        if isinstance(socket_path, Path):
            return f"ipc://{socket_path}"
        if socket_path.startswith(("tcp://", "ipc://", "inproc://")):
            return socket_path
        return f"tcp://{socket_path}"

    def start_consumer(self) -> None:
        """Bind a PULL socket for the daemon.

        Enforces a single daemon by failing if the socket path is already bound.
        """
        if isinstance(BASE_DIR, Path):
            BASE_DIR.mkdir(parents=True, exist_ok=True)
        if isinstance(self._producer_socket, zmq.Socket):
            raise RuntimeError((
                "Producer socket already initialized.",
                "this is either a producer or a daemon process",
            ))
        if not isinstance(self._consumer_socket, zmq.Socket):
            self._consumer_socket = self._context.socket(zmq.PULL)

        endpoint = _build_endpoint(SOCKET_PATH)

        try:
            self._consumer_socket.bind(endpoint)
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                if isinstance(SOCKET_PATH, Path) and SOCKET_PATH.exists():
                    try:
                        SOCKET_PATH.unlink()
                        logger.warning(
                            "Removed stale daemon socket file at %s", SOCKET_PATH
                        )
                    except OSError as cleanup_err:
                        logger.warning(
                            "Daemon socket in use and cleanup failed: %s",
                            cleanup_err,
                        )
                        sys.exit(1)
                    try:
                        self._consumer_socket.bind(endpoint)
                    except zmq.error.ZMQError as retry_err:
                        if retry_err.errno == zmq.EADDRINUSE:
                            logger.warning("Daemon already running! Exiting...")
                            sys.exit(1)
                        raise
                    return
                else:
                    logger.warning("Daemon already running! Exiting...")
                    sys.exit(1)
            raise

        logger.info("Daemon (PULL) bound to %s", endpoint)

    def receive_raw(self) -> bytes | None:
        """Receive a raw message from the consumer socket."""
        if self._consumer_socket is None:
            raise RuntimeError("Consumer socket not initialized")
        if not self._consumer_socket.poll(timeout=10):
            return None
        return self._consumer_socket.recv()

    def create_producer_socket(self) -> None:
        """Create a PUSH socket to send messages to the daemon."""
        if self._consumer_socket is not None:
            raise RuntimeError((
                "Consumer socket already initialized. ",
                "this is either a producer or a daemon process",
            ))
        if isinstance(self._producer_socket, zmq.Socket):
            return

        self._producer_socket = self._context.socket(zmq.PUSH)

        # Do not allow sends until connected
        self._producer_socket.setsockopt(zmq.IMMEDIATE, 1)
        # 1 second timeout for backwards compatibility
        self._producer_socket.setsockopt(zmq.LINGER, 1)

        endpoint = _build_endpoint(SOCKET_PATH)
        self._producer_socket.connect(endpoint)
        logger.debug(f"Producer connected to {endpoint}")

    def send_message(self, message: MessageEnvelope) -> None:
        """Serialize and send a ManagementMessage."""
        if self._producer_socket is None:
            raise RuntimeError(
                "Producer socket not initialized, use create_producer_socket()"
            )

        self._producer_socket.send(message.to_bytes())

    def cleanup_daemon(self) -> None:
        """Cleanup function for the daemon process."""
        if self._consumer_socket is not None:
            self._consumer_socket.close(0)
            self._consumer_socket = None

        if isinstance(SOCKET_PATH, Path) and SOCKET_PATH.exists():
            try:
                SOCKET_PATH.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove socket file: {e}")

        if self._owns_context:
            self._context.term()

    def cleanup_producer(self) -> None:
        """Cleanup for a producer."""
        if self._producer_socket is not None:
            self._producer_socket.close(0)
            self._producer_socket = None

        if self._owns_context:
            self._context.term()
