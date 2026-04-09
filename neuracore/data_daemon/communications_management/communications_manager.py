"""Utilities for managing communication with the neuracore data daemon."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import zmq

from neuracore.data_daemon.const import (
    BASE_DIR,
    BRIDGE_QUERY_SOCKET_PATH,
    SOCKET_PATH,
)
from neuracore.data_daemon.models import CommandType, MessageEnvelope

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
        self._context = context or zmq.Context.instance()

        self._consumer_socket: zmq.Socket | None = None
        self._query_responder_socket: zmq.Socket | None = None

        self._producer_socket: zmq.Socket | None = None
        self._query_request_socket: zmq.Socket | None = None

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
        if not isinstance(self._query_responder_socket, zmq.Socket):
            self._query_responder_socket = self._context.socket(zmq.REP)

        endpoint = _build_endpoint(SOCKET_PATH)
        query_endpoint = _build_endpoint(BRIDGE_QUERY_SOCKET_PATH)
        stale_socket_paths = [SOCKET_PATH, BRIDGE_QUERY_SOCKET_PATH]

        try:
            self._consumer_socket.bind(endpoint)
            self._query_responder_socket.bind(query_endpoint)
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                removed_any = False
                for socket_path in stale_socket_paths:
                    if not isinstance(socket_path, Path) or not socket_path.exists():
                        continue
                    try:
                        socket_path.unlink()
                        removed_any = True
                        logger.warning("Removed stale daemon socket file at %s", socket_path)
                    except OSError as cleanup_err:
                        logger.warning(
                            "Daemon socket in use and cleanup failed: %s",
                            cleanup_err,
                        )
                        sys.exit(1)
                if not removed_any:
                    logger.warning("Daemon already running! Exiting...")
                    sys.exit(1)
                try:
                    self._consumer_socket.bind(endpoint)
                    self._query_responder_socket.bind(query_endpoint)
                except zmq.error.ZMQError as retry_err:
                    if retry_err.errno == zmq.EADDRINUSE:
                        logger.warning("Daemon already running! Exiting...")
                        sys.exit(1)
                    raise
                return
            raise

        logger.info("Daemon (PULL) bound to %s", endpoint)
        logger.info("Daemon bridge query responder (REP) bound to %s", query_endpoint)

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
        self._producer_socket.setsockopt(zmq.LINGER, 1000)

        endpoint = _build_endpoint(SOCKET_PATH)
        self._producer_socket.connect(endpoint)
        logger.debug(f"Producer connected to {endpoint}")

    def create_query_request_socket(self) -> None:
        """Create a REQ socket for querying daemon cutoff observations."""
        if self._consumer_socket is not None:
            raise RuntimeError((
                "Consumer socket already initialized.",
                "this is either a producer or a daemon process",
            ))
        if isinstance(self._query_request_socket, zmq.Socket):
            return

        self._query_request_socket = self._context.socket(zmq.REQ)
        self._query_request_socket.setsockopt(zmq.LINGER, 0)
        self._query_request_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._query_request_socket.setsockopt(zmq.SNDTIMEO, 1000)
        query_endpoint = _build_endpoint(BRIDGE_QUERY_SOCKET_PATH)
        self._query_request_socket.connect(query_endpoint)

    def receive_query_message(self, timeout_ms: int = 0) -> MessageEnvelope | None:
        """Receive a cutoff query message on the daemon side."""
        if self._query_responder_socket is None:
            return None
        if not self._query_responder_socket.poll(timeout=timeout_ms):
            return None
        return MessageEnvelope.from_bytes(self._query_responder_socket.recv())

    def send_query_response(self, message: MessageEnvelope) -> None:
        """Send a cutoff query response from the daemon side."""
        if self._query_responder_socket is None:
            raise RuntimeError("Query responder socket not initialized")
        self._query_responder_socket.send(message.to_bytes())

    def query_bridge_cutoff_observations(
        self,
        *,
        recording_id: str,
        producer_stop_sequence_numbers: dict[str, int],
    ) -> dict[str, int]:
        """Query which producer cutoff sequences the daemon has observed."""
        if self._query_request_socket is None:
            raise RuntimeError("Query request socket not initialized")

        request = MessageEnvelope(
            producer_id=None,
            command=CommandType.BRIDGE_CUTOFF_QUERY,
            payload={
                "bridge_cutoff_query": {
                    "recording_id": recording_id,
                    "producer_stop_sequence_numbers": producer_stop_sequence_numbers,
                }
            },
        )
        try:
            self._query_request_socket.send(request.to_bytes())
            response = MessageEnvelope.from_bytes(self._query_request_socket.recv())
        except zmq.error.ZMQError:
            self._query_request_socket.close(0)
            self._query_request_socket = None
            self.create_query_request_socket()
            return {}
        payload = response.payload.get("bridge_cutoff_query_response", {})
        observed_producer_sequence_numbers = payload.get(
            "observed_producer_sequence_numbers",
            {},
        )
        if not isinstance(observed_producer_sequence_numbers, dict):
            return {}
        observed: dict[str, int] = {}
        for producer_id, observed_sequence_number in (
            observed_producer_sequence_numbers.items()
        ):
            try:
                observed[str(producer_id)] = int(observed_sequence_number)
            except (TypeError, ValueError):
                continue
        return observed

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
        if self._query_responder_socket is not None:
            self._query_responder_socket.close(0)
            self._query_responder_socket = None

        if isinstance(SOCKET_PATH, Path) and SOCKET_PATH.exists():
            try:
                SOCKET_PATH.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove socket file: {e}")
        if isinstance(BRIDGE_QUERY_SOCKET_PATH, Path) and BRIDGE_QUERY_SOCKET_PATH.exists():
            try:
                BRIDGE_QUERY_SOCKET_PATH.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove bridge query socket file: {e}")

        if self._owns_context:
            self._context.term()

    def cleanup_producer(self) -> None:
        """Cleanup for a producer."""
        if self._producer_socket is not None:
            self._producer_socket.close(0)
            self._producer_socket = None
        if self._query_request_socket is not None:
            self._query_request_socket.close(0)
            self._query_request_socket = None

        if self._owns_context:
            self._context.term()
