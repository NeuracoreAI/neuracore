"""Utilities for managing communication with the neuracore data daemon."""

from __future__ import annotations

import logging
import sys

import zmq

from neuracore.data_daemon.const import (
    BASE_DIR,
    RECORDING_EVENTS_SOCKET_PATH,
    SOCKET_PATH,
)
from neuracore.data_daemon.models import MessageEnvelope

logger = logging.getLogger(__name__)


class CommunicationsManager:
    """Low-level ZeroMQ IPC manager for the data daemon.

    - Daemon uses `start_consumer()` and `receive_message()`.
    - Producers use `create_producer_socket()` and `send_message()`.
    """

    def __init__(self) -> None:
        """Initialize the CommunicationsManager."""
        self.context = zmq.Context()
        self.consumer_socket: zmq.Socket | None = None
        self.publisher_socket: zmq.Socket | None = None

    def start_consumer(self) -> None:
        """Bind a PULL socket for the daemon.

        Enforces a single daemon by failing if the socket path is already bound.
        """
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.consumer_socket = self.context.socket(zmq.PULL)

        endpoint = f"ipc://{SOCKET_PATH}"

        try:
            self.consumer_socket.bind(endpoint)
        except zmq.error.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                logger.warning("Daemon already running! Exiting...")
                sys.exit(1)
            raise

        logger.info(f"Daemon (PULL) bound to {endpoint}")

    def receive_message(self) -> MessageEnvelope:
        """Receive and deserialize a ManagementMessage."""
        if self.consumer_socket is None:
            raise RuntimeError("Consumer socket not initialized")
        msg_bytes = self.consumer_socket.recv()
        return MessageEnvelope.from_bytes(msg_bytes)

    def start_publisher(self) -> None:
        """Bind a PUB socket for recording control messages."""
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.publisher_socket = self.context.socket(zmq.PUB)
        endpoint = f"ipc://{RECORDING_EVENTS_SOCKET_PATH}"
        self.publisher_socket.bind(endpoint)
        logger.info("Daemon (PUB) bound to %s", endpoint)

    def create_subscriber_socket(self) -> zmq.Socket | None:
        """Create a SUB socket to receive daemon control messages."""
        if not RECORDING_EVENTS_SOCKET_PATH.exists():
            logger.warning("Recording events socket does not exist. Cannot connect.")
            return None
        socket = self.context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        endpoint = f"ipc://{RECORDING_EVENTS_SOCKET_PATH}"
        socket.connect(endpoint)
        logger.debug("Producer subscribed to %s", endpoint)
        return socket

    def create_producer_socket(self) -> zmq.Socket | None:
        """Create a PUSH socket to send messages to the daemon.

        Returns:
            The socket if connection is possible, otherwise None.
        """
        if not SOCKET_PATH.exists():
            logger.warning("Daemon socket does not exist. Cannot connect.")
            return None

        socket = self.context.socket(zmq.PUSH)
        endpoint = f"ipc://{SOCKET_PATH}"
        socket.connect(endpoint)
        logger.debug(f"Producer connected to {endpoint}")
        return socket

    def send_message(self, socket: zmq.Socket | None, message: MessageEnvelope) -> None:
        """Serialize and send a ManagementMessage."""
        if socket is None:
            logger.warning("No consumer available to send message")
            return
        socket.send(message.to_bytes())

    def publish_message(self, message: MessageEnvelope) -> None:
        """Publish a control message to all subscribers."""
        if self.publisher_socket is None:
            logger.warning("No publisher available to send message")
            return
        self.publisher_socket.send(message.to_bytes())

    def cleanup_daemon(self) -> None:
        """Cleanup function for the daemon process.

        Closes the consumer socket, removes the IPC socket file,
        and terminates the ZMQ context.
        """
        if self.consumer_socket is not None:
            self.consumer_socket.close(0)
            self.consumer_socket = None
        if self.publisher_socket is not None:
            self.publisher_socket.close(0)
            self.publisher_socket = None

        if SOCKET_PATH.exists():
            try:
                SOCKET_PATH.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove socket file: {e}")
        if RECORDING_EVENTS_SOCKET_PATH.exists():
            try:
                RECORDING_EVENTS_SOCKET_PATH.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove recording events socket: {e}")

        self.context.term()

    def cleanup_producer(self) -> None:
        """Cleanup for a producer.

        Producers should NOT delete the socket file; they only close their context.
        """
        self.context.term()
