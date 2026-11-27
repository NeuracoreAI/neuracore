"""A network the test can take away and give back.

Every process the test starts (the pytest process, its producer children and
the daemon) reaches the backend through a local switch. For an HTTPS backend
the switch is an HTTP ``CONNECT`` proxy named in the proxy environment
variables. For a plain-HTTP backend, such as one running on this machine, it
sits in front of the backend instead: the run sets ``NEURACORE_API_URL`` to the
switch's address and ``NCD_TEST_BACKEND`` to the backend's ``host:port``, and
the switch forwards each connection there. The SDK reads ``NEURACORE_API_URL``
once, at import, so it must name the switch before the test starts rather than
be changed by it. Offline, the switch
drops each connection as soon as it is accepted, so every request fails with a
connection error, as it would with no internet. Online, it passes them through.
Going offline also closes connections already open, so pooled ones die too.
"""

from __future__ import annotations

import os
import select
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import urlsplit

_API_URL_ENV_VAR = "NEURACORE_API_URL"
_BACKEND_ENV_VAR = "NCD_TEST_BACKEND"
_PROXY_ENV_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
_NO_PROXY_ENV_VARS = ("NO_PROXY", "no_proxy")
_BUFFER_SIZE = 64 * 1024


class NetworkSwitch:
    """A connection switch that is either offline or online.

    With no ``forward_to`` it is an HTTP ``CONNECT`` proxy; with one it passes
    each connection straight to that ``(host, port)``.
    """

    def __init__(
        self, forward_to: tuple[str, int] | None = None, port: int = 0
    ) -> None:
        """Listen on a loopback port, starting offline.

        Args:
            forward_to: The ``(host, port)`` to pass every connection to, or
                None to act as a ``CONNECT`` proxy.
            port: The port to listen on; 0 picks a free one.
        """
        self._forward_to = forward_to
        self._listener = socket.create_server(("127.0.0.1", port))
        self._online = False
        self._lock = threading.Lock()
        self._tunnels: set[socket.socket] = set()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._accept_loop, name="network-switch", daemon=True
        )
        self._thread.start()

    @property
    def url(self) -> str:
        """The switch's own URL: the proxy URL, or the backend's stand-in."""
        host, port = self._listener.getsockname()
        return f"http://{host}:{port}"

    @property
    def online(self) -> bool:
        """Whether connections are currently tunnelled to the backend."""
        return self._online

    def go_online(self) -> None:
        """Tunnel every new connection to the host it asks for."""
        self._online = True

    def go_offline(self) -> None:
        """Drop every new connection and close the tunnels already open."""
        with self._lock:
            self._online = False
            tunnels, self._tunnels = self._tunnels, set()
        for tunnel in tunnels:
            _close(tunnel)

    def close(self) -> None:
        """Stop listening and close every tunnel."""
        self._closed.set()
        self.go_offline()
        _close(self._listener)
        self._thread.join(timeout=5)

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._listener.accept()
            except OSError:
                return
            if not self._online:
                _close(client)
                continue
            threading.Thread(
                target=self._serve, args=(client,), name="network-tunnel", daemon=True
            ).start()

    def _serve(self, client: socket.socket) -> None:
        upstream = None
        try:
            if self._forward_to is not None:
                address = self._forward_to
            else:
                request = _read_request_head(client)
                method, target, _ = request.split(b"\r\n", 1)[0].split(b" ", 2)
                if method.upper() != b"CONNECT" or not self._online:
                    client.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                    return
                host, _, port = target.decode().rpartition(":")
                address = (host, int(port))
            upstream = socket.create_connection(address, timeout=30)
            upstream.settimeout(None)
            with self._lock:
                if not self._online:
                    return
                self._tunnels.update((client, upstream))
            if self._forward_to is None:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _pipe(client, upstream)
        except (OSError, ValueError):
            pass
        finally:
            with self._lock:
                self._tunnels.discard(client)
                if upstream is not None:
                    self._tunnels.discard(upstream)
            _close(client)
            if upstream is not None:
                _close(upstream)


def _read_request_head(client: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = client.recv(_BUFFER_SIZE)
        if not chunk:
            raise ValueError("client closed before sending a request")
        data += chunk
    return data


def _pipe(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    while True:
        readable, _, _ = select.select(sockets, [], [])
        for source in readable:
            chunk = source.recv(_BUFFER_SIZE)
            if not chunk:
                return
            (right if source is left else left).sendall(chunk)


def _close(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


@contextmanager
def switchable_network() -> Iterator[NetworkSwitch]:
    """Route every process started inside the block through a network switch.

    The switch starts offline. With ``NCD_TEST_BACKEND`` set, it listens where
    ``NEURACORE_API_URL`` points and forwards to that backend. Otherwise it is
    a ``CONNECT`` proxy, and the proxy variables are set before anything is
    started, so the daemon and every spawned producer inherit them.

    Yields:
        The switch, offline.

    Raises:
        ValueError: If ``NCD_TEST_BACKEND`` is set but ``NEURACORE_API_URL`` is
            not a ``http://127.0.0.1:<port>`` address for the switch.
    """
    names = _PROXY_ENV_VARS + _NO_PROXY_ENV_VARS
    previous = {name: os.environ.get(name) for name in names}
    backend = os.environ.get(_BACKEND_ENV_VAR)
    if backend:
        api_url = urlsplit(os.environ.get(_API_URL_ENV_VAR, ""))
        if api_url.hostname != "127.0.0.1" or not api_url.port:
            raise ValueError(
                f"{_BACKEND_ENV_VAR} needs {_API_URL_ENV_VAR} to be "
                f"http://127.0.0.1:<port>/..., the switch; got {api_url.geturl()!r}"
            )
        host, _, port = backend.rpartition(":")
        switch = NetworkSwitch(forward_to=(host, int(port)), port=api_url.port)
    else:
        switch = NetworkSwitch()
        for name in _PROXY_ENV_VARS:
            os.environ[name] = switch.url
        for name in _NO_PROXY_ENV_VARS:
            os.environ.pop(name, None)
    try:
        yield switch
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        switch.close()
