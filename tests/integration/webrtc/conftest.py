"""Fixtures for the in-process WebRTC integration suite.

Selects the Rust WebRTC stack (``NCD_RUST_WEBRTC=1``), loads the native module,
and hands tests a started :class:`Relay` (or a factory for several, for the
multi-consumer case). The factory and the relay are real and never call a
stubbed method, so fixture setup always succeeds against the PR0 stubs — only
the test bodies go red.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest

# The suite exercises the Rust stack; select it before the native module loads.
os.environ.setdefault("NCD_RUST_WEBRTC", "1")

from neuracore.core.streaming.p2p.webrtc_selection import load_native  # noqa: E402
from tests.integration.webrtc.shared.harness import BroadcastRelay, Relay  # noqa: E402
from tests.integration.webrtc.shared.metrics import Metrics, emit  # noqa: E402
from tests.integration.webrtc.shared.server_transport import (  # noqa: E402
    ServerBroadcastRelay,
    ServerRelay,
    signaling_config_from_env,
)


@pytest.fixture(scope="session")
def native() -> object:
    """The compiled ``_native_webrtc`` module, or skip if it is not built."""
    try:
        return load_native()
    except Exception as exc:  # noqa: BLE001 - RuntimeError hint from the loader
        pytest.skip(f"native webrtc module unavailable: {exc}")


RelayFactory = Callable[..., Relay]


@pytest.fixture
def make_relay(native: object) -> Iterator[RelayFactory]:
    """Factory creating started producer<->consumer relays, closed on teardown."""
    relays: list[Relay] = []

    # When the operator sets the signaling env, the same native peers connect
    # through the real backend; otherwise the in-process relay is used unchanged.
    config = signaling_config_from_env()

    def _make(*, frame_queue_capacity: int = 16, name: str = "relay") -> Relay:
        producer = native.Producer(
            connection_id=None, frame_queue_capacity=frame_queue_capacity
        )
        consumer = native.Consumer(connection_id=None)
        if config is not None:
            relay: Relay = ServerRelay(
                producer, consumer, config=config, name=name
            ).start()
        else:
            relay = Relay(producer, consumer, name=name).start()
        relays.append(relay)
        return relay

    yield _make

    for relay in relays:
        relay.close()


@pytest.fixture
def relay(make_relay: RelayFactory) -> Relay:
    """A single started producer<->consumer relay."""
    return make_relay()


BroadcastFactory = Callable[..., BroadcastRelay]


@pytest.fixture
def make_broadcast(native: object) -> Iterator[BroadcastFactory]:
    """Factory creating started one-broadcaster<->many-consumers relays.

    The returned relay owns one ``Broadcaster``; ``add_consumer(id)`` builds an
    answer-only ``Consumer`` peer and wires its signaling. Closed on teardown.
    """
    relays: list[BroadcastRelay] = []
    config = signaling_config_from_env()

    def _make(
        *, frame_queue_capacity: int = 16, name: str = "broadcast"
    ) -> BroadcastRelay:
        broadcaster = native.Broadcaster(
            connection_id=None, frame_queue_capacity=frame_queue_capacity
        )
        if config is not None:
            relay: BroadcastRelay = ServerBroadcastRelay(
                broadcaster, config=config, name=name
            ).start()
        else:
            relay = BroadcastRelay(broadcaster, name=name).start()
        relays.append(relay)
        return relay

    def _add_consumer(relay: BroadcastRelay, consumer_id: str) -> object:
        consumer = native.Consumer(connection_id=consumer_id)
        relay.add_consumer(consumer_id, consumer)
        return consumer

    # Expose the consumer constructor so tests can join consumers without reaching
    # for the native module directly.
    _make.add_consumer = _add_consumer  # type: ignore[attr-defined]

    yield _make

    for relay in relays:
        relay.close()


@pytest.fixture(scope="session")
def perf_metrics() -> Iterator[Metrics]:
    """Shared structured perf output, emitted as JSON at session teardown."""
    metrics = Metrics()
    yield metrics
    emit(metrics)
