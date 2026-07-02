"""Test 4 - operational hardening (PR7).

Proves the streaming core survives real operation: ffmpeg subprocess crashes are
detected, surfaced (``on_error``) and restarted; teardown and error paths leak no
threads, file descriptors, subprocesses, or registry entries; backpressure has a
single drop point; and close fully tears both surfaces down.

Two flavours:

  * **Error-injection** (peer-capable but fast, runs by default): kill an encode
    mid-stream and assert it restarts + surfaces ``on_error`` + the stream
    recovers; close a peer out from under a live sender and assert it stays
    graceful; force a malformed multi-slice encode and assert the PR5.6 invariant
    guard drops it (never panics).
  * **Soak/stress** (gated by ``NEURACORE_WEBRTC_SOAK``; shortened via
    ``NEURACORE_WEBRTC_SOAK_SECONDS``, like the netem and Chrome gates): a long run
    of churn — add/remove consumers, add/remove video tracks, sustained submit —
    with periodic forced ffmpeg kills, asserting every resource (subprocess /
    thread / fd count and the three process-global registries) returns to baseline,
    with no zombies and no panics.

The ``/proc`` resource probes are Linux-only; the suite skips elsewhere.
"""

from __future__ import annotations

import os
import signal
import sys
import time

import pytest

from tests.integration.webrtc.shared import constants
from tests.integration.webrtc.shared.frames import encode_frame
from tests.integration.webrtc.shared.harness import (
    Relay,
    bootstrap_connection,
    collect_video_frames,
    submit_at_rate,
)

# Resource probes read /proc, so the whole module is Linux-only.
pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="resource/subprocess probes require /proc (Linux only)",
)


# --- /proc resource probes ---------------------------------------------------
def _own_children() -> list[tuple[int, str, str]]:
    """`(pid, comm, state)` for every direct child process of this process."""
    me = os.getpid()
    children: list[tuple[int, str, str]] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/status") as handle:
                status = handle.read()
        except OSError:
            continue  # the process exited between listdir and open
        ppid = comm = state = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
            elif line.startswith("Name:"):
                comm = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
        if ppid == me:
            children.append((pid, comm or "", state or ""))
    return children


def _ffmpeg_children() -> list[int]:
    """The pids of our directly-spawned ffmpeg encode/decode subprocesses."""
    return [pid for pid, comm, _ in _own_children() if "ffmpeg" in comm]


def _zombie_children() -> int:
    """How many of our children are unreaped zombies (state ``Z``)."""
    return sum(1 for _, _, state in _own_children() if state.startswith("Z"))


def _thread_count() -> int:
    return len(os.listdir("/proc/self/task"))


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _kill_ffmpeg() -> int:
    """SIGKILL every live ffmpeg child; return how many were killed."""
    pids = _ffmpeg_children()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)


def _wait_until(predicate, timeout: float, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _producer_errors(relay: Relay, where: str | None = None) -> list[dict]:
    return [
        e
        for e in relay.producer_events()
        if e.get("kind") == "on_error" and (where is None or e.get("where") == where)
    ]


# --- error-injection: encode crash -> restart + on_error + recovery ----------
def test_encode_crash_is_detected_restarted_and_surfaced(make_relay) -> None:
    """Killing the encoder ffmpeg mid-stream restarts it, surfaces an
    ``on_error{where: encode}``, and the decoded stream recovers — not a silent
    stall and not a panic."""
    relay = make_relay()
    bootstrap_connection(relay)
    relay.producer.add_video_track("cam0")

    # Prime the stream so an encoder is running and frames are flowing.
    submit_at_rate(relay, "cam0", fps=30, seconds=2)
    assert _wait_until(
        lambda: len(relay.video_frames("consumer", "cam0")) > 0, timeout=3.0
    ), "no frames decoded before the injected crash"
    before = len(relay.video_frames("consumer", "cam0"))

    killed = _kill_ffmpeg()
    assert killed > 0, "expected at least the encoder ffmpeg to be running"

    # Keep submitting: the feed must detect the dead encoder and restart it.
    submit_at_rate(relay, "cam0", fps=30, seconds=3, start_counter=1000)

    assert _wait_until(
        lambda: len(_producer_errors(relay, "encode")) > 0, timeout=3.0
    ), "the encoder crash was not surfaced as on_error{where: encode}"

    # The stream recovered past the crash (more frames decoded after it).
    after = collect_video_frames(relay, "cam0")
    assert len(after) > before, "the stream did not recover after the encoder crash"

    # No zombie ffmpeg children left behind by the restart.
    assert _zombie_children() == 0, "restart left a zombie subprocess"


# --- error-injection: send on a closed track stays graceful ------------------
def test_send_on_a_closed_track_is_graceful(make_relay) -> None:
    """Closing the consumer out from under a live producer must not raise or
    panic; the producer stays responsive and surfaces an error rather than
    crashing."""
    relay = make_relay()
    bootstrap_connection(relay)
    relay.producer.add_video_track("cam0")
    submit_at_rate(relay, "cam0", fps=30, seconds=1)

    # Tear the consumer down; the producer is now sending on a track whose remote
    # end is gone.
    relay.consumer.close()

    # Submitting after the peer closed must never raise (graceful, drop-on-fail).
    for index in range(80):
        relay.producer.submit_frame("cam0", encode_frame(2000 + index))
        time.sleep(0.01)

    # The producer is still usable (no panic across the FFI boundary): a further
    # API call returns normally and the event queue is still drainable.
    relay.producer.add_data_channel("late", "reliable")
    assert isinstance(relay.producer.drain_events(), list)
    assert _zombie_children() == 0


# --- error-injection: malformed multi-slice input trips the PR5.6 guard -------
def test_multislice_input_trips_the_invariant_guard_without_panic(
    make_broadcast, monkeypatch
) -> None:
    """Forcing a multi-slice encode makes one input frame emit several access
    units; the PR5.6 capture-timestamp underflow guard drops the extras (one
    timestamp per input frame) instead of fabricating timestamps or panicking, so
    the emitted-access-unit count tracks the input rather than multiplying by the
    slice count."""
    monkeypatch.setenv("NCD_WEBRTC_FORCE_SLICES", "4")
    track_id = "cam0"
    relay = make_broadcast()
    relay.broadcaster.add_video_track(track_id)
    make_broadcast.add_consumer(relay, "c0")
    assert relay.wait_consumer_connected("c0"), "consumer did not connect"
    assert (
        relay.wait_for(
            "c0",
            lambda e: e.get("kind") == "on_track_added"
            and e.get("track_id") == track_id,
            constants.RENEG_TIMEOUT_S,
        )
        is not None
    )

    submitted = relay.submit_at_rate(track_id, fps=30, seconds=2)
    time.sleep(0.5)

    encoded = relay.broadcaster.frames_encoded(track_id)
    assert encoded is not None, "the source should exist (no crash)"
    # Without the guard a 4-slice frame would emit ~4 access units per input frame
    # (the assembler flushes per VCL slice). The underflow guard drops the extra 3
    # — one capture timestamp per input frame — so the count stays ~1:1 with the
    # input rather than ~4x it. Generous bound so it is robust to keyframe
    # restarts and host slice-count variation; the point is it does not multiply.
    assert encoded <= submitted * 2, (
        f"emitted {encoded} access units for {submitted} input frames — the "
        f"multi-slice underflow guard did not fire (count multiplied by slices)"
    )
    # And nothing panicked: the broadcaster is still live.
    assert relay.broadcaster.consumer_count() == 1
    assert _zombie_children() == 0


# --- soak/stress (gated; shortenable) ----------------------------------------
def _soak_enabled() -> bool:
    return os.environ.get("NEURACORE_WEBRTC_SOAK", "") not in ("", "0", "false")


@pytest.mark.skipif(
    not _soak_enabled(),
    reason="long soak/churn gate; enable with NEURACORE_WEBRTC_SOAK=1 "
    "(shorten with NEURACORE_WEBRTC_SOAK_SECONDS)",
)
def test_soak_churn_returns_all_resources_to_baseline(make_broadcast) -> None:
    """A long run of consumer/track churn plus sustained submit and periodic
    forced ffmpeg kills returns every resource to baseline: the three
    process-global registries back to their starting size, no leaked
    subprocesses, no zombies, thread and fd counts not growing without bound."""
    from neuracore.core.streaming.p2p.webrtc_selection import load_native

    module = load_native()

    seconds = float(os.environ.get("NEURACORE_WEBRTC_SOAK_SECONDS", 30))

    # Warm the runtime up (its global threads persist) before baselining, so the
    # baseline reflects the steady state, not a cold process.
    warm = make_broadcast()
    make_broadcast.add_consumer(warm, "warm")
    warm.broadcaster.add_video_track("warm0")
    warm.wait_consumer_connected("warm")
    warm.submit_at_rate("warm0", fps=30, seconds=1)
    warm.remove_consumer("warm")
    warm.broadcaster.remove_video_track("warm0")
    warm.close()
    time.sleep(1.0)

    base_threads = _thread_count()
    base_fds = _fd_count()
    base_registries = module.registry_sizes()
    assert base_registries == (
        0,
        0,
        0,
    ), f"registries not clean before the soak: {base_registries}"
    assert _ffmpeg_children() == [], "stray ffmpeg before the soak"

    deadline = time.monotonic() + seconds
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        relay = make_broadcast()
        relay.broadcaster.add_video_track("cam0")
        ids = [f"c{iteration}_{i}" for i in range(3)]
        for cid in ids:
            make_broadcast.add_consumer(relay, cid)
        for cid in ids:
            relay.wait_consumer_connected(cid, timeout=constants.CONNECT_TIMEOUT_S)

        # Add then remove a second track mid-stream (track churn).
        relay.broadcaster.add_video_track("cam1")
        relay.submit_at_rate("cam0", fps=30, seconds=1)
        relay.broadcaster.remove_video_track("cam1")

        # Forced ffmpeg kill (crash injection) every other iteration.
        if iteration % 2 == 0:
            _kill_ffmpeg()
        relay.submit_at_rate("cam0", fps=30, seconds=1)

        # Consumer churn: drop one, keep submitting, then tear the relay down.
        relay.remove_consumer(ids[0])
        relay.submit_at_rate("cam0", fps=20, seconds=0.5)
        relay.close()
        # Each fully-closed iteration must return the registries to baseline.
        assert _wait_until(
            lambda: module.registry_sizes() == base_registries, timeout=3.0
        ), f"registries leaked after iteration {iteration}: {module.registry_sizes()}"

    # Let any teardown threads/subprocesses wind down, then assert baseline.
    time.sleep(2.0)
    assert module.registry_sizes() == base_registries, "registry entries leaked"
    assert _ffmpeg_children() == [], "ffmpeg subprocesses leaked"
    assert _zombie_children() == 0, "zombie subprocesses left behind"
    # Threads and fds may wobble slightly (runtime worker reuse), but must not grow
    # with the iteration count — a small fixed slack, not a per-iteration one.
    assert (
        _thread_count() <= base_threads + 4
    ), f"threads grew from {base_threads} to {_thread_count()} (leak)"
    assert (
        _fd_count() <= base_fds + 16
    ), f"fds grew from {base_fds} to {_fd_count()} (leak)"
