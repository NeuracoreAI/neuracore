"""Constrained-link harness body, run inside a private network namespace.

`test_perf_under_constrained_link` cannot shape the loopback the in-process peers
use: the container is `network_mode: host`, so its `lo` is the host's, and a
long-lived global tokio runtime cannot be moved into a namespace after its
sockets exist. So the test re-execs *this* script under ``unshare -n`` (a fresh
net namespace), and here — already inside that namespace, before any peer is
built — we bring up the namespace's private `lo` and apply a real ``tc netem``
profile to it. Everything the peers do then traverses the shaped loopback.

The script runs one producer->consumer relay at the source fps for a fixed
window, then prints a single JSON line of results to stdout for the parent test
to assert against:

    {"delivered_fps", "corrupted", "closed", "max_step", "sent", "ok"}

Env in:
  * ``NEURACORE_WEBRTC_NETEM``   tc netem args (default: a profile that bites)
  * ``NEURACORE_WEBRTC_NETEM_SECONDS`` window length (default 12)
  * ``NCD_WEBRTC_DISABLE_ADAPT`` if set, the producer pins the finest rung, so
    the same constraint should *fail* the floor — the proof the test bites.

Requires CAP_NET_ADMIN (tc) and that it is already in a private netns (the
parent supplies CAP_SYS_ADMIN via ``unshare -n``).
"""

# cspell: ignore WEBRTC

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# Default netem profile: a delay plus a rate low enough that the full-resolution
# top rung overflows the qdisc (drops -> RR loss -> the producer's estimator
# degrades), but a downscaled coarser rung fits. Tuned so the test is red without
# adaptation and green with it.
#
# The rate sits in the wide window between the full-resolution top rung
# (~2.5 Mbit, maxrate-capped) and the half-resolution bottom rung (~0.3 Mbit).
# That separation only exists because the synthetic frames carry high-frequency
# detail (see shared/frames.py): a smooth gradient compresses to almost nothing at
# every resolution, leaving the downscale ladder with no rate lever — the earlier
# `rate 400kbit limit 48` profile only bit because a since-fixed producer bug
# emitted ~7 RTP slices per frame, inflating packet/header rate ~7x (see
# reports/SPIKE-chrome-pframe.md). With one slice per frame and high-frequency
# content the gate is driven by genuine bitrate, not a packetization artefact.
DEFAULT_NETEM = "delay 20ms rate 800kbit limit 64"


def _sh(cmd: str) -> tuple[int, str]:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)  # noqa: S602
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _setup_link() -> str:
    """Bring up the namespace's private loopback and shape it with netem."""
    netem = os.environ.get("NEURACORE_WEBRTC_NETEM", DEFAULT_NETEM)
    rc, out = _sh("ip link set lo up")
    if rc != 0:
        raise RuntimeError(f"could not bring up lo in the netns: {out}")
    # Clear any inherited qdisc, then apply the profile to loopback.
    _sh("tc qdisc del dev lo root")
    rc, out = _sh(f"tc qdisc add dev lo root netem {netem}")
    if rc != 0:
        raise RuntimeError(f"could not apply netem '{netem}': {out}")
    return netem


def _run_relay(seconds: float) -> dict:
    # Imported here so the import cost is paid inside the namespace, after the
    # link is shaped.
    os.environ.setdefault("NCD_RUST_WEBRTC", "1")
    from neuracore.core.streaming.p2p.webrtc_selection import load_native
    from tests.integration.webrtc.shared import metrics
    from tests.integration.webrtc.shared.frames import (
        decode_frame,
        parse_video_frame_event,
    )
    from tests.integration.webrtc.shared.harness import (
        Relay,
        bootstrap_connection,
        collect_video_frames,
        decoded_counters,
        recv_time,
        submit_at_rate,
    )

    # The adaptation is loss-driven, so a constrained link has a transient
    # head — packets lost while the estimator settles the ladder — before it
    # reaches a steady, fitting rung. The contract ("degrades gracefully,
    # delivered fps holds the floor, no corruption") is a *steady-state*
    # property, so we measure only the tail after the settle window.
    settle = float(os.environ.get("NEURACORE_WEBRTC_NETEM_SETTLE", 12.0))

    native = load_native()
    track_id = "cam0"
    relay = Relay(
        native.Producer(connection_id=None, frame_queue_capacity=16),
        native.Consumer(connection_id=None),
    ).start()
    try:
        relay.producer.add_data_channel("control", "reliable")
        relay.producer.add_video_track(track_id)
        bootstrap_connection(relay)

        # Poll the ladder rung while submitting so the timeline is visible.
        import threading

        steps: list[int] = []
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                s = relay.producer.congestion_step(track_id)
                if s is not None:
                    steps.append(s)
                time.sleep(0.5)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        origin = time.perf_counter()
        sent, _ = submit_at_rate(
            relay, track_id, fps=metrics.SOURCE_FPS, seconds=seconds
        )
        stop.set()
        poller.join(timeout=1.0)
        time.sleep(0.3)
        frames = collect_video_frames(relay, track_id)

        # Whole-window and steady-state (tail) figures. Delivered counts only
        # frames whose embedded checksum verifies — a corrupt frame's recovered
        # counter is garbage and must not inflate the delivered count.
        all_counters, all_corrupted = decoded_counters(frames)
        tail_start = origin + settle
        tail_window = max(seconds - settle, 1e-9)
        tail_ok: set[int] = set()
        tail_corrupt = 0
        for event in frames:
            rt = recv_time(event)
            if rt is None or rt < tail_start:
                continue
            _, _, array = parse_video_frame_event(event)
            counter, ok = decode_frame(array)
            if ok:
                tail_ok.add(counter)
            else:
                tail_corrupt += 1

        max_step = relay.producer.congestion_max_step(track_id)
        final_step = relay.producer.congestion_step(track_id)
        closed = "closed" in relay.state_sequence("consumer")
        return {
            "sent": sent,
            "window_delivered_fps": len(set(all_counters)) / seconds,
            "window_corrupted": len(all_corrupted),
            "delivered_fps": len(tail_ok) / tail_window,
            "corrupted": tail_corrupt,
            "closed": closed,
            "max_step": max_step,
            "final_step": final_step,
            "step_timeline": steps,
            "settle": settle,
        }
    finally:
        relay.close()


def main() -> int:
    seconds = float(os.environ.get("NEURACORE_WEBRTC_NETEM_SECONDS", 24))
    try:
        netem = _setup_link()
        result = _run_relay(seconds)
        result["netem"] = netem
        result["adapt_disabled"] = bool(os.environ.get("NCD_WEBRTC_DISABLE_ADAPT"))
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - report failure as JSON, not a trace
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
