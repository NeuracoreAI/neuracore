"""Test 2 - data integrity over reliable channels and over video.

Reliable-ordered data channels must deliver every message exactly once and in
order; decoded video frames must carry monotonic, uncorrupted counters.

xfail groups (see shared/markers.py): data-channel delivery, data-channel add,
and the manifest green in PR2; video frame integrity greens in PR5.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from tests.integration.webrtc.shared import constants, metrics
from tests.integration.webrtc.shared.harness import (
    BroadcastRelay,
    Relay,
    bootstrap_connection,
    collect_video_frames,
    decoded_counters,
    submit_at_rate,
)

BroadcastFactory = Callable[..., BroadcastRelay]


def test_data_channels_zero_loss_zero_reorder(relay: Relay) -> None:
    relay.producer.add_data_channel("json", "reliable")
    relay.producer.add_data_channel("joints", "reliable")
    bootstrap_connection(relay)

    json_seq = [{"i": i, "kind": "json", "payload": f"msg-{i}"} for i in range(50)]
    joints_seq = [{"i": i, "q": [float(i + j) for j in range(7)]} for i in range(50)]

    for payload in json_seq:
        relay.producer.send_json("json", json.dumps(payload))
    for payload in joints_seq:
        relay.producer.send_json("joints", json.dumps(payload))

    got_json = relay.wait_messages(
        "consumer", "json", len(json_seq), constants.MESSAGE_TIMEOUT_S
    )
    got_joints = relay.wait_messages(
        "consumer", "joints", len(joints_seq), constants.MESSAGE_TIMEOUT_S
    )

    # Completeness + ordering in one comparison: decode in arrival order and
    # require exact equality with what was sent.
    assert [json.loads(d) for d in got_json] == json_seq, "json loss or reorder"
    assert [json.loads(d) for d in got_joints] == joints_seq, "joints loss or reorder"


def test_data_channel_add_observed_with_manifest(relay: Relay) -> None:
    bootstrap_connection(relay)

    relay.producer.add_data_channel("telemetry", "reliable")

    observed = relay.wait_consumer(
        lambda e: e.get("kind") == "on_data_channel" and e.get("label") == "telemetry",
        constants.DC_OPEN_TIMEOUT_S,
    )
    assert observed is not None, "consumer never observed the new data channel"

    manifest = relay.wait_consumer(
        lambda e: e.get("kind") == "on_manifest" and "telemetry" in e["json"],
        constants.RENEG_TIMEOUT_S,
    )
    assert manifest is not None, "manifest not republished with the new channel"
    assert isinstance(json.loads(manifest["json"]), dict), "manifest is not a mid map"


def test_video_frames_monotonic_and_intact(relay: Relay) -> None:
    track_id = "cam0"
    relay.producer.add_data_channel("json", "reliable")
    relay.producer.add_video_track(track_id)
    bootstrap_connection(relay)

    submitted, _ = submit_at_rate(relay, track_id, fps=30, seconds=4)
    assert submitted > 0

    # allow the tail of the pipeline to flush
    time.sleep(0.2)
    frames = collect_video_frames(relay, track_id)
    assert frames, "no video frames delivered to the consumer"

    counters, corrupted = decoded_counters(frames)
    assert not corrupted, f"corrupted frames detected at counters {corrupted}"
    # Drops are allowed (lossy ingress/network); reorder and duplication are not.
    assert counters == sorted(counters), "frames delivered out of order"
    assert len(counters) == len(set(counters)), "duplicate frames delivered"


def test_multi_consumer_data_zero_loss_zero_reorder(
    make_broadcast: BroadcastFactory,
) -> None:
    """One Broadcaster fans json/joints data channels to N consumers losslessly.

    The data analogue of ``test_multi_consumer_perf``: every consumer must receive
    the exact known sequence on each reliable channel with zero loss and zero
    reorder. ``json`` is registered before the consumers join (each gets it at
    bootstrap) and ``joints`` after they connect (DCEP over the live association,
    no renegotiation — PR2), so both the bootstrap and the live-add fan-out paths
    are covered.
    """
    n = metrics.MULTI_CONSUMER_N
    relay = make_broadcast()

    # Registered before any consumer: late joiners pick it up at bootstrap.
    relay.broadcaster.add_data_channel("json", "reliable")

    consumer_ids = [f"c{i}" for i in range(n)]
    for consumer_id in consumer_ids:
        make_broadcast.add_consumer(relay, consumer_id)
    for consumer_id in consumer_ids:
        assert relay.wait_consumer_connected(
            consumer_id
        ), f"consumer {consumer_id} did not connect"

    # Added on the live association: every connected consumer gets it via DCEP.
    relay.broadcaster.add_data_channel("joints", "reliable")

    json_seq = [{"i": i, "kind": "json", "payload": f"msg-{i}"} for i in range(50)]
    joints_seq = [{"i": i, "q": [float(i + j) for j in range(7)]} for i in range(50)]

    for payload in json_seq:
        relay.broadcaster.send_json("json", json.dumps(payload))
    for payload in joints_seq:
        relay.broadcaster.send_json("joints", json.dumps(payload))

    # Every consumer receives both full sequences, in order, exactly once.
    for consumer_id in consumer_ids:
        got_json = relay.wait_messages(
            consumer_id, "json", len(json_seq), constants.MESSAGE_TIMEOUT_S
        )
        got_joints = relay.wait_messages(
            consumer_id, "joints", len(joints_seq), constants.MESSAGE_TIMEOUT_S
        )
        assert [
            json.loads(d) for d in got_json
        ] == json_seq, f"consumer {consumer_id}: json loss or reorder"
        assert [
            json.loads(d) for d in got_joints
        ] == joints_seq, f"consumer {consumer_id}: joints loss or reorder"
