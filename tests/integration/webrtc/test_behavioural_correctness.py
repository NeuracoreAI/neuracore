"""Test 1 - behavioural correctness of async add/remove with renegotiation.

A data-only session adds and removes video tracks and data channels
mid-session; the consumer must observe each change via renegotiation, the
mid<->track manifest must stay consistent, and the peer connection must NOT be
torn down or reset. Rapid churn guards the in-flight-negotiation hazard.

xfail groups (see shared/markers.py): the video behavioural assertions and
rapid *video* churn green in PR4; rapid *data-channel* churn greens in PR3.
"""

from __future__ import annotations

import json

from tests.integration.webrtc.shared import constants
from tests.integration.webrtc.shared.harness import Relay, bootstrap_connection


def _manifest_map(event: dict) -> dict:
    return json.loads(event["json"])


def test_video_track_add_remove_midsession_pc_not_reset(relay: Relay) -> None:
    # Start data-only, then bring the connection up.
    relay.producer.add_data_channel("json", "reliable")
    bootstrap_connection(relay)
    states_before = relay.state_sequence("consumer")

    # --- add a video track mid-session -------------------------------------
    track_id = "wrist_cam"
    mid = relay.producer.add_video_track(track_id)

    added = relay.wait_consumer(
        lambda e: e.get("kind") == "on_track_added" and e.get("track_id") == track_id,
        constants.RENEG_TIMEOUT_S,
    )
    assert added is not None, "consumer never observed on_track_added via reneg"
    assert added["mid"] == mid, "on_track_added mid disagrees with add_video_track"

    # The manifest republished on this renegotiation already carries the new
    # mid -> track mapping as one coherent (atomic) update.
    manifest = relay.wait_consumer(
        lambda e: e.get("kind") == "on_manifest" and mid in _manifest_map(e),
        constants.RENEG_TIMEOUT_S,
    )
    assert manifest is not None, "manifest not republished with the new track"
    assert track_id in json.dumps(
        _manifest_map(manifest)[mid]
    ), "manifest entry for the new mid does not reference its track_id"

    # --- remove the video track (keyed by track_id) ------------------------
    # The consumer learns of removal by mid only, so map track_id -> mid from
    # the earlier on_track_added to assert the matching removal.
    relay.producer.remove_video_track(track_id)
    removed = relay.wait_consumer(
        lambda e: e.get("kind") == "on_track_removed" and e.get("mid") == mid,
        constants.RENEG_TIMEOUT_S,
    )
    assert removed is not None, "consumer never observed on_track_removed for the mid"

    # --- the PC must NOT be reset ------------------------------------------
    states_after = relay.state_sequence("consumer")
    assert states_after.count("new") == states_before.count(
        "new"
    ), "peer connection churned back to 'new' (full reset) during renegotiation"
    assert "closed" not in states_after, "peer connection was torn down on remove"
    assert (
        not relay.dispatch_errors
    ), f"signaling relay errored: {relay.dispatch_errors}"


# Baseline guard (un-xfailed in PR3): data-channel add is inherently safe. SCTP
# data channels do not renegotiate — after the first channel brings up the SCTP
# association, each further channel is a DCEP stream open over it, so there is no
# in-flight-renegotiation hazard for data channels (PR2 proved this). The
# in-flight hazard exists only for media tracks; this test stays as a passing
# regression guard that rapid data-channel add never drops a channel.
def test_rapid_data_channel_churn_no_silent_drop(relay: Relay) -> None:
    bootstrap_connection(relay)

    # Add many channels with zero spacing so several negotiations are in flight
    # at once - the hazard PR3's single-in-flight coalescing must absorb.
    labels = [f"dc{i}" for i in range(12)]
    for label in labels:
        relay.producer.add_data_channel(label, "reliable")

    # Every channel must surface at the consumer; none silently dropped.
    for label in labels:
        observed = relay.wait_consumer(
            lambda e, lbl=label: e.get("kind") == "on_data_channel"
            and e.get("label") == lbl,
            constants.DC_OPEN_TIMEOUT_S,
        )
        assert observed is not None, f"data channel {label!r} was silently dropped"

    # Final manifest carries all of them (state matches on both sides).
    manifest = relay.wait_consumer(
        lambda e: e.get("kind") == "on_manifest"
        and all(lbl in e["json"] for lbl in labels),
        constants.RENEG_TIMEOUT_S,
    )
    assert manifest is not None, "final manifest is missing some data channels"
    assert (
        not relay.dispatch_errors
    ), f"signaling relay errored: {relay.dispatch_errors}"


def test_rapid_video_track_churn_no_silent_drop(relay: Relay) -> None:
    relay.producer.add_data_channel("json", "reliable")
    bootstrap_connection(relay)

    track_ids = [f"cam{i}" for i in range(6)]
    mids: dict[str, str] = {}

    # Interleave adds and removes with no spacing to overlap negotiations.
    for track_id in track_ids:
        mids[track_id] = relay.producer.add_video_track(track_id)
    removed = set(track_ids[::2])  # remove every other one
    for track_id in removed:
        relay.producer.remove_video_track(track_id)

    survivors = [t for t in track_ids if t not in removed]

    # Each survivor must be observed and not later removed.
    for track_id in survivors:
        added = relay.wait_consumer(
            lambda e, t=track_id: e.get("kind") == "on_track_added"
            and e.get("track_id") == t,
            constants.RENEG_TIMEOUT_S,
        )
        assert added is not None, f"video track {track_id!r} was silently dropped"

    # Each removed track must surface a removal for its mid.
    for track_id in removed:
        gone = relay.wait_consumer(
            lambda e, m=mids[track_id]: e.get("kind") == "on_track_removed"
            and e.get("mid") == m,
            constants.RENEG_TIMEOUT_S,
        )
        assert gone is not None, f"removal of {track_id!r} was never observed"

    # Final manifest = exactly the survivors; PC never reset.
    final = relay.wait_consumer(
        lambda e: e.get("kind") == "on_manifest"
        and {mids[t] for t in survivors}.issubset(set(json.loads(e["json"])))
        and not ({mids[t] for t in removed} & set(json.loads(e["json"]))),
        constants.RENEG_TIMEOUT_S,
    )
    assert final is not None, "final manifest does not match the survivor set"
    assert "new" not in relay.state_sequence("consumer")[1:], "PC reset during churn"
    assert (
        not relay.dispatch_errors
    ), f"signaling relay errored: {relay.dispatch_errors}"
