"""Tuning constants for the WebRTC integration suite.

Timeouts are deliberately generous — comfortably above the performance SLOs in
[metrics.py](metrics.py) so green runs do not flake — yet irrelevant to the red
runs, where the first stubbed call raises long before any wait elapses. All are
overridable from the environment so CI can shorten or lengthen them.
"""

from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# Wait for both peers to report on_state "connected" after the first
# negotiation-triggering call (add_data_channel / add_video_track).
CONNECT_TIMEOUT_S = _f("NEURACORE_WEBRTC_CONNECT_TIMEOUT", 5.0)

# Wait for a renegotiation to surface a track add/remove at the consumer.
RENEG_TIMEOUT_S = _f("NEURACORE_WEBRTC_RENEG_TIMEOUT", 3.0)

# Wait for a freshly added data channel to be observed at the consumer.
DC_OPEN_TIMEOUT_S = _f("NEURACORE_WEBRTC_DC_OPEN_TIMEOUT", 3.0)

# Wait for a known count of data-channel messages to arrive at the consumer.
MESSAGE_TIMEOUT_S = _f("NEURACORE_WEBRTC_MESSAGE_TIMEOUT", 5.0)

# After the last frame is submitted, how long to keep draining before deciding
# the decoded video stream has gone quiet.
FRAME_SETTLE_TIMEOUT_S = _f("NEURACORE_WEBRTC_FRAME_SETTLE", 5.0)
