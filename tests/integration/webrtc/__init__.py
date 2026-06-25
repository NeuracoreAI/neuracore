"""In-process integration suite for the Rust WebRTC streaming core.

Two `neuracore` native peers — a `Producer` (sole offerer) and a `Consumer`
(answer-only) — are wired together over an in-process signaling relay (no
separate signaling server). The suite drives asynchronous add/remove of data
and video streams with renegotiation across three tests: behavioural
correctness, data integrity, and performance.

Written red-first against the PR0 stubs: every assertion that depends on real
WebRTC behaviour is marked ``xfail(strict=True)`` and grouped by the PR that
greens it. Later PRs flip a slice to green by deleting its marker only — never
by editing an assertion. See [markers.py](shared/markers.py) for the map.
"""
