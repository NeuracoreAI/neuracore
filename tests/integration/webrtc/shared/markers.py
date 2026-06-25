"""Centralised xfail markers grouping each assertion to the PR that greens it.

Every test in this suite depends on real WebRTC behaviour the PR0 stubs do not
provide: the protocol methods raise ``NotImplementedError`` and the
signaling-out events (``on_local_description`` / ``on_local_candidate``) are
never emitted. Each test is therefore expected to *fail* against the stubs.

We mark every such test ``xfail(strict=True)`` so that:

  * a test that fails or errors against the stubs is reported ``xfailed`` — the
    expected red state for this PR, and
  * a test that *passes* is reported ``XPASS`` which, under ``strict``, turns
    the run RED. That is the signal to the implementing PR: the behaviour is now
    real, so DELETE this marker (never edit the assertion to keep it red).

Markers are grouped by the PR that makes the slice real. A later PR greens its
slice by removing exactly the marker(s) tagged with its number. Because the PRs
land in order (PR2 < PR3 < ... < PR7), a later-PR test may freely rely on
earlier-PR behaviour being real by the time its own marker is removed (e.g. a
PR4 video test bootstraps a PR2 data channel).
"""

from __future__ import annotations

import pytest

# PR number -> one-line description of the slice that PR turns green. The keys
# are the only valid `pr` arguments to `greened_by`; the report's
# test-name -> marker-group table is generated from this same mapping.
PR_SLICES: dict[str, str] = {
    "PR2": (
        "data channel send/recv integrity (zero loss/reorder), data channel "
        "add observed at consumer, mid->track manifest, connect & dc-add timing"
    ),
    "PR3": (
        "rapid data-channel add correctness under coalesced in-flight "
        "renegotiation (no channel silently dropped)"
    ),
    "PR4": (
        "video track add/remove via renegotiation, manifest atomicity, "
        "PC-not-reset, rapid video churn, add/remove reneg timing"
    ),
    "PR5": (
        "video frame integrity (monotonic counters, no corruption), "
        "glass-to-glass latency, sustained fps from a 45fps source (single "
        "consumer)"
    ),
    "PR6": "performance under a constrained link (netem-shaped loopback)",
    "PR7": "multi-consumer performance (per-consumer SLOs hold with N consumers)",
}


def greened_by(pr: str, detail: str) -> pytest.MarkDecorator:
    """Return an ``xfail(strict=True)`` marker tagged to the PR that greens it.

    Args:
        pr: one of the keys in :data:`PR_SLICES` (e.g. ``"PR4"``).
        detail: a one-line description of what this specific test asserts, used
            in the xfail reason so a red run is self-documenting.
    """
    if pr not in PR_SLICES:
        raise KeyError(f"unknown PR slice {pr!r}; expected one of {sorted(PR_SLICES)}")
    return pytest.mark.xfail(
        strict=True,
        reason=f"red until {pr} greens [{PR_SLICES[pr]}] -- {detail}",
    )
