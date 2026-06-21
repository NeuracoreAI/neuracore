"""Performance SLOs, percentile maths, and the structured CI output schema.

The performance test (Test 3) asserts each timing against the agreed SLO and
records it into a single :class:`Metrics` dataclass. At session teardown the
dataclass is emitted as one JSON object so CI can scrape it. The schema is
fixed here so it is stable across the PRs that progressively fill it in.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

# --- Agreed SLOs (the thresholds the perf tests assert against) ---------------
CONNECT_MS_P95 = 500.0  # connection established under 500ms p95 (trickle on)
RENEG_ADD_MS_P95 = 300.0  # add-track renegotiation under 300ms p95
RENEG_REMOVE_MS_P95 = 300.0  # remove-track renegotiation under 300ms p95
DC_ADD_MS_P95 = 300.0  # data channel add usable at consumer under 300ms p95
G2G_P50_MS = 120.0  # glass-to-glass under 120ms p50
G2G_P95_MS = 200.0  # glass-to-glass under 200ms p95
MIN_DELIVERED_FPS = 30.0  # delivered fps floor from an over-rate source

# --- Workload knobs (overridable for CI) --------------------------------------
SOURCE_FPS = float(os.environ.get("NEURACORE_WEBRTC_SOURCE_FPS", 45))
AT_RATE_FPS = float(os.environ.get("NEURACORE_WEBRTC_AT_RATE_FPS", 30))
PERF_DURATION_S = float(os.environ.get("NEURACORE_WEBRTC_PERF_SECONDS", 60))
PERF_SAMPLES = int(os.environ.get("NEURACORE_WEBRTC_PERF_SAMPLES", 20))
MULTI_CONSUMER_N = int(os.environ.get("NEURACORE_WEBRTC_CONSUMERS", 3))


@dataclass
class Metrics:
    """Structured perf output for CI. ``None`` means "not measured this run".

    Field names are the CI contract — do not rename without updating the report.
    """

    connect_ms: float | None = None
    reneg_add_ms: float | None = None
    reneg_remove_ms: float | None = None
    dc_add_ms: float | None = None
    g2g_p50_ms: float | None = None
    g2g_p95_ms: float | None = None
    delivered_fps: float | None = None
    drop_rate: float | None = None


def percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile (``pct`` in 0..100)."""
    if not samples:
        raise ValueError("percentile of empty sample set")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def emit(metrics: Metrics, *, label: str = "neuracore-webrtc-perf") -> str:
    """Emit ``metrics`` as one JSON line for CI; return the JSON payload.

    Always prints ``[label] {json}`` to stderr. If ``NEURACORE_WEBRTC_PERF_OUT``
    is set, also writes the JSON to that path.
    """
    payload = json.dumps(asdict(metrics), sort_keys=True)
    print(f"\n[{label}] {payload}", file=sys.stderr, flush=True)
    out_path = os.environ.get("NEURACORE_WEBRTC_PERF_OUT")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return payload
