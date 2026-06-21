"""Unit tests for the perf metrics helpers (PR1 ``shared/metrics.py``).

Pins the percentile maths on known inputs and the structured CI output schema
``emit`` writes — the contract CI scrapes.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from tests.integration.webrtc.shared import metrics
from tests.integration.webrtc.shared.metrics import Metrics, emit, percentile


def test_percentile_on_known_inputs() -> None:
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(data, 0) == 10.0
    assert percentile(data, 50) == 30.0  # median
    assert percentile(data, 100) == 50.0
    assert percentile(data, 25) == 20.0


def test_percentile_linear_interpolation_between_ranks() -> None:
    # rank = (4-1)*0.5 = 1.5 -> halfway between ordered[1]=2 and ordered[2]=3.
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_is_order_independent() -> None:
    assert percentile([50.0, 10.0, 40.0, 20.0, 30.0], 50) == 30.0


def test_percentile_single_sample_returns_that_sample() -> None:
    assert percentile([7.5], 95) == 7.5


def test_percentile_of_empty_raises() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_emit_writes_the_full_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys
) -> None:
    out = tmp_path / "perf.json"
    monkeypatch.setenv("NEURACORE_WEBRTC_PERF_OUT", str(out))

    sample = Metrics(
        connect_ms=57.0,
        reneg_add_ms=4.7,
        reneg_remove_ms=5.6,
        dc_add_ms=2.2,
        g2g_p50_ms=80.0,
        g2g_p95_ms=150.0,
        delivered_fps=42.0,
        drop_rate=0.06,
    )
    payload = emit(sample)

    # Every field of the dataclass is a schema key, present in all three sinks:
    # the returned payload, the file, and the stderr line.
    expected_keys = {f.name for f in dataclasses.fields(Metrics)}
    assert expected_keys == {
        "connect_ms",
        "reneg_add_ms",
        "reneg_remove_ms",
        "dc_add_ms",
        "g2g_p50_ms",
        "g2g_p95_ms",
        "delivered_fps",
        "drop_rate",
    }
    assert set(json.loads(payload)) == expected_keys
    assert set(json.loads(out.read_text())) == expected_keys
    assert "[neuracore-webrtc-perf]" in capsys.readouterr().err


def test_emit_null_fields_still_pin_the_schema(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    # An all-None Metrics (nothing measured this run) still emits every key.
    monkeypatch.delenv("NEURACORE_WEBRTC_PERF_OUT", raising=False)
    payload = json.loads(emit(Metrics()))
    assert set(payload) == {f.name for f in dataclasses.fields(Metrics)}
    assert all(value is None for value in payload.values())


def test_slo_constants_are_present() -> None:
    # The SLO thresholds the perf suite asserts against are part of the contract.
    assert metrics.CONNECT_MS_P95 == 500.0
    assert metrics.RENEG_ADD_MS_P95 == 300.0
    assert metrics.RENEG_REMOVE_MS_P95 == 300.0
    assert metrics.DC_ADD_MS_P95 == 300.0
    assert metrics.MIN_DELIVERED_FPS == 30.0
