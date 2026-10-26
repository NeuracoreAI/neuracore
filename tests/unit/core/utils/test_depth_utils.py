"""Tests for the depth viewer log curve."""

import re
from pathlib import Path

import numpy as np

from neuracore.core.utils import depth_utils
from neuracore.core.utils.depth_utils import (
    DEPTH_PREVIEW_MAX_M,
    DEPTH_PREVIEW_MIN_M,
    depth_to_log_gray,
    log_gray_to_depth,
)

# Levels the data daemon's ffmpeg lut filter writes for D405 samples at
# 0.1 mm per unit. Checked against the filter on every uint16 value.
DAEMON_LEVELS_AT_0_1_MM = [
    (0, 0),
    (1, 1),
    (500, 1),
    (501, 1),
    (1000, 7),
    (2500, 23),
    (5000, 47),
    (7543, 67),
    (10000, 84),
    (20000, 136),
    (30000, 173),
    (45000, 214),
    (60000, 245),
    (65500, 255),
    (65535, 255),
]


def test_log_gray_matches_the_daemon_viewer_video_levels():
    """Map D405 samples to the grey levels the daemon writes."""
    samples = np.array([sample for sample, _ in DAEMON_LEVELS_AT_0_1_MM])
    levels = depth_to_log_gray(samples.astype(np.float64) * 1e-4)

    assert levels.dtype == np.uint8
    assert levels.tolist() == [level for _, level in DAEMON_LEVELS_AT_0_1_MM]


def test_log_gray_maps_holes_and_invalid_depth_to_zero():
    """Keep level 0 for no return, NaN, infinity and negative depth."""
    depth = np.array([0.0, np.nan, np.inf, -1.0, 1e-9])

    assert depth_to_log_gray(depth).tolist() == [0, 0, 0, 0, 1]


def test_log_gray_round_trips_within_half_a_level():
    """Recover depth within half a level step across the viewer range."""
    depth = np.linspace(DEPTH_PREVIEW_MIN_M, DEPTH_PREVIEW_MAX_M, 1000)

    recovered = log_gray_to_depth(depth_to_log_gray(depth))

    step = np.log(DEPTH_PREVIEW_MAX_M + 1.0) - np.log(DEPTH_PREVIEW_MIN_M + 1.0)
    log_error = np.abs(np.log(recovered + 1.0) - np.log(depth + 1.0))
    assert log_error.max() <= step / 254 / 2 + 1e-6
    assert log_gray_to_depth(np.array([0]))[0] == 0.0


def test_log_curve_constants_match_the_daemon():
    """Match the log curve constants of the data daemon's viewer video encoder."""
    encoder_source = (
        Path(__file__).resolve().parents[4]
        / "rust"
        / "data_daemon"
        / "src"
        / "encoding"
        / "video_encoder.rs"
    ).read_text(encoding="utf-8")
    daemon_constants = {
        name: float(value)
        for name, value in re.findall(
            r"pub const (DEPTH_PREVIEW_\w+): (?:f64|u32) = ([\d.]+);",
            encoder_source,
        )
    }

    assert daemon_constants == {
        name: getattr(depth_utils, name)
        for name in (
            "DEPTH_PREVIEW_SHIFT_M",
            "DEPTH_PREVIEW_MIN_M",
            "DEPTH_PREVIEW_MAX_M",
            "DEPTH_PREVIEW_LEVELS",
        )
    }
