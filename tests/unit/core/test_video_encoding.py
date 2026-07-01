"""Tests for the SDK video encoding codec helpers."""

from __future__ import annotations

import logging

import pytest

from neuracore.core.video_encoding import Codec, lossy_only_codec_options, resolve_codec

_MODULE_LOGGER = "neuracore.core.video_encoding"


def test_h264_medium_resolves_and_maps_to_crf_23_medium() -> None:
    assert resolve_codec("h264_medium") is Codec.H264_MEDIUM
    assert lossy_only_codec_options("h264_medium") == {
        "crf": "23",
        "preset": "medium",
    }


def test_h264_lossless_resolves_but_uses_default_encoders() -> None:
    # H264_LOSSLESS is a recognised codec (no warning) that maps to the default
    # lossless+lossy encoders, so it is the explicit "switch back" value.
    assert resolve_codec("h264_lossless") is Codec.H264_LOSSLESS
    assert lossy_only_codec_options("h264_lossless") is None


def test_unset_and_unknown_resolve_to_default() -> None:
    for value in (None, "", "not-a-codec"):
        assert resolve_codec(value) is None
        assert lossy_only_codec_options(value) is None


def test_options_are_a_fresh_copy_each_call() -> None:
    first = lossy_only_codec_options("h264_medium")
    assert first is not None
    first["crf"] = "0"
    # Mutating the returned dict must not corrupt the source-of-truth mapping.
    assert lossy_only_codec_options("h264_medium") == {
        "crf": "23",
        "preset": "medium",
    }


def test_unknown_codec_logs_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_MODULE_LOGGER):
        assert resolve_codec("not-a-codec") is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "not-a-codec" in warnings[0].getMessage()


def test_known_and_unset_codecs_do_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The explicit default and unset/empty values are silent — only a genuinely
    # unrecognised value warns.
    with caplog.at_level(logging.WARNING, logger=_MODULE_LOGGER):
        resolve_codec("h264_lossless")
        resolve_codec("h264_medium")
        resolve_codec(None)
        resolve_codec("")
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
