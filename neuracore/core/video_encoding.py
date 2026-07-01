"""Video encoding options for recorded camera streams.

By default each RGB camera is stored as a lossless archive (``lossless.mp4``,
used for training) plus a small lossy preview (``lossy.mp4``). Selecting a
:class:`Codec` switches RGB cameras to a single, more compact lossy video that
is also used for training -- trading a little image fidelity for much smaller
uploads, which matters for long recordings.

This module is the single source of truth for the codec identifiers and their
ffmpeg/PyAV settings on the Python side. The Rust data daemon mirrors the same
``h264_medium`` -> ``libx264 -crf 23 -preset medium`` mapping.

Depth cameras always keep their lossless storage: their lossy proxy is a
visualisation, not precise depth, so it is never a valid training source.
"""

from __future__ import annotations

import enum
import logging

logger = logging.getLogger(__name__)


class Codec(str, enum.Enum):
    """Codecs available for recorded camera video.

    Members are string values so they round-trip cleanly through the daemon
    profile and the ``NCD_VIDEO_CODEC`` environment variable.

    Attributes:
        H264_LOSSLESS: The default — keep the lossless archive (used for
            training) plus a small lossy preview. Select this to switch back
            from a lossy mode.
        H264_MEDIUM: Lossy-only — a single full-resolution libx264 CRF 23 video
            (no lossless archive), used for both preview and training.
    """

    H264_LOSSLESS = "h264_lossless"
    H264_MEDIUM = "h264_medium"


# Lossy libx264 options per codec, passed straight to PyAV/ffmpeg as
# codec-context options for the (yuv420p) RGB encoder. The Rust daemon keeps an
# equivalent mapping in its video encoder. Codecs absent here (e.g.
# ``H264_LOSSLESS``) use the default lossless+lossy encoders.
_LOSSY_CODEC_OPTIONS: dict[Codec, dict[str, str]] = {
    Codec.H264_MEDIUM: {"crf": "23", "preset": "medium"},
}


def resolve_codec(value: str | None) -> Codec | None:
    """Resolve a config/env string to a :class:`Codec`, or ``None``.

    Empty/unset values resolve to ``None`` silently; an unrecognised value also
    resolves to ``None`` but logs a warning. Either way a stray env or config
    value can never break recording -- it simply falls back to the default
    lossless+lossy encoders.

    Args:
        value: The configured codec identifier, or ``None``.

    Returns:
        The matching :class:`Codec`, or ``None`` when unset/unknown.
    """
    if not value:
        return None
    try:
        return Codec(value)
    except ValueError:
        logger.warning(
            "Ignoring unknown video codec %r; expected one of: %s",
            value,
            ", ".join(member.value for member in Codec),
        )
        return None


def lossy_only_codec_options(value: str | None) -> dict[str, str] | None:
    """Return the lossy libx264 options for a codec string, or ``None``.

    ``None`` means "use the default lossless archive plus lossy preview"
    (returned for unset, unknown, and the explicit ``H264_LOSSLESS`` default).
    A dict means "encode a single lossy RGB video with these options and drop
    the lossless archive".

    Args:
        value: The configured codec identifier, or ``None``.

    Returns:
        A fresh options dict for the lossy encoder, or ``None`` for the default.
    """
    codec = resolve_codec(value)
    if codec is None:
        return None
    options = _LOSSY_CODEC_OPTIONS.get(codec)
    return dict(options) if options is not None else None
