"""Video encoding options for recorded camera streams.

By default each RGB camera is stored as a lossless archive (``lossless.mp4``,
used for training) plus a small lossy preview (``lossy.mp4``). Selecting a
:class:`Codec` switches RGB cameras to a single, more compact lossy video that
is also used for training -- trading a little image fidelity for much smaller
uploads, which matters for long recordings.

Codec is re-exported here so nc.Codec keeps working. The data daemon mirrors
the same h264_medium -> libx264 -crf 23 -preset medium mapping.

Depth cameras always keep their lossless storage: their lossy proxy is a
visualisation, not precise depth, so it is never a valid training source.
"""

from __future__ import annotations

import copy
import logging
from typing import TypedDict

from neuracore_types import Codec

__all__ = ["Codec", "Libx264Options", "codec_option_overrides", "resolve_codec"]

logger = logging.getLogger(__name__)


class Libx264Options(TypedDict, total=False):
    """libx264 codec-context options for the lossy RGB encoder."""

    crf: str
    preset: str


_LOSSY_CODEC_OPTIONS: dict[Codec, Libx264Options] = {
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


def codec_option_overrides(value: str | None) -> Libx264Options | None:
    """Return the lossy-only libx264 options for a codec string, or ``None``.

    ``None`` selects the default lossless-plus-preview encoders; a dict selects a
    single lossy RGB video with those options. A fresh copy is returned so the
    caller can mutate it safely.

    Args:
        value: The configured codec identifier, or ``None``.

    Returns:
        A fresh options dict for the lossy encoder, or ``None`` for the default.
    """
    codec = resolve_codec(value)
    if codec is None:
        return None
    options = _LOSSY_CODEC_OPTIONS.get(codec)
    return copy.copy(options) if options is not None else None
