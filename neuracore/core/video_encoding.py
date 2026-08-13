"""Video encoding options for recorded camera streams.

By default each RGB camera is stored as a lossless archive (``lossless.mp4``,
used for training) plus a small lossy preview (``lossy.mp4``). Selecting a
:class:`Codec` switches RGB cameras to a single, more compact lossy video that
is also used for training -- trading a little image fidelity for much smaller
uploads, which matters for long recordings.

This module is the single source of truth for the codec identifiers the SDK
exposes. The data daemon mirrors the same ``h264_medium`` ->
``libx264 -crf 23 -preset medium`` mapping.

Depth cameras always keep their lossless storage: their lossy proxy is a
visualisation, not precise depth, so it is never a valid training source.
"""

from __future__ import annotations

import copy
import enum
import logging
import os
from typing import TypedDict

logger = logging.getLogger(__name__)

INFERENCE_CODEC_ENV_VAR = "NEURACORE_INFERENCE_VIDEO_CODEC"
"""Env var naming the codec a served policy's *training data* was encoded with.

Deliberately distinct from the daemon's ``NCD_VIDEO_CODEC``: that one says how the
machine records *today*, which is unrelated to how the data behind an already-trained
model was encoded. A user recording lossy while serving a policy trained on lossless
data would otherwise have their inference frames silently degraded.
"""


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


def resolve_inference_codec_options() -> Libx264Options | None:
    """Return the lossy encoder options a policy's training data was encoded with.

    Reads :data:`INFERENCE_CODEC_ENV_VAR` and resolves it through the same
    :func:`codec_option_overrides` the recording path uses, so the CRF/preset an
    inference-time re-encode targets can never drift from what the daemon wrote.

    ``None`` means "do not degrade": the var is unset, names the lossless default,
    or is unrecognised (:func:`resolve_codec` warns in that last case). Frames then
    reach the model untouched, which is the correct behaviour for a model trained on
    the bit-exact ``lossless.mp4`` archive.

    Note:
        This is an interim mechanism. The encoding of a dataset is not recorded at
        training time, so it cannot currently be read back off a trained model.

    Returns:
        A fresh options dict for the lossy encoder, or ``None`` to leave frames as-is.
    """
    return codec_option_overrides(os.getenv(INFERENCE_CODEC_ENV_VAR))
