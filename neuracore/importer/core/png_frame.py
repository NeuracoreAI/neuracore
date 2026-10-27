"""RGB frames kept as PNG so they reach the daemon without a decode."""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass

import numpy as np
from neuracore_types.importer.transform import (
    CastToNumpyDtype,
    Clip,
    DataTransformSequence,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR = b"\x00\x00\x00\rIHDR"
_PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"
_PNG_COLOR_TYPE_RGB = 2


@dataclass(frozen=True)
class PNGFrame:
    """An 8-bit RGB, non-interlaced PNG frame."""

    data: bytes
    width: int
    height: int

    @classmethod
    def from_bytes(cls, data: bytes) -> PNGFrame | None:
        """Return a PNGFrame, or None if data is not a whole 8-bit RGB PNG."""
        if (
            len(data) < 29 + len(_PNG_IEND)
            or data[:8] != _PNG_SIGNATURE
            or data[8:16] != _PNG_IHDR
            or data[-len(_PNG_IEND) :] != _PNG_IEND
        ):
            return None
        width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
            ">IIBBBBB", data[16:29]
        )
        if bit_depth != 8 or color_type != _PNG_COLOR_TYPE_RGB or interlace != 0:
            return None
        return cls(data=data, width=width, height=height)

    def decode(self) -> np.ndarray:
        """Decode to an HxWx3 uint8 array."""
        from PIL import Image

        with Image.open(io.BytesIO(self.data)) as image:
            return np.array(image)


def keeps_uint8_rgb_unchanged(transforms: DataTransformSequence) -> bool:
    """Whether transforms leave an HxWx3 uint8 frame as it is."""
    for transform in transforms.transforms:
        if isinstance(transform, Clip):
            if transform.min > 0 or transform.max < 255:
                return False
        elif isinstance(transform, CastToNumpyDtype):
            if np.dtype(transform.dtype) != np.uint8:
                return False
        else:
            return False
    return True
