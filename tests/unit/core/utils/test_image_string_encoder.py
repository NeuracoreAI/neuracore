"""Tests for the data URI image encoder used on streams."""

import base64
import io

import numpy as np
from PIL import Image

from neuracore.core.utils.image_string_encoder import (
    DATA_URI_PREFIX,
    ImageStringEncoder,
)


def _pixels():
    return np.random.default_rng(0).integers(0, 256, (12, 16, 3), dtype=np.uint8)


def _pil_data_uri(pixels):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return DATA_URI_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def test_encoded_image_is_a_png_that_pillow_reads_back_exactly():
    pixels = _pixels()

    encoded = ImageStringEncoder.encode_image(pixels)

    assert encoded.startswith(DATA_URI_PREFIX)
    png = base64.b64decode(encoded.removeprefix(DATA_URI_PREFIX))
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG"
        np.testing.assert_array_equal(np.array(image), pixels)


def test_decode_reads_a_png_written_by_pillow():
    pixels = _pixels()

    decoded = ImageStringEncoder.decode_image(_pil_data_uri(pixels))

    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, pixels)


def test_encode_accepts_non_contiguous_arrays():
    view = _pixels()[:, :, ::-1]
    assert not view.flags["C_CONTIGUOUS"]

    decoded = ImageStringEncoder.decode_image(ImageStringEncoder.encode_image(view))

    np.testing.assert_array_equal(decoded, view)
