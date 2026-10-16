"""Check that imagecodecs decodes the depth archive fixture codestream bit-exact."""

from pathlib import Path

import numpy as np
import pytest

FIXTURES = (
    Path(__file__).resolve().parents[3] / "rust" / "data_daemon" / "tests" / "fixtures"
)


def test_imagecodecs_decodes_the_depth_archive_fixture_bit_exact() -> None:
    """Decode the fixture codestream to the exact uint16 samples, zeros included."""
    imagecodecs = pytest.importorskip("imagecodecs")
    expected = np.fromfile(FIXTURES / "depth_16x12.u16le", dtype="<u2").reshape(12, 16)

    decoded = imagecodecs.jpegxl_decode((FIXTURES / "depth_16x12.jxl").read_bytes())

    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, expected)
    assert (decoded == 0).sum() == (expected == 0).sum() > 0
