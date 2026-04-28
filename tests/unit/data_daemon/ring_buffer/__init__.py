"""Legacy byte-ring tests.

The production ring transport is now shared-only and frame-based, so the
historical byte-ring suite is intentionally skipped.
"""

import pytest

pytest.skip(
    "Legacy byte-ring tests are out of scope for shared-only ring transport",
    allow_module_level=True,
)
