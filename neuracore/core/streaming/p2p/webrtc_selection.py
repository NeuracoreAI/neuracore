"""Feature-flag selection and native loader for the Rust WebRTC stack.

Mirrors [rust_selection.py](neuracore/data_daemon/rust_selection.py): one place
for the ``NCD_RUST_WEBRTC`` environment-variable check that gates the new Rust
streaming core alongside the existing aiortc path, and for importing the
compiled PyO3 module that exposes the ``Producer``/``Consumer`` entry points.

The new stack lives in the ``neuracore_webrtc`` Rust crate, built into the
package tree as ``neuracore.core.streaming.p2p._native_webrtc`` by
[build_wheel_artefacts.sh](rust/scripts/build_wheel_artefacts.sh). When the flag
is off (the default), nothing here is imported and the aiortc connections in
[provider/](neuracore/core/streaming/p2p/provider/) and
[consumer/](neuracore/core/streaming/p2p/consumer/) are used unchanged.

Kept dependency-free so it can be imported without pulling in the streaming
runtime or aiortc.
"""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "y"})

_NATIVE_MODULE: ModuleType | None = None

_NATIVE_IMPORT_HINT = (
    "neuracore.core.streaming.p2p._native_webrtc is not available. Build the "
    "Rust neuracore_webrtc crate with rust/scripts/build_wheel_artefacts.sh "
    "(which places the extension in the package tree), or unset NCD_RUST_WEBRTC "
    "to use the legacy aiortc streaming path."
)


def rust_webrtc_enabled() -> bool:
    """Return True when ``NCD_RUST_WEBRTC`` selects the Rust WebRTC stack."""
    return os.environ.get("NCD_RUST_WEBRTC", "").strip().lower() in _TRUTHY_VALUES


def load_native() -> ModuleType:
    """Lazily import and cache the PyO3 WebRTC module for the process.

    Raises:
        RuntimeError: if the compiled extension is not importable, with a hint
            on how to build it or how to fall back to aiortc.
    """
    global _NATIVE_MODULE
    if _NATIVE_MODULE is None:
        try:
            _NATIVE_MODULE = import_module(
                "neuracore.core.streaming.p2p._native_webrtc"
            )
        except ImportError as error:
            raise RuntimeError(_NATIVE_IMPORT_HINT) from error
    return _NATIVE_MODULE
