"""Unit tests for the feature-flag selection and native loader (PR0).

``rust_webrtc_enabled`` gates the Rust stack on ``NCD_RUST_WEBRTC``;
``load_native`` imports the compiled extension and, when it is absent, raises a
``RuntimeError`` carrying a build/fallback hint. Neither path needs the native
module to be built — the import is monkeypatched.
"""

from __future__ import annotations

from types import ModuleType

import pytest

from neuracore.core.streaming.p2p import webrtc_selection


@pytest.mark.parametrize("value", ["1", "true", "True", "YES", "y", "  yes  "])
def test_rust_webrtc_enabled_truthy_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("NCD_RUST_WEBRTC", value)
    assert webrtc_selection.rust_webrtc_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "n", "", "off", "2"])
def test_rust_webrtc_enabled_falsy_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("NCD_RUST_WEBRTC", value)
    assert webrtc_selection.rust_webrtc_enabled() is False


def test_rust_webrtc_enabled_unset_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCD_RUST_WEBRTC", raising=False)
    assert webrtc_selection.rust_webrtc_enabled() is False


def test_load_native_raises_a_hinted_runtime_error_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force a fresh import attempt and make the extension import fail.
    monkeypatch.setattr(webrtc_selection, "_NATIVE_MODULE", None)

    def _missing(name: str) -> ModuleType:
        raise ImportError(f"no module named {name!r}")

    monkeypatch.setattr(webrtc_selection, "import_module", _missing)

    with pytest.raises(RuntimeError) as excinfo:
        webrtc_selection.load_native()

    # The hint must point at the build script and the aiortc fallback.
    message = str(excinfo.value)
    assert "build_wheel_artefacts.sh" in message
    assert "NCD_RUST_WEBRTC" in message
    # The original ImportError is chained for debuggability.
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_load_native_caches_and_returns_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(webrtc_selection, "_NATIVE_MODULE", None)
    sentinel = ModuleType("fake_native_webrtc")
    calls: list[str] = []

    def _import(name: str) -> ModuleType:
        calls.append(name)
        return sentinel

    monkeypatch.setattr(webrtc_selection, "import_module", _import)

    first = webrtc_selection.load_native()
    second = webrtc_selection.load_native()

    assert first is sentinel and second is sentinel
    # Imported once (the documented dotted path), then served from the cache.
    assert calls == ["neuracore.core.streaming.p2p._native_webrtc"]
