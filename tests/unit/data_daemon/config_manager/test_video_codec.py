"""Tests for the profile-backed video codec writer and the ConfigWatcher cache."""

from __future__ import annotations

import asyncio
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

from neuracore.data_daemon.config_manager.config_watcher import ConfigWatcher
from neuracore.data_daemon.config_manager.daemon_config import DaemonConfig
from neuracore.data_daemon.config_manager.profiles import ProfileManager
from neuracore.data_daemon.config_manager.video_codec import (
    set_active_profile_video_codec,
)
from neuracore.data_daemon.const import active_profile_name


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Root the profile system and recordings dir under a temporary home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("NEURACORE_DAEMON_RECORDINGS_ROOT", str(tmp_path / "rec"))
    monkeypatch.delenv("NEURACORE_DAEMON_PROFILE", raising=False)
    monkeypatch.delenv("NCD_VIDEO_CODEC", raising=False)
    return home


def _profile_codec() -> str | None:
    """Read the codec back from the active profile on disk."""
    return ProfileManager().get_profile(active_profile_name()).video_codec


# --- Writer: set_active_profile_video_codec / nc.set_video_encoding_options ---


def test_set_persists_codec_to_profile(isolated_home: Path) -> None:
    set_active_profile_video_codec("h264_medium")
    assert _profile_codec() == "h264_medium"


def test_set_lossless_reverts_to_the_default_codec(isolated_home: Path) -> None:
    # There is no "clear" — switching back from a lossy mode persists the
    # explicit h264_lossless default, which resolves to the default encoders.
    set_active_profile_video_codec("h264_medium")
    set_active_profile_video_codec("h264_lossless")
    assert _profile_codec() == "h264_lossless"


def test_set_video_encoding_options_api_writes_profile(isolated_home: Path) -> None:
    # nc imported lazily so the isolated_home fixture has already patched
    # Path.home before the (heavy) package import resolves any config.
    import neuracore as nc

    nc.set_video_encoding_options(nc.Codec.H264_MEDIUM)
    assert _profile_codec() == "h264_medium"

    nc.set_video_encoding_options(nc.Codec.H264_LOSSLESS)
    assert _profile_codec() == "h264_lossless"


def test_set_video_encoding_options_rejects_unknown_codec(isolated_home: Path) -> None:
    import neuracore as nc

    with pytest.raises(ValueError, match="Unknown video codec"):
        nc.set_video_encoding_options("not-a-codec")


def test_set_video_encoding_options_nudges_daemon(isolated_home: Path) -> None:
    # A successful set persists the codec AND nudges a running daemon to reload
    # (so a set -> start_recording sequence picks it up immediately).
    import neuracore as nc
    from neuracore.api import logging as logging_api

    with patch.object(logging_api, "notify_daemon_config_changed") as notify:
        nc.set_video_encoding_options(nc.Codec.H264_MEDIUM)
    notify.assert_called_once_with()
    assert _profile_codec() == "h264_medium"


def test_set_video_encoding_options_does_not_nudge_on_invalid_codec(
    isolated_home: Path,
) -> None:
    # An invalid codec must raise before persisting or nudging the daemon.
    import neuracore as nc
    from neuracore.api import logging as logging_api

    with patch.object(logging_api, "notify_daemon_config_changed") as notify:
        with pytest.raises(ValueError, match="Unknown video codec"):
            nc.set_video_encoding_options("not-a-codec")
    notify.assert_not_called()


# --- ConfigWatcher: the in-memory cache the daemon actually reads ---


def test_watcher_seed_value_is_available_immediately() -> None:
    # The launch-resolved config is the seed, so the value is valid before the
    # first background poll.
    watcher = ConfigWatcher(initial_config=DaemonConfig(video_codec="h264_medium"))
    assert watcher.video_codec() == "h264_medium"


@pytest.mark.asyncio
async def test_watcher_refresh_picks_up_change() -> None:
    watcher = ConfigWatcher(
        initial_config=DaemonConfig(video_codec="h264_lossless"),
        resolver=lambda: DaemonConfig(video_codec="h264_medium"),
    )
    assert watcher.video_codec() == "h264_lossless"
    await watcher.refresh()
    assert watcher.video_codec() == "h264_medium"


@pytest.mark.asyncio
async def test_watcher_keeps_last_good_on_resolve_error() -> None:
    def boom() -> DaemonConfig:
        raise RuntimeError("profile read failed")

    watcher = ConfigWatcher(
        initial_config=DaemonConfig(video_codec="h264_medium"),
        resolver=boom,
    )
    await watcher.refresh()
    assert watcher.video_codec() == "h264_medium"


def test_watcher_refresh_now_picks_up_change() -> None:
    # The synchronous variant the encoding path uses on the recording hot path:
    # re-resolves inline (off the loop) so a codec set just before recording is
    # honoured for the new encoder without waiting for the ~1s poll.
    watcher = ConfigWatcher(
        initial_config=DaemonConfig(video_codec="h264_lossless"),
        resolver=lambda: DaemonConfig(video_codec="h264_medium"),
    )
    assert watcher.video_codec() == "h264_lossless"
    watcher.refresh_now()
    assert watcher.video_codec() == "h264_medium"


def test_watcher_refresh_now_keeps_last_good_on_resolve_error() -> None:
    def boom() -> DaemonConfig:
        raise RuntimeError("profile read failed")

    watcher = ConfigWatcher(
        initial_config=DaemonConfig(video_codec="h264_medium"),
        resolver=boom,
    )
    watcher.refresh_now()
    assert watcher.video_codec() == "h264_medium"


@pytest.mark.asyncio
async def test_watcher_background_loop_refreshes() -> None:
    watcher = ConfigWatcher(
        initial_config=DaemonConfig(video_codec="h264_lossless"),
        check_interval=0.01,
        resolver=lambda: DaemonConfig(video_codec="h264_medium"),
    )
    await watcher.start()
    try:
        for _ in range(200):
            if watcher.video_codec() == "h264_medium":
                break
            await asyncio.sleep(0.01)
        assert watcher.video_codec() == "h264_medium"
    finally:
        await watcher.stop()
