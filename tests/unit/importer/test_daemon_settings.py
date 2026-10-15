from collections import namedtuple

from neuracore_types import Codec

from neuracore.importer.core import daemon_settings
from neuracore.importer.core.daemon_settings import (
    MIN_SPOOL_LIMIT_BYTES,
    ImporterDaemonConfig,
    auto_spool_limit_bytes,
)

_Usage = namedtuple("_Usage", "total used free")


def test_auto_spool_limit_is_a_quarter_of_free_disk(monkeypatch, tmp_path):
    free = 100 * 1024**3
    monkeypatch.setattr(
        daemon_settings.shutil, "disk_usage", lambda _: _Usage(0, 0, free)
    )

    assert auto_spool_limit_bytes(tmp_path / "missing" / "root") == int(free * 0.25)


def test_auto_spool_limit_has_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(daemon_settings.shutil, "disk_usage", lambda _: _Usage(0, 0, 1))

    assert auto_spool_limit_bytes(tmp_path) == MIN_SPOOL_LIMIT_BYTES


def test_env_sets_only_given_values():
    config = ImporterDaemonConfig(
        spool_limit=1024, upload_workers=8, video_codec=Codec.H264_FAST
    )

    assert config.env() == {
        "NCD_SPOOL_LIMIT": "1024",
        "NCD_UPLOAD_WORKERS": "8",
        "NCD_VIDEO_CODEC": "h264_fast",
    }
    assert config.sets_daemon_launch_options


def test_env_uses_auto_spool_limit_by_default(monkeypatch):
    monkeypatch.setattr(daemon_settings, "auto_spool_limit_bytes", lambda: 42)

    assert ImporterDaemonConfig().env() == {"NCD_SPOOL_LIMIT": "42"}
    assert not ImporterDaemonConfig().sets_daemon_launch_options


def test_apply_to_environment_sets_env(monkeypatch):
    # setenv first so monkeypatch restores the original value afterwards.
    monkeypatch.setenv("NCD_SPOOL_LIMIT", "0")
    monkeypatch.setattr(daemon_settings, "read_pid_from_file", lambda _: None)

    ImporterDaemonConfig(spool_limit=1024).apply_to_environment()

    assert daemon_settings.os.environ["NCD_SPOOL_LIMIT"] == "1024"


def test_apply_to_environment_warns_when_daemon_already_running(monkeypatch, caplog):
    monkeypatch.setenv("NCD_SPOOL_LIMIT", "1")
    monkeypatch.setenv("NCD_TRANSCODE_WORKERS", "1")
    monkeypatch.setattr(daemon_settings, "read_pid_from_file", lambda _: 123)
    monkeypatch.setattr(daemon_settings, "pid_is_running", lambda _: True)

    ImporterDaemonConfig(spool_limit=1024, transcode_workers=4).apply_to_environment()

    assert "already running (pid=123)" in caplog.text
