"""Saving the URDF packages of robots connected offline for the data daemon."""

from pathlib import Path

import pytest

from neuracore.data_daemon.offline_robots import (
    OFFLINE_ROBOTS_DIR,
    PACKAGE_FILENAME,
    offline_robot_dir,
    save_offline_robot,
)


@pytest.fixture(autouse=True)
def recordings_root(monkeypatch, tmp_path):
    root = tmp_path / "rec"
    monkeypatch.setenv("NEURACORE_DAEMON_RECORDINGS_ROOT", str(root))
    return root


def test_robots_are_saved_in_a_hidden_directory_under_the_recordings_root(
    recordings_root,
):
    assert offline_robot_dir("local-robot-1") == (
        recordings_root / OFFLINE_ROBOTS_DIR / "local-robot-1"
    )
    assert OFFLINE_ROBOTS_DIR.startswith(".")


def test_saving_writes_the_package_and_no_temp_files():
    path = save_offline_robot("local-robot-1", b"zip bytes")

    assert path == offline_robot_dir("local-robot-1") / PACKAGE_FILENAME
    assert [p.name for p in path.parent.iterdir()] == [PACKAGE_FILENAME]
    assert path.read_bytes() == b"zip bytes"


def test_saving_a_new_package_replaces_the_old_one():
    for package in (b"first", b"second"):
        path = save_offline_robot("local-robot-1", package)

    assert path.read_bytes() == b"second"


def test_saving_survives_the_daemon_removing_the_directory_meanwhile(monkeypatch):
    # The daemon removes a robot's directory once it has sent the package,
    # which can happen just after a new save created it.
    directory = offline_robot_dir("local-robot-1")
    original_mkdir = Path.mkdir
    removed = []

    def mkdir_then_removed_once(self, *args, **kwargs):
        original_mkdir(self, *args, **kwargs)
        if self == directory and not removed:
            removed.append(self)
            self.rmdir()

    monkeypatch.setattr(Path, "mkdir", mkdir_then_removed_once)

    path = save_offline_robot("local-robot-1", b"zip bytes")

    assert removed == [directory]
    assert path.read_bytes() == b"zip bytes"
