"""MCAP import throughput on a local MCAP dataset, against main's importer.

Report-only. Each case writes one JSON row to RESULTS_PATH.

Env:
    MCAP_PERF_DATASET_DIR: folder of .mcap files, one episode each. Required.
    MCAP_PERF_DATASET_CONFIG: importer config for the dataset. Required.
    MCAP_PERF_ROBOT_DIR: robot description folder (.urdf/.xml). Optional.
    MCAP_PERF_CASES: comma list of cases to run (default: all).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from neuracore_types import Codec

import neuracore as nc
from neuracore.data_daemon.helpers import get_daemon_recordings_root_path
from neuracore.importer import importer as importer_module
from neuracore.importer.core.daemon_settings import ImporterDaemonConfig
from neuracore.importer.core.png_frame import PNGFrame
from neuracore.importer.importer import _run_import
from neuracore.importer.mcap.mcap_importer import MCAPDatasetImporter
from tests.integration.importer.cli_option_tests import _cleanup_importer_resources
from tests.integration.importer.utils import skip_without_configured_org
from tests.integration.platform.data_daemon.shared.db_constants import RECORDINGS_TABLE
from tests.integration.platform.data_daemon.shared.db_helpers import (
    fetch_all_rows,
    latching_trace_write_observer,
    latching_upload_observer,
)
from tests.integration.platform.data_daemon.shared.process_control import (
    delete_recordings_folder,
)
from tests.integration.platform.data_daemon.shared.runners import (
    online_daemon_running,
    scoped_daemon_storage_env,
    stop_daemon_and_verify,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    OFFLINE_DB_PATH,
)

RESULTS_PATH = (
    Path(__file__).resolve().parents[3]
    / ".integration_test_logs"
    / "mcap_import_throughput.jsonl"
)
TIMEOUT_S = 3600.0
LEGACY_STORAGE_LIMIT_BYTES = 5 * 1024**3
LEGACY_DISK_CHECK_INTERVAL_S = 10.0
_DAEMON_ENV_NAMES = ("NCD_SPOOL_LIMIT", "NCD_VIDEO_CODEC")


class LegacyImporter(MCAPDatasetImporter):
    """Emulates the importer on main as a reference point.

    Decodes every frame, waits for each upload, pauses at item boundaries while
    the recordings root exceeds 5 GiB, and fails the episode on a stall.
    """

    def _log_data(self, data_type, source_data, *args, **kwargs) -> None:
        if isinstance(source_data, PNGFrame):
            source_data = source_data.decode()
        super()._log_data(data_type, source_data, *args, **kwargs)

    def _stop_recording(self, timestamp: float) -> None:
        nc.stop_recording(
            robot_name=self.robot_name,
            instance=self.robot_instance(self._worker_id),
            wait=True,
            timestamp=timestamp,
        )

    def _step(self, *args, **kwargs) -> None:
        while _directory_size(get_daemon_recordings_root_path()) >= (
            LEGACY_STORAGE_LIMIT_BYTES
        ):
            time.sleep(LEGACY_DISK_CHECK_INTERVAL_S)
        super()._step(*args, **kwargs)


@dataclass(frozen=True)
class ThroughputCase:
    """One importer setup to measure."""

    name: str
    importer_class: type[MCAPDatasetImporter]
    workers: int
    spool_limit: int | None
    video_codec: Codec


CASES = (
    # Main's defaults: one worker, 2 GiB daemon spool, h264_medium.
    ThroughputCase("legacy", LegacyImporter, 1, 2 * 1024**3, Codec.H264_MEDIUM),
    ThroughputCase("importer", MCAPDatasetImporter, 5, None, Codec.H264_FAST),
)


def _directory_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _selected_cases() -> list[ThroughputCase]:
    names = os.environ.get("MCAP_PERF_CASES")
    if not names:
        return list(CASES)
    wanted = {name.strip() for name in names.split(",")}
    unknown = wanted - {case.name for case in CASES}
    assert not unknown, f"Unknown MCAP_PERF_CASES: {sorted(unknown)}"
    return [case for case in CASES if case.name in wanted]


@dataclass(frozen=True)
class ThroughputDataset:
    """The local MCAP dataset to import."""

    dataset_dir: Path
    config_path: Path
    robot_dir: Path | None


def _dataset() -> ThroughputDataset:
    dataset_dir = os.environ.get("MCAP_PERF_DATASET_DIR")
    config_path = os.environ.get("MCAP_PERF_DATASET_CONFIG")
    if not dataset_dir or not config_path:
        pytest.skip("Set MCAP_PERF_DATASET_DIR and MCAP_PERF_DATASET_CONFIG.")
    robot_dir = os.environ.get("MCAP_PERF_ROBOT_DIR")
    return ThroughputDataset(
        dataset_dir=Path(dataset_dir),
        config_path=Path(config_path),
        robot_dir=Path(robot_dir) if robot_dir else None,
    )


@contextmanager
def _fresh_daemon(case: ThroughputCase) -> Iterator[int]:
    """Run a fresh online daemon on empty local storage with the case settings.

    Yields the resolved spool limit in bytes.
    """
    previous = {name: os.environ.get(name) for name in _DAEMON_ENV_NAMES}
    env = ImporterDaemonConfig(
        spool_limit=case.spool_limit, video_codec=case.video_codec
    ).env()
    os.environ.update(env)
    try:
        with scoped_daemon_storage_env():
            stop_daemon_and_verify()
            delete_recordings_folder()
            OFFLINE_DB_PATH.unlink(missing_ok=True)
            with online_daemon_running():
                yield int(env["NCD_SPOOL_LIMIT"])
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _timeline(expected_recordings: int) -> Iterator[Callable[[], dict[str, float]]]:
    """Time setup, then first recording to all data on disk and in the cloud."""
    start = time.monotonic()
    marks: dict[str, float] = {}
    stop = threading.Event()
    with (
        latching_trace_write_observer() as written,
        latching_upload_observer() as uploaded,
    ):

        def _watch() -> None:
            while not stop.is_set():
                now = time.monotonic()
                if "first_recording" not in marks:
                    try:
                        if fetch_all_rows(RECORDINGS_TABLE):
                            marks["first_recording"] = now
                    except sqlite3.OperationalError:
                        pass
                keys = written.written_keys()
                if "disk" not in marks and len(keys) >= expected_recordings:
                    marks["disk"] = now
                if "disk" in marks and all(uploaded.is_complete(k) for k in keys):
                    marks["cloud"] = now
                    return
                stop.wait(0.2)

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

        def elapsed() -> dict[str, float]:
            watcher.join(timeout=TIMEOUT_S)
            assert "cloud" in marks, (
                f"{len(written.written_keys())}/{expected_recordings} recordings "
                f"on disk; not all uploaded within {TIMEOUT_S:.0f}s."
            )
            first = marks.get("first_recording", start)
            return {
                "setup_s": first - start,
                "recording_to_disk_s": marks["disk"] - first,
                "recording_to_cloud_s": marks["cloud"] - first,
            }

        try:
            yield elapsed
        finally:
            stop.set()


def _run_case(
    case: ThroughputCase,
    dataset: ThroughputDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    dataset_dir = dataset.dataset_dir
    episode_count = len(list(dataset_dir.glob("*.mcap")))
    config = yaml.safe_load(dataset.config_path.read_text(encoding="utf-8"))
    run_id = uuid.uuid4().hex[:8]
    dataset_name = f"{config['output_dataset']['name']}_{case.name}_{run_id}"
    robot_name = f"{config['robot']['name']}_{run_id}"
    config["output_dataset"]["name"] = dataset_name
    config["robot"]["name"] = robot_name
    config_path = tmp_path / f"{case.name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(importer_module, "MCAPDatasetImporter", case.importer_class)
    resume_files_before = set(dataset_dir.glob(".neuracore_import_completed_*"))
    try:
        with (
            _fresh_daemon(case) as spool_bytes,
            _timeline(episode_count) as elapsed,
        ):
            start = time.monotonic()
            _run_import(
                dataset_config=config_path,
                dataset_dir=dataset_dir,
                robot_dir=dataset.robot_dir,
                max_workers=case.workers,
                suppress_validation_warnings=True,
                daemon_config=ImporterDaemonConfig(spool_limit=case.spool_limit),
            )
            import_wall_s = time.monotonic() - start
            return {
                "spool_limit_bytes": spool_bytes,
                "recordings": episode_count,
                **elapsed(),
                "import_wall_s": import_wall_s,
            }
    finally:
        for resume_file in (
            set(dataset_dir.glob(".neuracore_import_completed_*")) - resume_files_before
        ):
            resume_file.unlink(missing_ok=True)
        _cleanup_importer_resources(dataset_name=dataset_name, robot_name=robot_name)


_RESULTS: dict[str, dict] = {}


@pytest.mark.parametrize("case", _selected_cases(), ids=lambda case: case.name)
def test_mcap_import_throughput(
    case: ThroughputCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skip_without_configured_org()
    dataset = _dataset()
    source_bytes = sum(
        path.stat().st_size for path in dataset.dataset_dir.glob("*.mcap")
    )

    result = _run_case(case, dataset, tmp_path, monkeypatch)
    _RESULTS[case.name] = result

    row = {
        "case": case.name,
        "workers": case.workers,
        "video_codec": case.video_codec.value,
        "transcode_workers": os.environ.get("NCD_TRANSCODE_WORKERS"),
        "upload_workers": os.environ.get("NCD_UPLOAD_WORKERS"),
        "source_mib_per_s_to_disk": (
            source_bytes / 1024**2 / result["recording_to_disk_s"]
        ),
        **result,
    }
    legacy = _RESULTS.get("legacy")
    if legacy is not None and case.name != "legacy":
        for stage in ("disk", "cloud"):
            key = f"recording_to_{stage}_s"
            row[f"{stage}_speedup_vs_legacy"] = legacy[key] / result[key]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
