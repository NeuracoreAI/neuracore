# cspell:: disable
"""Pytest plugin that records per-test-case timing to a CSV.

Used by ``run_performance_logging.sh`` to capture, for every data-integrity
case, the daemon under test, pass/fail (or the failure summary), the case wall
time, and the aggregate ``nc.log_*`` call timings (avg/max + per-label detail).

It is non-invasive: loaded via ``PYTEST_ADDOPTS="-p perf_logging_plugin"`` with
this directory on ``PYTHONPATH``, and writes nothing unless ``NDD_CSV_PATH`` is
set. A row is appended the moment each test finishes (after teardown, when that
case's timer stats have been recorded into the suite's ``SESSION_RUNS``).

Env:
  NDD_CSV_PATH     CSV file to append per-case rows to (enables recording)
  NDD_DAEMON       "rust" / "python" (recorded in each row)
  NDD_RUN_INDEX    run identifier recorded in each row
  NDD_STARTED_AT   ISO timestamp of the run
  NDD_CSV_JSON_DIR optional dir for a per-process JSON backup
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import time
from pathlib import Path

CSV_COLUMNS = [
    "run_index",
    "started_at",
    "daemon",
    "result",
    "issue_type",
    "case_id",
    "nodeid",
    "wall_clock_s",
    "log_avg_ms",
    "log_max_ms",
    "log_call_count",
    "log_label_detail",
    "pytest_duration_s",
]

# Timer labels representing a user-facing data-logging call; their avg/max are
# the "log timings" recorded per case.
LOG_LABEL_PREFIX = "nc.log_"

# nodeid -> aggregated outcome record for this pytest process.
_reports: dict[str, dict] = {}
# nodeids already written to the CSV (guard against double writes).
_written: set[str] = set()
# High-water mark into SESSION_RUNS so each test claims only its own entries.
_session_mark = 0
# Cached reference to the suite's SESSION_RUNS list (appended to in place).
_session_runs: list | None = None


def _get_session_runs() -> list:
    global _session_runs
    if _session_runs is None:
        try:
            from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (  # noqa: E501
                SESSION_RUNS,
            )

            _session_runs = SESSION_RUNS
        except Exception:  # noqa: BLE001
            _session_runs = []
    return _session_runs


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _param_id(nodeid: str) -> str | None:
    if "[" not in nodeid or not nodeid.endswith("]"):
        return None
    return nodeid[nodeid.rfind("[") + 1 : -1]


def _matches(case_param_id: str | None, case_id: str | None) -> bool:
    if not case_param_id or not case_id:
        return False
    if case_param_id == case_id:
        return True
    if case_id.endswith("-" + case_param_id) or case_id.endswith("/" + case_param_id):
        return True
    if case_param_id.startswith(case_id + "-"):
        return True
    return False


def _aggregate_log_stats(session_runs: list[dict]) -> dict:
    per_label: dict[str, dict[str, float]] = {}
    for run in session_runs:
        for label, stats in (run.get("timer_stats") or {}).items():
            if not label.startswith(LOG_LABEL_PREFIX):
                continue
            acc = per_label.setdefault(label, {"count": 0.0, "total": 0.0, "max": 0.0})
            acc["count"] += float(stats.get("count", 0.0))
            acc["total"] += float(stats.get("total", 0.0))
            acc["max"] = max(acc["max"], float(stats.get("max", 0.0)))

    total_count = sum(acc["count"] for acc in per_label.values())
    total_time = sum(acc["total"] for acc in per_label.values())
    overall_max = max((acc["max"] for acc in per_label.values()), default=0.0)

    detail = {
        label: {
            "n": int(acc["count"]),
            "avg_ms": (
                round((acc["total"] / acc["count"]) * 1000, 3) if acc["count"] else None
            ),
            "max_ms": round(acc["max"] * 1000, 3),
        }
        for label, acc in sorted(per_label.items())
    }
    return {
        "log_avg_ms": (
            round((total_time / total_count) * 1000, 3) if total_count else ""
        ),
        "log_max_ms": round(overall_max * 1000, 3) if total_count else "",
        "log_call_count": int(total_count) if total_count else 0,
        "log_label_detail": json.dumps(detail) if detail else "",
    }


def _build_row(report: dict, matched_runs: list[dict]) -> dict:
    test_walls = [
        run["test_wall_s"] for run in matched_runs if run.get("test_wall_s") is not None
    ]
    pytest_duration = float(report.get("duration_s", 0.0))
    wall_clock = max(test_walls) if test_walls else pytest_duration

    row = {
        "run_index": os.environ.get("NDD_RUN_INDEX", ""),
        "started_at": os.environ.get("NDD_STARTED_AT", ""),
        "daemon": os.environ.get("NDD_DAEMON", ""),
        "result": report.get("outcome", "unknown"),
        "issue_type": report.get("issue") or "",
        "case_id": _param_id(report["nodeid"]) or "",
        "nodeid": report["nodeid"],
        "wall_clock_s": round(wall_clock, 3) if wall_clock is not None else "",
        "pytest_duration_s": round(pytest_duration, 3),
    }
    row.update(_aggregate_log_stats(matched_runs))
    return row


def _append_row(csv_path: str, row: dict) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _crash_message(report) -> str | None:
    longrepr = getattr(report, "longrepr", None)
    if longrepr is None:
        return None
    crash = getattr(longrepr, "reprcrash", None)
    if crash is not None and getattr(crash, "message", None):
        first = str(crash.message).strip().splitlines()
        if first:
            return first[0][:500]
    text = str(longrepr).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1][:500] if lines else None


def pytest_runtest_logreport(report) -> None:
    record = _reports.setdefault(
        report.nodeid,
        {
            "nodeid": report.nodeid,
            "outcome": "passed",
            "duration_s": 0.0,
            "issue": None,
        },
    )
    record["duration_s"] += float(getattr(report, "duration", 0.0) or 0.0)
    if report.failed:
        record["outcome"] = (
            "error" if report.when in ("setup", "teardown") else "failed"
        )
        record["issue"] = _crash_message(report)
    elif report.skipped and record["outcome"] == "passed":
        record["outcome"] = "skipped"
        record["issue"] = _crash_message(report)


def pytest_runtest_logfinish(nodeid, location) -> None:
    global _session_mark
    csv_path = os.environ.get("NDD_CSV_PATH")
    if not csv_path or nodeid in _written:
        return
    report = _reports.get(nodeid)
    if report is None:
        return

    # SESSION_RUNS entries appended since the previous test belong to this one
    # (tests run serially within a process).
    session_runs = _get_session_runs()
    new_runs = session_runs[_session_mark:]
    _session_mark = len(session_runs)

    case_param = _param_id(nodeid)
    matched = [run for run in new_runs if _matches(case_param, run.get("case_id"))]
    if not matched and new_runs:
        matched = list(new_runs)

    try:
        _append_row(csv_path, _build_row(report, matched))
        _written.add(nodeid)
    except Exception as exc:  # noqa: BLE001 - never let reporting break the run
        print(f"[perf_logging_plugin] failed to write row for {nodeid}: {exc}")


def pytest_sessionfinish(session, exitstatus) -> None:
    out_dir = os.environ.get("NDD_CSV_JSON_DIR")
    if not out_dir:
        return
    payload = {
        "pid": os.getpid(),
        "exitstatus": int(exitstatus),
        "finished_at": time.time(),
        "reports": list(_reports.values()),
        "session_runs": list(_get_session_runs()),
    }
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / f"cases_{os.getpid()}_{int(time.time() * 1000)}.json").write_text(
        json.dumps(payload, default=str)
    )
