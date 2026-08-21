from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import neuracore as nc
from neuracore.data_daemon.const import active_profile_name
from tests.integration.platform.data_daemon.shared.assertions import (
    clear_daemon_timer_stats as _clear_daemon_timer_stats,
)
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import (
    Timer,
    reclaim_leftover_daemons,
)
from tests.integration.platform.data_daemon.shared.profiles import (
    cleanup_test_profiles,
    profile_path,
)
from tests.integration.platform.data_daemon.shared.reporting import (
    PerformanceReportContext,
    attach_json,
    attach_text,
    build_performance_metrics,
    format_event_timeline,
    format_metric_summary,
    format_phase_summary,
    load_performance_events,
    performance_metrics_enabled,
    report_headline_metrics,
    summarize_performance_events,
    write_metrics_artifact,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    SESSION_RUNS,
    DataDaemonTestCase,
    _format_timer_stats_line,
    has_configured_org,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    apply_storage_state_action,
    set_case_analysis_report,
)

# cspell:ignore terminalreporter exitstatus finalizer NODEIDS exitfirst unparameterized
# cspell:ignore nodeid getfixturevalue modifyitems callspec

# Add the repo root to the path so sub workers on macos can unpickle pool tasks
# correctly. pytest --import-mode=importlib imports tests files without touching the
# sys.path. So this is to compensate. Repo root at the front would shadow the installed
# neuracore wheel so the path is appended rather than prepended.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)


_BATCH_START_CLEANED_NODEIDS: set[str] = set()

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the optional directory for stable per-case metric artifacts."""
    parser.getgroup("data-daemon reporting").addoption(
        "--performance-metrics-dir",
        action="store",
        default=None,
        help="Write one performance metrics JSON file per data-daemon test case.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast when explicitly enabled phase capture cannot write events."""
    del config
    if not performance_metrics_enabled():
        return
    raw_path = os.environ.get("NCD_PERF_EVENTS_PATH")
    if not raw_path:
        raise pytest.UsageError(
            "NCD_PERF_METRICS is enabled but NCD_PERF_EVENTS_PATH is not set"
        )
    event_path = Path(raw_path)
    if not event_path.is_absolute():
        raise pytest.UsageError(
            "NCD_PERF_EVENTS_PATH must be absolute because the daemon runs in "
            "a background process"
        )
    try:
        event_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.touch(exist_ok=True)
    except OSError as error:
        raise pytest.UsageError(
            f"cannot write NCD_PERF_EVENTS_PATH {event_path}: {error}"
        ) from error


@pytest.fixture(autouse=True)
def login_parent_process() -> Iterator[None]:
    """Authenticate the parent pytest process for each test.

    Logs in during setup and logs out during teardown, so every test
    performs (and times) a real login against a clean environment.
    """
    ensure_login()
    yield
    # The dedicated local performance-report runner preserves the saved login
    # across cases. Normal integration runs still exercise and time a fresh
    # login per test exactly as before.
    if os.environ.get("NCD_PRESERVE_TEST_LOGIN") != "1":
        nc.logout()


@pytest.fixture(autouse=True)
def cleanup_profiles():
    """Remove test-created daemon profiles after each test."""
    yield
    cleanup_test_profiles()


@pytest.fixture(autouse=True)
def reset_video_codec():
    """Restore the active daemon profile to its exact pre-test state.

    The codec is a persistent daemon-profile setting, so a case selecting a
    lossy codec must not leak into a later test or the developer's real profile.
    Offline cases write an ephemeral profile (removed by ``cleanup_profiles``),
    but online cases write the default profile. Snapshot the active profile file
    and restore it byte-for-byte (or delete it if it did not exist), rather than
    forcing a literal ``h264_lossless`` — which would otherwise leave a residual
    ``video_codec`` key on a developer's default profile that had none. Only
    writes when the test actually changed the file, so non-video tests are free.
    """
    active_path = profile_path(active_profile_name())
    original_bytes = active_path.read_bytes() if active_path.exists() else None
    try:
        yield
    finally:
        current_bytes = active_path.read_bytes() if active_path.exists() else None
        if current_bytes == original_bytes:
            return
        if original_bytes is not None:
            active_path.write_bytes(original_bytes)
        elif active_path.exists():
            active_path.unlink()


@pytest.fixture(autouse=True)
def skip_marked_cases(request: pytest.FixtureRequest) -> None:
    """Skip any parametrized case flagged with ``skip=True``.

    Lets unstable or not-yet-validated workloads remain documented in the
    suite's test-case tables (rather than being commented out) while still
    being excluded from execution.
    """
    if "case" not in request.fixturenames:
        return
    case = request.getfixturevalue("case")
    if getattr(case, "skip", False):
        pytest.skip("case marked skip=True")


@pytest.fixture(autouse=True)
def apply_batch_start_storage_state(request: pytest.FixtureRequest) -> None:
    """Apply local storage cleanup once before the first case in each batch.

    A batch is defined as one parametrized test function over ``case``.  The
    fixture keys by the unparameterized node id, so cleanup runs once before
    the first case and is skipped for the remaining cases in that batch.
    """
    if "case" not in request.fixturenames:
        return

    nodeid_without_param = request.node.nodeid.split("[", 1)[0]
    if nodeid_without_param in _BATCH_START_CLEANED_NODEIDS:
        return

    request.getfixturevalue("case")
    apply_storage_state_action(STORAGE_STATE_DELETE)
    _BATCH_START_CLEANED_NODEIDS.add(nodeid_without_param)


@pytest.fixture(autouse=True)
def skip_cases_the_deployment_cannot_run(request: pytest.FixtureRequest) -> None:
    """Skip a case this deployment is not equipped to run.

    Properties of the case and the deployment, so every case-driven test gets
    the same checks from here.
    """
    if "case" not in request.fixturenames:
        return
    if not has_configured_org():
        pytest.skip(
            "Case-driven daemon tests require NEURACORE_ORG_ID"
            " or a saved current organization."
        )


@pytest.fixture()
def clear_daemon_timer_stats() -> None:
    """Clear daemon timer stats before each matrix-style test."""
    _clear_daemon_timer_stats()


@pytest.fixture()
def test_wall_timer() -> Callable[[], float]:
    """Return a callable that computes elapsed wall time since test start."""
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


@pytest.fixture(autouse=True)
def correlate_performance_events(request: pytest.FixtureRequest) -> Iterator[None]:
    """Correlate opt-in daemon events to every pytest case before launch."""
    if not performance_metrics_enabled():
        yield
        return
    previous_case_id = os.environ.get("NCD_PERF_CASE_ID")
    os.environ["NCD_PERF_CASE_ID"] = request.node.nodeid
    try:
        yield
    finally:
        if previous_case_id is None:
            os.environ.pop("NCD_PERF_CASE_ID", None)
        else:
            os.environ["NCD_PERF_CASE_ID"] = previous_case_id


@pytest.fixture(autouse=True)
def install_performance_reporter(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Make generic metrics emission available to every case-reporting test."""
    emitted = False

    def emit_report(
        *,
        case: DataDaemonTestCase,
        results: list[object],
        test_wall_s: float | None,
        analysis_report: str,
    ) -> None:
        nonlocal emitted
        if emitted:
            return
        try:
            request.node.run_analysis_report = analysis_report
            metrics = build_performance_metrics(
                case=case,
                results=results,
                timer_stats=Timer._stats,
                test_wall_s=test_wall_s,
            )
            capture_enabled = performance_metrics_enabled()
            events = (
                load_performance_events(
                    os.environ.get("NCD_PERF_EVENTS_PATH"),
                    case_id=request.node.nodeid,
                )
                if capture_enabled
                else []
            )
            event_summary = summarize_performance_events(events)
            event_summary["enabled"] = capture_enabled
            metrics["structured_phase_events"] = event_summary
            attach_text("Run timing analysis", analysis_report)
            attach_text("Performance metrics", format_metric_summary(metrics))
            if capture_enabled:
                attach_text("Daemon phase summary", format_phase_summary(event_summary))
                attach_text("Exact phase timeline", format_event_timeline(events))
                attach_json("Structured phase events (JSON)", events)
            attach_json("Performance metrics (JSON)", metrics)
            report_headline_metrics(metrics)
            metrics_dir = request.config.getoption("--performance-metrics-dir")
            if metrics_dir:
                write_metrics_artifact(
                    metrics_dir,
                    nodeid=request.node.nodeid,
                    metrics=metrics,
                )
            emitted = True
        except Exception:  # noqa: BLE001
            # Opt-in capture is a promised test output, so never silently lose
            # it in local reporting or staging CI. Preserve the historical
            # best-effort behavior when structured capture is disabled.
            if performance_metrics_enabled():
                raise

    attribute = "_data_daemon_performance_reporter"
    setattr(request.node, attribute, emit_report)
    try:
        yield
    finally:
        if getattr(request.node, attribute, None) is emit_report:
            delattr(request.node, attribute)


@pytest.fixture()
def performance_report(
    request: pytest.FixtureRequest,
) -> Callable[..., PerformanceReportContext]:
    """Create a context-managed report for any data-daemon test case."""

    def create(
        case: DataDaemonTestCase,
        *,
        dataset_name: str | None = None,
        label_prefix: str | None = None,
    ) -> PerformanceReportContext:
        def finalize(
            finalized_case: DataDaemonTestCase,
            results: list[object],
            test_wall_s: float,
        ) -> None:
            set_case_analysis_report(
                request=request,
                case=finalized_case,
                results=results,
                label_prefix=label_prefix,
                test_wall_s=test_wall_s,
            )

        return PerformanceReportContext(
            case=case,
            dataset_name=dataset_name,
            finalize=finalize,
        )

    return create


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Stop any daemon still running when the session ends."""
    del session, exitstatus
    try:
        reclaim_leftover_daemons()
    except Exception as error:
        logger.warning("could not stop leftover daemons at session end: %s", error)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    del exitstatus, config
    if not SESSION_RUNS:
        return

    timer_stats = Timer._stats
    separator = "=" * 64
    lines: list[str] = [
        "",
        separator,
        f"Session summary  ({len(SESSION_RUNS)} test(s) completed)",
        separator,
    ]

    all_labels = sorted({label for run in SESSION_RUNS for label in run["timer_stats"]})
    for run in SESSION_RUNS:
        dataset_suffix = (
            run.get("label_prefix") + "  " if run.get("label_prefix") else ""
        ) + (f"  dataset={run['dataset_name']!r}" if run.get("dataset_name") else "")
        ctx_parts = "  ".join(
            f"ctx[{c['context_index']}]={c['wall_s']:.1f}s"
            for c in sorted(run["context_results"], key=lambda c: c["context_index"])
        )
        test_wall_s = run.get("test_wall_s")
        if test_wall_s is not None:
            wall_info = (
                f"test_wall={test_wall_s:.1f}s  {ctx_parts}"
                if ctx_parts
                else f"test_wall={test_wall_s:.1f}s"
            )
        else:
            wall_info = ctx_parts or "wall=n/a"
        lines.append(f"\n  {run['case_id']}  ({wall_info}){dataset_suffix}")
        for label in all_labels:
            stats = run["timer_stats"].get(label)
            if stats is not None:
                lines.append(_format_timer_stats_line(label, stats))
            else:
                lines.append(f"    {label:<42}  ---")

    infra_labels = sorted(label for label in timer_stats if label not in all_labels)
    if infra_labels:
        lines.append("\n  Infrastructure timings:")
        for label in infra_labels:
            lines.append(_format_timer_stats_line(label, timer_stats[label]))

    lines.append(separator)
    terminalreporter.write_line("\n".join(lines))
