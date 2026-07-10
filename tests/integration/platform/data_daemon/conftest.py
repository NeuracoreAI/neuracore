from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import neuracore as nc
from neuracore.data_daemon.config_manager.profiles import ProfileManager
from neuracore.data_daemon.const import active_profile_name
from neuracore.data_daemon.rust_selection import is_rust_daemon_enabled
from tests.integration.platform.data_daemon.shared.assertions import (
    clear_daemon_timer_stats as _clear_daemon_timer_stats,
)
from tests.integration.platform.data_daemon.shared.auth import ensure_login
from tests.integration.platform.data_daemon.shared.process_control import Timer
from tests.integration.platform.data_daemon.shared.profiles import cleanup_test_profiles
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    SESSION_RUNS,
    DataDaemonTestCase,
    _format_timer_stats_line,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case_context import (  # noqa: E501
    ContextResult,
)
from tests.integration.platform.data_daemon.shared.test_case.constants import (
    STORAGE_STATE_DELETE,
)
from tests.integration.platform.data_daemon.shared.test_infrastructure import (
    apply_storage_state_action,
    build_isolation_run_analysis,
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


@pytest.fixture(autouse=True)
def login_parent_process() -> Iterator[None]:
    """Authenticate the parent pytest process for each test.

    Logs in during setup and logs out during teardown, so every test
    performs (and times) a real login against a clean environment.
    """
    ensure_login()
    yield
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
    profile_path = ProfileManager()._get_profile_path(active_profile_name())
    original_bytes = profile_path.read_bytes() if profile_path.exists() else None
    try:
        yield
    finally:
        current_bytes = profile_path.read_bytes() if profile_path.exists() else None
        if current_bytes == original_bytes:
            return
        if original_bytes is not None:
            profile_path.write_bytes(original_bytes)
        elif profile_path.exists():
            profile_path.unlink()


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
def skip_rust_only_cases(request: pytest.FixtureRequest) -> None:
    """Skip cases flagged ``requires_rust_daemon`` on the legacy Python daemon.

    Gating them on the active daemon keeps them running under Rust while restoring the
    Python suite to its green baseline.
    """
    if "case" not in request.fixturenames:
        return
    case = request.getfixturevalue("case")
    if getattr(case, "requires_rust_daemon", False) and not is_rust_daemon_enabled():
        pytest.skip("case requires the Rust data daemon (NCD_RUST_DAEMON)")


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


@pytest.fixture()
def clear_daemon_timer_stats() -> None:
    """Clear daemon timer stats before each matrix-style test."""
    _clear_daemon_timer_stats()


@pytest.fixture()
def test_wall_timer() -> Callable[[], float]:
    """Return a callable that computes elapsed wall time since test start."""
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


@pytest.fixture()
def log_run_analysis_on_teardown(
    request: pytest.FixtureRequest,
) -> Callable[[DataDaemonTestCase, list[ContextResult], float | None], None]:
    """Register case+results to be passed to log_run_analysis at teardown."""
    state: dict[str, any] = {}

    def register(
        case: DataDaemonTestCase,
        results: list[ContextResult],
        test_wall_s: float | None = None,
    ) -> None:
        state["case"] = case
        state["results"] = results
        state["test_wall_s"] = test_wall_s

    def finalizer() -> None:
        if state.get("results"):
            try:
                request.node.run_analysis_report = build_isolation_run_analysis(
                    case=state["case"],
                    results=state["results"],
                    test_wall_s=state.get("test_wall_s"),
                )
            except Exception:  # noqa: BLE001
                pass

    request.addfinalizer(finalizer)
    return register


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
