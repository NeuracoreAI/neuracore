import logging
from collections.abc import Callable, Generator

import pytest

import neuracore as nc

# cspell:ignore hookwrapper makereport terminalreporter pluginmanager
# cspell:ignore getplugin nodeid longreprtext

logger = logging.getLogger(__name__)


@pytest.fixture
def dataset_cleanup() -> Generator[Callable[[str], None], None, None]:
    """Register dataset names to be deleted after the test."""
    dataset_names: list[str] = []

    def register(dataset_name: str) -> None:
        dataset_names.append(dataset_name)

    yield register

    for dataset_name in dataset_names:
        try:
            nc.login()
            nc.get_dataset(dataset_name).delete()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to delete test dataset: %s", dataset_name)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"_report_{report.when}", report)
    if report.when != "teardown":
        return

    terminalreporter = item.config.pluginmanager.getplugin("terminalreporter")
    if terminalreporter is None:
        return

    setup_report = getattr(item, "_report_setup", None)
    call_report = getattr(item, "_report_call", None)
    teardown_report = getattr(item, "_report_teardown", None)
    reports = [
        report for report in (setup_report, call_report, teardown_report) if report
    ]
    if not reports:
        return

    if any(report.failed for report in reports):
        final_outcome = "FAILED"
    elif any(report.skipped for report in reports):
        final_outcome = "SKIPPED"
    else:
        final_outcome = "PASSED"

    terminalreporter.write_line("")
    terminalreporter.write_line(f"[{final_outcome}] {item.nodeid}", bold=True)

    failure_report = next(
        (
            candidate
            for candidate in (setup_report, call_report, teardown_report)
            if candidate is not None and candidate.failed
        ),
        None,
    )
    if failure_report is not None and getattr(failure_report, "longreprtext", ""):
        terminalreporter.write_line(failure_report.longreprtext)

    analysis_report = getattr(item, "run_analysis_report", None)
    if analysis_report:
        terminalreporter.write_line(analysis_report)
