import json
from types import SimpleNamespace

import pytest

from tests.integration.platform.data_daemon.shared import test_infrastructure
from tests.integration.platform.data_daemon.shared.reporting import (
    PerformanceReport,
    PerformanceReportContext,
    build_performance_metrics,
    format_event_timeline,
    format_metric_summary,
    format_phase_summary,
    load_performance_events,
    performance_metrics_enabled,
    record_performance_event,
    summarize_performance_events,
    write_metrics_artifact,
)
from tests.integration.platform.data_daemon.shared.test_case.build_test_case import (
    DataDaemonTestCase,
)

# cspell:ignore nodeid


def test_build_performance_metrics_distinguishes_producer_and_end_to_end_rates():
    case = DataDaemonTestCase(
        duration_sec=2,
        recording_count=2,
        parallel_contexts=2,
        joint_fps=10,
        video_fps=5,
        video_count=1,
        image_width=4,
        image_height=3,
    )
    results = [
        SimpleNamespace(
            wall_started_at=10.0,
            wall_stopped_at=15.0,
            joint_frame_count=20,
            video_frame_count=10,
            recording_ids=["rec-1"],
            camera_names=["cam-1"],
        ),
        SimpleNamespace(
            wall_started_at=11.0,
            wall_stopped_at=16.0,
            joint_frame_count=20,
            video_frame_count=10,
            recording_ids=["rec-2"],
            camera_names=["cam-1"],
        ),
    ]
    timer_stats = {
        "nc.log_joint_positions": {"count": 20.0, "total": 1.0, "max": 0.1},
        "nc.log_rgb": {"count": 10.0, "total": 2.0, "max": 0.3},
        "performance.dataset_ready_wait": {
            "count": 1.0,
            "total": 2.5,
            "max": 2.5,
        },
    }

    metrics = build_performance_metrics(
        case=case,
        results=results,
        timer_stats=timer_stats,
        test_wall_s=10.0,
    )

    assert metrics["workload"] == {
        "recordings": 2,
        "parallel_contexts": 2,
        "joint_frame_sets": 40,
        "video_frames": 20,
        "logging_api_calls": 30,
        "raw_video_mib": pytest.approx(720 / (1024 * 1024), abs=0.001),
    }
    assert metrics["wall_times"] == {
        "producer_span_s": 6.0,
        "end_to_end_wall_s": 10.0,
    }
    assert metrics["producer_throughput"]["logging_api_calls_per_s"] == 5.0
    assert metrics["end_to_end_throughput"]["logging_api_calls_per_s"] == 3.0
    assert metrics["phase_timings"]["performance.dataset_ready_wait"] == {
        "count": 1,
        "total_s": 2.5,
        "average_s": 2.5,
        "maximum_s": 2.5,
    }


def test_metric_summary_labels_raw_video_as_submitted_data():
    case = DataDaemonTestCase(recording_count=1)
    metrics = build_performance_metrics(
        case=case,
        results=[
            SimpleNamespace(
                wall_started_at=1.0,
                wall_stopped_at=2.0,
                joint_frame_count=case.expected_joint_frames,
                video_frame_count=0,
                recording_ids=["rec-1"],
                camera_names=[],
            )
        ],
        timer_stats={},
        test_wall_s=2.0,
    )

    summary = format_metric_summary(metrics)

    assert "Raw RGB submitted" in summary
    assert "Producer-span throughput" in summary
    assert "End-to-end throughput" in summary


def test_write_metrics_artifact_uses_a_safe_discoverable_filename(tmp_path):
    metrics = {"wall_times": {"end_to_end_wall_s": 2.0}}

    path = write_metrics_artifact(
        tmp_path,
        nodeid="tests/performance/test_network.py::test_case[video/10s]",
        metrics=metrics,
    )

    assert path.parent == tmp_path
    assert "/" not in path.name
    assert json.loads(path.read_text()) == metrics


def test_performance_report_facade_is_safe_with_or_without_allure():
    report = PerformanceReport()

    report.parameter("throughput", "12.5 calls/s")
    with report.step("Measured phase"):
        report.attach_text("Phase summary", "duration=0.1s")
        report.attach_json("Phase data", {"duration_s": 0.1})


def test_performance_report_context_captures_results_and_finalizes():
    case = DataDaemonTestCase(recording_count=1)
    results = [SimpleNamespace(recording_ids=["rec-1"])]
    finalized = []

    def finalize(finalized_case, finalized_results, elapsed_s):
        finalized.append((finalized_case, finalized_results, elapsed_s))

    with PerformanceReportContext(
        case=case,
        dataset_name="dataset-name",
        finalize=finalize,
    ) as report:
        assert report.capture_results(results) is results

    assert len(finalized) == 1
    assert finalized[0][0] is case
    assert finalized[0][1] is results
    assert finalized[0][2] >= 0.0


def test_case_analysis_uses_installed_generic_performance_reporter(monkeypatch):
    node = SimpleNamespace()
    request = SimpleNamespace(node=node)
    emitted = []
    node._data_daemon_performance_reporter = lambda **values: emitted.append(values)
    monkeypatch.setattr(
        test_infrastructure,
        "build_isolation_run_analysis",
        lambda **values: "analysis report",
    )
    case = DataDaemonTestCase(recording_count=1)

    test_infrastructure.set_case_analysis_report(
        request=request,
        case=case,
        results=[],
        test_wall_s=2.5,
    )

    assert node.run_analysis_report == "analysis report"
    assert emitted == [{
        "case": case,
        "results": [],
        "test_wall_s": 2.5,
        "analysis_report": "analysis report",
    }]


def test_structured_events_are_filtered_aggregated_and_rendered(tmp_path):
    path = tmp_path / "events.jsonl"
    records = [
        {
            "timestamp_unix_ns": 2_000_000_000,
            "case_id": "wanted",
            "component": "daemon",
            "phase": "upload",
            "event": "completed",
            "elapsed_ms": 500.0,
            "recording_index": 7,
            "trace_id": "trace-123456789",
            "details": {"bytes_uploaded": 1024 * 1024},
        },
        {
            "timestamp_unix_ns": 1_000_000_000,
            "case_id": "wanted",
            "component": "daemon",
            "phase": "retry_backoff",
            "event": "scheduled",
            "trace_id": "trace-123456789",
            "details": {
                "attempt": 2,
                "reason": "http_503",
                "backoff_ms": 2000.0,
            },
        },
        {
            "timestamp_unix_ns": 3_000_000_000,
            "case_id": "other",
            "component": "daemon",
            "phase": "upload",
            "event": "completed",
            "elapsed_ms": 1.0,
            "details": {"bytes_uploaded": 99},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    events = load_performance_events(path, case_id="wanted")
    summary = summarize_performance_events(events)

    assert [event["phase"] for event in events] == ["retry_backoff", "upload"]
    assert summary["phases"]["upload"]["total_ms"] == 500.0
    assert summary["upload"]["mib_per_s"] == 2.0
    assert summary["retries"] == {
        "count": 1,
        "scheduled_backoff_ms": 2000.0,
    }
    assert "reason=http_503" in format_event_timeline(events)
    assert "upload" in format_phase_summary(summary)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_performance_metrics_truthy_flag(monkeypatch, value):
    monkeypatch.setenv("NCD_PERF_METRICS", value)

    assert performance_metrics_enabled()


def test_performance_event_capture_is_explicitly_opt_in(monkeypatch, tmp_path):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("NCD_PERF_EVENTS_PATH", str(path))
    monkeypatch.setenv("NCD_PERF_CASE_ID", "case-1")
    monkeypatch.delenv("NCD_PERF_METRICS", raising=False)

    record_performance_event("upload", "completed", elapsed_s=0.1)
    assert not path.exists()

    monkeypatch.setenv("NCD_PERF_METRICS", "1")
    record_performance_event(
        "upload",
        "completed",
        elapsed_s=0.1,
        details={"bytes_uploaded": 42},
    )

    event = json.loads(path.read_text())
    assert event["case_id"] == "case-1"
    assert event["phase"] == "upload"
    assert event["elapsed_ms"] == pytest.approx(100.0)
