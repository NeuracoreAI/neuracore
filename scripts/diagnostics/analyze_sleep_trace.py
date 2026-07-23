#!/usr/bin/env python3
"""Correlate sleep anomaly records with exportable xctrace evidence.

The xctrace scheduler schema varies by Xcode release. This tool preserves the
raw candidate table exports and marks transition fields unavailable unless it
can prove them; it never converts total sleep overshoot into scheduler delay.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCHEMA_KEYWORDS = ("signpost", "thread", "sched", "state", "context-switch")


def run_export(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["xcrun", "xctrace", "export", *arguments],
        check=False,
        text=True,
        capture_output=True,
    )


def candidate_schemas(toc_path: Path) -> list[str]:
    try:
        root = ET.parse(toc_path).getroot()
    except (ET.ParseError, OSError):
        return []
    schemas = {
        value
        for element in root.iter()
        if (value := element.attrib.get("schema"))
        and any(keyword in value.lower() for keyword in SCHEMA_KEYWORDS)
    }
    return sorted(schemas)


def classify_native(record: dict[str, Any]) -> tuple[str, list[str]]:
    evidence: list[str] = []
    native_wait_ns = record.get("native_wait_ns")
    requested_ns = record.get("requested_ns")
    gil_ns = record.get("gil_reacquire_ns")
    post_ns = record.get("python_post_native_ns")
    overshoot_ns = record.get("overshoot_ns", 0)
    if native_wait_ns is not None and requested_ns is not None:
        native_overshoot = native_wait_ns - requested_ns
        evidence.append(f"native_wait_overshoot_ns={native_overshoot}")
        if native_overshoot <= 1_000_000 and (
            (gil_ns or 0) + (post_ns or 0) >= max(1_000_000, overshoot_ns // 2)
        ):
            evidence.append(f"gil_reacquire_ns={gil_ns}")
            evidence.append(f"python_post_native_ns={post_ns}")
            return "RESUMED_THEN_BLOCKED", evidence
    evidence.append(
        "No proven Waiting→Runnable and Runnable→Running scheduler transitions"
    )
    return "INSUFFICIENT_TRACE_DATA", evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--anomalies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    toc_path = args.output_dir / "trace-toc.xml"
    toc = run_export([str(args.trace), "--toc"])
    toc_path.write_text(toc.stdout, encoding="utf-8")
    (args.output_dir / "trace-toc.stderr.txt").write_text(toc.stderr, encoding="utf-8")

    # Keep CI analysis bounded even if a future Xcode template exposes many
    # related internal schemas. The untouched .trace remains the source of
    # truth for manual analysis.
    schemas = candidate_schemas(toc_path)[:8]
    exported: list[dict[str, Any]] = []
    for index, schema in enumerate(schemas):
        destination = args.output_dir / f"candidate-{index:02d}.xml"
        xpath = f"/trace-toc/run[@number='1']/data/table[@schema='{schema}']"
        result = run_export([
            str(args.trace),
            "--output",
            str(destination),
            "--xpath",
            xpath,
        ])
        exported.append({
            "schema": schema,
            "path": str(destination),
            "returncode": result.returncode,
            "stderr": result.stderr,
        })

    anomalies = []
    if args.anomalies.exists():
        anomalies = [
            json.loads(line)
            for line in args.anomalies.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    reports = []
    for record in anomalies:
        classification, evidence = classify_native(record)
        reports.append({
            "sleep_id": record.get("sleep_id"),
            "requested_duration_ns": record.get("requested_ns"),
            "actual_duration_ns": record.get("actual_ns"),
            "overshoot_ns": record.get("overshoot_ns"),
            "sleep_start": record.get("monotonic_start_ns"),
            "expected_deadline": record.get("expected_deadline_ns"),
            "observed_return": record.get("observed_return_ns"),
            "native_thread_id": record.get("native_thread_id"),
            "thread_waiting_start": None,
            "thread_waiting_end": None,
            "first_runnable_time": None,
            "first_running_time_after_deadline": None,
            "timer_delivery_lateness": None,
            "scheduler_queue_delay": None,
            "unexplained_post_run_delay": None,
            "state_at_deadline": "UNKNOWN",
            "eligible_cpu_activity": "UNAVAILABLE_FROM_AUTOMATIC_EXPORT",
            "classification": classification,
            "evidence": evidence,
        })

    report = {
        "trace": str(args.trace),
        "anomalies": str(args.anomalies),
        "candidate_exports": exported,
        "reports": reports,
        "limitations": [
            "xctrace does not expose a stable cross-version scheduler-table schema.",
            "Null transition fields are unavailable, not measured as zero.",
            "Open the preserved trace in Instruments and correlate "
            "sleep_id, PID, and native_thread_id.",
        ],
    }
    (args.output_dir / "sleep-diagnostic-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
