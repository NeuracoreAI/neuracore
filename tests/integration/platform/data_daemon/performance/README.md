# Local network performance report

This report is intentionally local. It runs only
`performance/test_network.py`, captures correlated daemon phase events, and
renders them with Allure 3.

The reporting lifecycle itself is suite-wide. Data-daemon tests that register
case analysis through the shared infrastructure receive the same metrics and
Allure attachments. Tests that need explicit measured sections can use the
`performance_report` context-manager fixture from the shared pytest
configuration; it adds the standard case parameters, records end-to-end wall
time, retains results, and emits the report after the wrapped cleanup scopes
exit.

Build the instrumented daemon once after changing Rust code:

```bash
bash rust/scripts/build_wheel_artefacts.sh
```

Run the complete network-performance matrix:

```bash
bash rust/scripts/run_performance_network_report.sh
```

Structured daemon phase metrics are opt-in and this runner enables them by
default. Disable them while retaining the normal pytest/Allure report with:

```bash
NCD_PERF_METRICS=0 bash rust/scripts/run_performance_network_report.sh
```

Outside this runner, enable capture by setting both `NCD_PERF_METRICS=1` and an
absolute `NCD_PERF_EVENTS_PATH` before pytest launches the daemon.

Pass normal pytest selectors after the script to run one workload, for example:

```bash
bash rust/scripts/run_performance_network_report.sh \
  -k '10s-1recs-10joints-250hz'
```

The command prints the exact output directory. Open
`allure-report/index.html`, select a test case, and inspect these attachments:

- **Daemon phase summary**: count, summed time, average, maximum, failures,
  upload throughput, and retry/backoff totals.
- **Exact phase timeline**: ordered events relative to the first stop received
  by the daemon. Negative offsets occurred before stop; positive offsets show
  the post-stop readiness gap.
- **Structured phase events (JSON)**: full recording and trace correlation for
  detailed analysis.
- **Performance metrics (JSON)**: workload, producer/end-to-end throughput,
  pytest timers, and the structured phase aggregates in one stable document.

The run directory also retains `daemon-phase-events.jsonl`, `pytest.log`, the
daemon's `daemon.log`, JUnit XML, and one metrics JSON file per test case.

The staging data-daemon workflow enables capture on scheduled runs and uploads
`data-daemon-staging-<os>-py<version>-attempt<n>.html` directly from each matrix
job. The unarchived artifact contains only the self-contained report and is
linked from the job summary, so it can be downloaded and opened directly in a
browser. Manual workflow dispatches expose a **performance-metrics** checkbox
to turn capture off while still producing the standard Allure report.

Phase totals sum per-trace work. Traces execute concurrently, so those totals
can exceed wall time; use the exact timeline and maximum duration to find the
critical post-stop path.
