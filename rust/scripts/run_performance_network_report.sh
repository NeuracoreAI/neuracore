#!/usr/bin/env bash
# Run only the platform data-daemon network performance tests and build a
# self-contained local Allure report with per-case timing/throughput metrics.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
cd "$repo_root"

report_root="${NDD_PERFORMANCE_REPORT_DIR:-$repo_root/.data_daemon_test_reports/performance-network}"
run_stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
run_dir="$report_root/$run_stamp"
allure_results="$run_dir/allure-results"
allure_report="$run_dir/allure-report"
metrics_dir="$run_dir/metrics"
junit_path="$run_dir/junit.xml"
pytest_log="$run_dir/pytest.log"
events_path="$run_dir/daemon-phase-events.jsonl"
daemon_state_dir="$run_dir/daemon-state"
daemon_log="$daemon_state_dir/daemon.log"
test_target="tests/integration/platform/data_daemon/performance/test_cloud_performance.py"
python_bin="${PYTHON_BIN:-python3}"
perf_metrics="${NCD_PERF_METRICS:-1}"

perf_metrics_enabled() {
  case "$perf_metrics" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

if ! command -v pytest >/dev/null 2>&1; then
  echo "error: pytest is not installed in the active environment" >&2
  echo "install local test dependencies with: pip install -e '.[dev]'" >&2
  exit 2
fi

pytest_help="$(pytest --help 2>/dev/null || true)"
if [[ "$pytest_help" != *"--alluredir"* ]]; then
  echo "error: allure-pytest is not installed in pytest's active environment" >&2
  echo "install local test dependencies with: pip install -e '.[dev]'" >&2
  exit 2
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "error: Node.js/npx is required to build the visual Allure report" >&2
  echo "install Node.js 18 or newer and run this command again" >&2
  exit 2
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "error: Python executable not found: $python_bin" >&2
  exit 2
fi

if ! "$python_bin" -c '
import os
import sys
from neuracore.core.config.config_manager import get_config_manager

config = get_config_manager().config
api_key = os.environ.get("NEURACORE_API_KEY") or config.api_key
org_id = os.environ.get("NEURACORE_ORG_ID") or config.current_org_id
sys.exit(0 if api_key and org_id else 1)
'; then
  echo "error: a saved/API environment key and organization are both required" >&2
  echo "run login-staging, select an organization, then rerun this command" >&2
  exit 2
fi

if perf_metrics_enabled && ! "$python_bin" -c '
from neuracore.data_daemon.binary import data_daemon_binary_path

binary = data_daemon_binary_path()
if binary is None:
    raise SystemExit(1)
raise SystemExit(0 if b"NCD_PERF_EVENTS_PATH" in binary.read_bytes() else 1)
'; then
  echo "error: the bundled Rust daemon does not contain v2 phase instrumentation" >&2
  echo "build it once with: bash rust/scripts/build_wheel_artefacts.sh" >&2
  exit 2
fi

mkdir -p "$allure_results" "$metrics_dir" "$daemon_state_dir"

export NCD_PRESERVE_TEST_LOGIN=1
export NCD_PERF_METRICS="$perf_metrics"
export NCD_PERF_EVENTS_PATH="$events_path"
export NEURACORE_DAEMON_DB_PATH="$daemon_state_dir/state.db"
export NEURACORE_DAEMON_RECORDINGS_ROOT="$daemon_state_dir/recordings"
export NCD_PATH_TO_STORE_RECORD="$daemon_state_dir/recordings"
export RUST_LOG="${RUST_LOG:-info}"
export PYTHONUNBUFFERED=1

echo "Running network performance tests only"
echo "  target:  $test_target"
echo "  output:  $run_dir"
echo "  phase metrics: NCD_PERF_METRICS=$NCD_PERF_METRICS"
if perf_metrics_enabled; then
  echo "  events:  $events_path"
else
  echo "  events:  disabled"
fi

set +e
pytest \
  -o log_cli=true \
  --log-cli-level=INFO \
  --log-file="$pytest_log" \
  --log-file-level=INFO \
  --import-mode=importlib \
  --durations=0 \
  --durations-min=0.05 \
  --junitxml="$junit_path" \
  --alluredir="$allure_results" \
  --clean-alluredir \
  --performance-metrics-dir="$metrics_dir" \
  "$test_target" \
  "$@"
pytest_exit=$?
set -e

report_exit=0
if compgen -G "$allure_results/*" >/dev/null; then
  echo "Generating self-contained Allure 3 report"
  npx --yes allure@3.14.3 awesome \
    --single-file \
    --output "$allure_report" \
    "$allure_results" || report_exit=$?
else
  echo "warning: pytest produced no Allure results; no visual report generated" >&2
  report_exit=1
fi

echo
echo "Local performance report outputs"
echo "  visual report:  $allure_report/index.html"
echo "  case metrics:   $metrics_dir"
echo "  JUnit results:  $junit_path"
echo "  pytest log:     $pytest_log"
if perf_metrics_enabled; then
  echo "  phase events:   $events_path"
else
  echo "  phase events:   disabled"
fi
echo "  daemon log:     $daemon_log"

if (( pytest_exit != 0 )); then
  exit "$pytest_exit"
fi
exit "$report_exit"
