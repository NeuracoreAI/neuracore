#!/usr/bin/env bash
# Run the data-integrity suite and record per-case timing to a CSV.
#
# This is the timing/logging harness (not a pass/fail gate): it runs every
# data-integrity case and, via perf_logging_plugin.py, appends one CSV row per
# case capturing the daemon under test, the result (or failure summary), the
# case wall time, and the aggregate nc.log_* call timings (avg/max, call count,
# and per-label detail). Rows append across invocations, so running it twice —
# once with NCD_RUST_DAEMON=1 and once with NCD_RUST_DAEMON=0 — accumulates a
# rust-vs-python comparison in a single CSV.
#
# Logging defaults to RUST_LOG=warn / NDD_DEBUG=false so per-frame daemon
# logging does not skew the measured latencies; override either for diagnosis.
#
# Output (the CSV + a structured pytest --log-file) is written under
# .integration_test_logs by default (override the CSV with NDD_CSV_PATH or the
# directory with NDD_PERF_LOG_DIR). The per-case nc.log_* timings are stored
# inline in the CSV's `log_label_detail` column. The diagnostic Timer._samples
# percentiles are left untouched.
#
# Not run with --exitfirst: every case runs so one breach does not hide the
# rest of the timing data.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$workspace_root/.." && pwd)"

cd "$repo_root"

# ---------------------------------------------------------------------------
# Paths (mirror run_integration_tests.sh so the same host state is cleaned)
# ---------------------------------------------------------------------------

ndd_base_dir="/tmp/ndd"
iceoryx_dir="/tmp/iceoryx2"
default_pid_path="$HOME/.neuracore/daemon.pid"
default_state_dir="$HOME/.neuracore/data_daemon"
test_state_dir="$repo_root/.data_daemon_test_state"

log_dir="${NDD_PERF_LOG_DIR:-$repo_root/.integration_test_logs}"
mkdir -p "$log_dir"
run_stamp="$(date +%Y%m%d_%H%M%S)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Stable CSV so repeat runs accumulate; override with NDD_CSV_PATH.
csv_path="${NDD_CSV_PATH:-$log_dir/data_integrity_timings.csv}"
log_file="$log_dir/perf_logging_${run_stamp}.log"
pytest_log_file="$log_dir/perf_logging_pytest_${run_stamp}.log"

log() {
  printf '==> %s\n' "$*" | tee -a "$log_file"
}

purge_path() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    rm -rf -- "$target" 2>>"$log_file" || true
  fi
}

purge_glob() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  if (( ${#matches[@]} > 0 )); then
    rm -rf -- "${matches[@]}" 2>>"$log_file" || true
  fi
}

stop_daemon() {
  log "stopping any running data daemon"
  neuracore data-daemon stop >>"$log_file" 2>&1 || true
  pkill -TERM -f 'neuracore[.]data_daemon' 2>/dev/null || true
  pkill -TERM -x 'data-daemon' 2>/dev/null || true
  sleep 1
  pkill -KILL -f 'neuracore[.]data_daemon' 2>/dev/null || true
  pkill -KILL -x 'data-daemon' 2>/dev/null || true
}

# Clean the daemon's host state, but leave prior logs/CSV in place so timing
# results accumulate under $log_dir.
cleanup_state() {
  purge_path "$iceoryx_dir"
  purge_path "$ndd_base_dir"
  purge_glob '/dev/shm/neuracore-*'
  purge_glob '/dev/shm/iox2_*'
  purge_path "$default_pid_path"
  purge_path "${default_pid_path}.lock"
  purge_path "$default_state_dir"
  purge_path "$test_state_dir"
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

export NEURACORE_API_URL="https://staging.api.neuracore.com/api"
export NCD_RUST_DAEMON="${NCD_RUST_DAEMON:-1}"

export NEURACORE_CONSUME_LIVE_DATA=no
export NEURACORE_PROVIDE_LIVE_DATA=no

# Low-overhead logging so it does not skew the measured latencies; override for
# diagnosis.
export NDD_DEBUG="${NDD_DEBUG:-false}"
export RUST_LOG="${RUST_LOG:-warn}"
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1

# Daemon label for the CSV.
if [[ "$NCD_RUST_DAEMON" =~ ^(1|true|yes|on|TRUE|True)$ ]]; then
  daemon_label="rust"
else
  daemon_label="python"
fi

# Wire the CSV-recording plugin in non-invasively.
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p perf_logging_plugin"
export NDD_CSV_PATH="$csv_path"
export NDD_RUN_INDEX="$run_stamp"
export NDD_DAEMON="$daemon_label"
export NDD_STARTED_AT="$started_at"

data_integrity_targets=(
  "tests/integration/platform/data_daemon/data_integrity/test_pre_network.py"
  "tests/integration/platform/data_daemon/data_integrity/test_network.py"
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  log "==== data-integrity timing run starting ===="
  log "daemon=$daemon_label RUST_LOG=$RUST_LOG NDD_DEBUG=$NDD_DEBUG"
  log "csv: $csv_path"
  log "stdout log: $log_file"

  stop_daemon
  cleanup_state

  set +e
  pytest \
    --tb=short \
    -vv \
    -o log_cli=true \
    -o log_cli_level=INFO \
    --log-file="$pytest_log_file" \
    --log-file-level=DEBUG \
    "${data_integrity_targets[@]}" \
    2>&1 | tee -a "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  log "==== data-integrity timing run finished (exit=$exit_code) ===="
  if [[ -f "$csv_path" ]]; then
    log "rows for this run ($daemon_label):"
    python3 - "$csv_path" "$run_stamp" <<'PY' | tee -a "$log_file"
import csv, sys
path, run_index = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(path)) if r["run_index"] == run_index]
for r in rows:
    print(f"    {r['daemon']:6} {r['result']:7} {r['case_id'][:40]:40} "
          f"wall={r['wall_clock_s']:>8}s  log_avg={r['log_avg_ms'] or '-':>7}ms  "
          f"log_max={r['log_max_ms'] or '-':>8}ms")
print(f"    ({len(rows)} case(s); full CSV at {path})")
PY
  fi
  exit "$exit_code"
}

main "$@"
