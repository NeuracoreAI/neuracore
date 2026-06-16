#!/usr/bin/env bash
# Run the data-daemon integration tests against staging with the Rust daemon
# enabled (NCD_RUST_DAEMON=1).
#
# The script cleans every piece of host state the daemon touches before
# starting — a previous SIGKILL'd run can otherwise leave behind iceoryx2
# nodes, /dev/shm slot segments, pid/socket files, and stale recordings that
# make a fresh run fail in confusing ways.
#
# Tests are ordered from fastest to slowest so a failure earlier in the
# pipeline fails fast instead of waiting for the long-running performance
# suites. With --exitfirst pytest stops at the first failure.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$workspace_root/.." && pwd)"

cd "$repo_root"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Daemon ipc/socket artefacts (BASE_DIR in neuracore/data_daemon/const.py).
ndd_base_dir="/tmp/ndd"
# iceoryx2 dead-node registry and per-service files (see
# rust/data_daemon/src/lifecycle/recovery.rs).
iceoryx_dir="/tmp/iceoryx2"
# Default daemon pid file (helpers.get_daemon_pid_path / config::env::pid_path).
default_pid_path="$HOME/.neuracore/daemon.pid"
# Default daemon state (logs / DB / recordings live here when no env override).
default_state_dir="$HOME/.neuracore/data_daemon"
# Test-local daemon state — the conftest pins
# NEURACORE_DAEMON_RECORDINGS_ROOT / NEURACORE_DAEMON_DB_PATH at this dir, so
# log files, state.db and recordings end up under it during a test run.
test_state_dir="$repo_root/.data_daemon_test_state"

log_dir="$repo_root/.integration_test_logs"
mkdir -p "$log_dir"
run_stamp="$(date +%Y%m%d_%H%M%S)"
# stdout/stderr (pytest progress + script output)
log_file="$log_dir/run_${run_stamp}.log"
# pytest structured --log-file (DEBUG records). Kept separate from $log_file
# because pytest opens its --log-file in truncate mode, racing the tee that
# appends script output otherwise.
pytest_log_file="$log_dir/pytest_${run_stamp}.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
  # Mirror to stdout *and* the log file so a tail -f works while pytest runs.
  printf '==> %s\n' "$*" | tee -a "$log_file"
}

# Remove a path tree if it exists; never fail the script on cleanup races.
purge_path() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    rm -rf -- "$target" 2>>"$log_file" || true
  fi
}

# Remove files matching a glob; expand inside the function so the glob is
# evaluated even when no matches exist (nullglob).
purge_glob() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  if (( ${#matches[@]} > 0 )); then
    rm -rf -- "${matches[@]}" 2>>"$log_file" || true
  fi
}

# ---------------------------------------------------------------------------
# Stop any running daemon
# ---------------------------------------------------------------------------

stop_daemon() {
  log "stopping any running data daemon"

  # Try the CLI first; falls through silently if no daemon is up.
  neuracore data-daemon stop >>"$log_file" 2>&1 || true

  # Belt-and-braces: kill anything the CLI missed by name. The Rust binary is
  # 'data-daemon'; the Python entry point is '-m neuracore.data_daemon'.
  pkill -TERM -f 'neuracore[.]data_daemon' 2>/dev/null || true
  pkill -TERM -x 'data-daemon' 2>/dev/null || true
  sleep 1
  pkill -KILL -f 'neuracore[.]data_daemon' 2>/dev/null || true
  pkill -KILL -x 'data-daemon' 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

cleanup_state() {
  log "removing iceoryx2 dead-node/service files in $iceoryx_dir"
  purge_path "$iceoryx_dir"

  log "removing daemon ipc dir $ndd_base_dir"
  purge_path "$ndd_base_dir"

  log "removing neuracore shared-memory segments from /dev/shm"
  # _NEURACORE_SHARED_SLOT_PREFIX in
  # neuracore/data_daemon/lifecycle/runtime_recovery.py.
  purge_glob '/dev/shm/neuracore-*'
  # iceoryx2 also stores its shared memory under /dev/shm.
  purge_glob '/dev/shm/iox2_*'

  log "removing default daemon pid/lock at $default_pid_path"
  purge_path "$default_pid_path"
  purge_path "${default_pid_path}.lock"

  log "removing default daemon state dir $default_state_dir"
  purge_path "$default_state_dir"

  log "removing test-local daemon state dir $test_state_dir"
  purge_path "$test_state_dir"

  log "removing previous integration-test logs (keeping current $log_file)"
  shopt -s nullglob
  for existing in "$log_dir"/*.log; do
    if [[ "$existing" != "$log_file" ]]; then
      rm -f -- "$existing" || true
    fi
  done
  shopt -u nullglob
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Staging API endpoint and Rust-daemon selection per the task brief.
export NEURACORE_API_URL="https://staging.api.neuracore.com/api"
# Default to the Rust daemon, but honour a caller-provided value so the same
# script can also exercise the Python daemon (NCD_RUST_DAEMON=0) on repeat runs.
export NCD_RUST_DAEMON="${NCD_RUST_DAEMON:-1}"

# Quiet the SSE consumer + WebRTC producer loops for the duration of the
# integration suite.
export NEURACORE_CONSUME_LIVE_DATA=no
export NEURACORE_PROVIDE_LIVE_DATA=no

# Highest level of logging across the Python and Rust surfaces.
# Honour a caller-provided value so a harness can disable debug logging.
export NDD_DEBUG="${NDD_DEBUG:-true}"
export PYTHONUNBUFFERED=1
# Rust tracing — see https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html
export RUST_LOG="${RUST_LOG:-trace}"
export RUST_BACKTRACE=1

# ---------------------------------------------------------------------------
# Test ordering (fastest to slowest)
#
# Each entry is a pytest target relative to repo root. The order is hand-rolled
# so a failure surfaces in the cheapest suite first:
#   1. behavioural_correctness/test_startup        — single startup smoke
#   2. behavioural_correctness/test_signal_cleanup — many short signal tests
#   3. behavioural_correctness/test_cancel_recording — short, no upload
#   4. behavioural_correctness/test_offline_to_online — recorded -> online flip
#   5. data_integrity/test_pre_network             — disk-only integrity
#   6. data_integrity/test_network                 — adds upload + dataset wait
#   7. performance/test_pre_network                — disk-only perf budgets
#   8. performance/test_network                    — full network perf run
#
# The 300s (5-minute) 1920x1080 performance cases are pulled OUT of the normal
# order and run last (see `run_tests`): they dominate wall time and their
# stochastic-timing / upload-readiness budgets are the most fragile under host
# load, so every cheaper test gets a chance to fail fast first. The 300s
# pre-network (disk-only) cases run immediately before their network (upload)
# equivalents. Both perf files keep their case-ids containing "300s", which is
# the keyword the two phases select on.
# ---------------------------------------------------------------------------

test_targets=(
  "tests/integration/platform/data_daemon/behavioural_correctness/test_startup.py"
  "tests/integration/platform/data_daemon/behavioural_correctness/test_signal_cleanup.py"
  "tests/integration/platform/data_daemon/behavioural_correctness/test_cancel_recording.py"
  "tests/integration/platform/data_daemon/behavioural_correctness/test_offline_to_online.py"
  "tests/integration/platform/data_daemon/data_integrity/test_pre_network.py"
  "tests/integration/platform/data_daemon/data_integrity/test_network.py"
  "tests/integration/platform/data_daemon/performance/test_pre_network.py"
  "tests/integration/platform/data_daemon/performance/test_network.py"
)

# Substring shared by every 300s case-id; the two perf files are the last two
# `test_targets` entries and the only place these cases live.
heavy_case_filter="300s"
perf_targets=( "${test_targets[@]: -2}" )

# Run one pytest phase. $1 is the --log-file path, $2 the -k expression, and the
# remaining args are the pytest targets.
#
# --exitfirst stops on the first failure so the fast-fail ordering pays off.
# --log-cli-level=DEBUG turns on live structured logging from the SUT into
# pytest's captured stdout (so it ends up in $log_file via tee). --log-file
# captures the same at DEBUG level into a separate file so the structured
# records survive even if the tee buffer is cut short; each phase gets its own
# --log-file because pytest opens it in truncate mode and would otherwise
# clobber the previous phase's records.
run_pytest_phase() {
  local phase_log_file="$1"
  local keyword_expr="$2"
  shift 2
  pytest \
    --exitfirst \
    --tb=short \
    -vv \
    -o log_cli=true \
    -o log_cli_level=DEBUG \
    --log-file="$phase_log_file" \
    --log-file-level=DEBUG \
    -k "$keyword_expr" \
    "$@" \
    2>&1 | tee -a "$log_file"
  # Return pytest's exit code, not tee's.
  return "${PIPESTATUS[0]}"
}

run_tests() {
  log "running pytest with NCD_RUST_DAEMON=$NCD_RUST_DAEMON, NEURACORE_API_URL=$NEURACORE_API_URL"
  log "stdout log: $log_file"
  log "pytest --log-file: $pytest_log_file (per-phase _phase1 / _phase2 suffix)"

  # Phase 1: every suite EXCEPT the heavy 300s performance cases, in the
  # fastest-to-slowest order above, so a real bug surfaces in the cheapest suite
  # and the lighter performance cases (incl. network upload) aren't blocked by a
  # flaky 300s case.
  log "phase 1/2: all suites except the 300s performance cases (-k 'not $heavy_case_filter')"
  run_pytest_phase "${pytest_log_file%.log}_phase1.log" "not $heavy_case_filter" "${test_targets[@]}"
  local phase1_rc=$?
  if [[ $phase1_rc -ne 0 ]]; then
    return "$phase1_rc"
  fi

  # Phase 2: the heavy 300s cases at the very end — the pre-network (disk-only)
  # ones first, immediately before their network (upload) equivalents (perf
  # files are ordered pre-network then network in `perf_targets`).
  log "phase 2/2: 300s performance cases — pre-network then network (-k '$heavy_case_filter')"
  run_pytest_phase "${pytest_log_file%.log}_phase2.log" "$heavy_case_filter" "${perf_targets[@]}"
  return "$?"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
  log "==== integration test run starting ===="
  stop_daemon
  cleanup_state

  set +e
  run_tests
  exit_code=$?
  set -e

  log "==== integration test run finished (exit=$exit_code) ===="
  log "stdout log: $log_file"
  log "pytest --log-file: $pytest_log_file"
  exit "$exit_code"
}

main "$@"
