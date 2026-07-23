#!/usr/bin/env bash
set -euo pipefail

output="${RUNNER_TEMP:-/tmp}/sleep-latency.trace"
time_limit="5m"
window="10s"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            output="$2"
            shift 2
            ;;
        --time-limit)
            time_limit="$2"
            shift 2
            ;;
        --window)
            window="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [--output PATH] [--time-limit 5m] [--window 10s] -- COMMAND [ARG ...]" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "This capture script requires macOS." >&2
    exit 2
fi

anomalies_path="${NEURACORE_SLEEP_ANOMALIES_PATH:-${output%.trace}-anomalies.jsonl}"
export NEURACORE_SLEEP_ANOMALIES_PATH="$anomalies_path"

echo "== Sleep-latency capture environment =="
sw_vers
system_profiler SPHardwareDataType
echo "logical_cpus=$(sysctl -n hw.logicalcpu) physical_cpus=$(sysctl -n hw.physicalcpu)"
echo "hardware_model=$(sysctl -n hw.model)"
if vm_present="$(sysctl -n kern.hv_vmm_present 2>/dev/null)"; then
    echo "hypervisor_present=$vm_present"
else
    echo "hypervisor_present=unavailable"
fi
echo "CI=${CI:-unset} RUNNER_ENVIRONMENT=${RUNNER_ENVIRONMENT:-unset}"
echo "cpu_affinity=unavailable_on_macos_public_api"
pmset -g batt || true
pmset -g custom || true
ulimit -a
ps -o pid=,ppid=,pri=,nice=,state=,command= -p "$$"
"$1" -VV 2>&1 || true
xcrun xctrace version
xcrun xctrace list instruments

trace_args=(
    record
    --template "System Trace"
    --output "$output"
    --time-limit "$time_limit"
    --window "$window"
    --no-prompt
    --target-stdout -
)
available_instruments="$(xcrun xctrace list instruments 2>&1)"
for instrument in "Points of Interest" "Thermal State"; do
    if grep -q "$instrument" <<<"$available_instruments"; then
        trace_args+=(--instrument "$instrument")
    else
        echo "Optional xctrace instrument unavailable: $instrument"
    fi
done

echo "trace_output=$output"
echo "anomalies_output=$anomalies_path"
echo "Launching workload under System Trace"
# xctrace only has an explicit target-stdout option. Route the workload's
# stderr into stdout so pytest live logs remain visible in GitHub Actions.
xcrun xctrace "${trace_args[@]}" --launch -- \
    /bin/bash -c 'exec "$@" 2>&1' bash "$@"
