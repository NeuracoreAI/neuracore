#!/usr/bin/env bash
# Build the Rust artefacts shipped in the neuracore wheel and place them
# inside the Python package tree where setup.py's package_data expects them.
#
# Two artefacts are produced:
#   1. The data-daemon binary -> neuracore/data_daemon/bin/data-daemon
#      Re-exec'd by neuracore/data_daemon/__main__.py when NCD_RUST_DAEMON
#      is truthy.
#   2. The data_daemon_producer cdylib -> neuracore/data_daemon/_native_producer.so
#      Renamed from libdata_daemon_producer.so so PyO3's PyInit__native_producer
#      is discoverable by the Python import machinery.
#   3. The neuracore_webrtc cdylib ->
#      neuracore/core/streaming/p2p/_native_webrtc.so
#      Renamed from libneuracore_webrtc.so so PyO3's PyInit__native_webrtc is
#      discoverable. Gated at runtime by NCD_RUST_WEBRTC (see
#      neuracore/core/streaming/p2p/webrtc_selection.py).
#
# See docs/rust_data_daemon_development.md#packaging-the-wheel for the rationale.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$workspace_root/.." && pwd)"

package_dir="$repo_root/neuracore/data_daemon"
bin_dst="$package_dir/bin/data-daemon"
cdylib_dst="$package_dir/_native_producer.so"

webrtc_dst="$repo_root/neuracore/core/streaming/p2p/_native_webrtc.so"

# PyO3's build-config probes (in order) PYO3_PYTHON, VIRTUAL_ENV/bin/python,
# CONDA_PREFIX/bin/python, then /usr/bin/python. On minimal Debian/Ubuntu
# images only python3 is on PATH and some dev environments set VIRTUAL_ENV to
# a host (e.g. /usr) where neither python nor python3 lives — both cases
# leave pyo3 with no interpreter and the build fails.
#
# When PYO3_PYTHON isn't set explicitly, walk the same probe chain ourselves
# and fall back to whatever `python3` resolves to on PATH if none of the
# usual candidates exist. Caller-supplied PYO3_PYTHON always wins.
if [[ -z "${PYO3_PYTHON:-}" ]]; then
  pyo3_candidates=()
  [[ -n "${VIRTUAL_ENV:-}" ]] && pyo3_candidates+=("$VIRTUAL_ENV/bin/python")
  [[ -n "${CONDA_PREFIX:-}" ]] && pyo3_candidates+=("$CONDA_PREFIX/bin/python")
  pyo3_candidates+=("/usr/bin/python")
  for candidate in "${pyo3_candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      export PYO3_PYTHON="$candidate"
      break
    fi
  done
  if [[ -z "${PYO3_PYTHON:-}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      export PYO3_PYTHON
      PYO3_PYTHON="$(command -v python3)"
      echo "==> using PYO3_PYTHON=$PYO3_PYTHON (no python found in VIRTUAL_ENV/CONDA_PREFIX/system)"
    else
      echo "error: no python interpreter found; set PYO3_PYTHON or install python3" >&2
      exit 1
    fi
  fi
fi

echo "==> cargo build --release -p data-daemon"
cargo build --release --manifest-path "$workspace_root/Cargo.toml" -p data-daemon

echo "==> cargo build --release -p data_daemon_producer"
cargo build --release --manifest-path "$workspace_root/Cargo.toml" -p data_daemon_producer

# Patch the libdatachannel that datachannel-sys builds to cap its DTLS retransmit
# timer (otherwise every WebRTC connection eats OpenSSL's 1s loopback retransmit;
# see reports/PR2-data-path.md). Idempotent and only forces a rebuild on change.
echo "==> patch libdatachannel (DTLS retransmit timer)"
bash "$script_dir/patch_libdatachannel.sh"

echo "==> cargo build --release -p neuracore_webrtc"
cargo build --release --manifest-path "$workspace_root/Cargo.toml" -p neuracore_webrtc

mkdir -p "$(dirname "$bin_dst")"
install -m 0755 "$workspace_root/target/release/data-daemon" "$bin_dst"
echo "    wrote $bin_dst"

# cdylib filename varies by platform: libfoo.so on Linux, libfoo.dylib on macOS,
# foo.dll on Windows. Linux-only support per data-daemon-rewrite.md §Open items.
cdylib_src="$workspace_root/target/release/libdata_daemon_producer.so"
if [[ ! -f "$cdylib_src" ]]; then
  echo "error: cdylib not found at $cdylib_src" >&2
  echo "       (data-daemon-rewrite.md is Linux-first; macOS/Windows are not supported)" >&2
  exit 1
fi
install -m 0755 "$cdylib_src" "$cdylib_dst"
echo "    wrote $cdylib_dst"

webrtc_src="$workspace_root/target/release/libneuracore_webrtc.so"
if [[ ! -f "$webrtc_src" ]]; then
  echo "error: cdylib not found at $webrtc_src" >&2
  echo "       (data-daemon-rewrite.md is Linux-first; macOS/Windows are not supported)" >&2
  exit 1
fi
install -m 0755 "$webrtc_src" "$webrtc_dst"
echo "    wrote $webrtc_dst"
