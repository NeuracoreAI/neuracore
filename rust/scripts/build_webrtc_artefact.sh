#!/usr/bin/env bash
# Build ONLY the neuracore_webrtc cdylib and drop it into the Python package tree
# as neuracore/core/streaming/p2p/_native_webrtc.so (renamed from
# libneuracore_webrtc.so so PyO3's PyInit__native_webrtc is import-discoverable).
#
# This is the slice of build_wheel_artefacts.sh the WebRTC stack needs: the
# frontend integration devcontainer wants the NCD_RUST_WEBRTC native module but
# not the data-daemon / producer artefacts, so it avoids building (and pulling the
# deps of) those two crates. The .so is gated at runtime by NCD_RUST_WEBRTC (see
# neuracore/core/streaming/p2p/webrtc_selection.py).
#
# See docs/rust_data_daemon_development.md#packaging-the-wheel for the rationale.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$workspace_root/.." && pwd)"

webrtc_dst="$repo_root/neuracore/core/streaming/p2p/_native_webrtc.so"

# Pick the interpreter PyO3 links against so the .so matches the target Python's
# ABI (the extension is not abi3). Caller-supplied PYO3_PYTHON always wins; else
# probe VIRTUAL_ENV / CONDA_PREFIX / system, then fall back to python3 on PATH.
# Mirrors build_wheel_artefacts.sh.
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
    else
      echo "error: no python interpreter found; set PYO3_PYTHON or install python3" >&2
      exit 1
    fi
  fi
fi
echo "==> building neuracore_webrtc against PYO3_PYTHON=$PYO3_PYTHON"

# Patch the libdatachannel that datachannel-sys builds to cap its DTLS retransmit
# timer (otherwise every WebRTC connection eats OpenSSL's 1s loopback retransmit;
# see reports/PR2-data-path.md). Idempotent and only forces a rebuild on change.
echo "==> patch libdatachannel (DTLS retransmit timer)"
bash "$script_dir/patch_libdatachannel.sh"

echo "==> cargo build --release -p neuracore_webrtc"
cargo build --release --manifest-path "$workspace_root/Cargo.toml" -p neuracore_webrtc

webrtc_src="$workspace_root/target/release/libneuracore_webrtc.so"
if [[ ! -f "$webrtc_src" ]]; then
  echo "error: cdylib not found at $webrtc_src (Linux-first; macOS/Windows unsupported)" >&2
  exit 1
fi
install -m 0755 "$webrtc_src" "$webrtc_dst"
echo "    wrote $webrtc_dst"
