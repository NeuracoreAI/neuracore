#!/usr/bin/env bash
# Idempotently patch the vendored libdatachannel that `datachannel-sys` builds,
# to cap the OpenSSL DTLS handshake retransmit timer.
#
# Why: on fast loopback the DTLS responder's first flight is dropped while its
# ICE transport is momentarily not yet Connected (libdatachannel gates outgoing
# packets on ICE state). OpenSSL's default DTLS retransmit then waits a full
# second before resending, so every WebRTC connection takes ~1006ms and the
# connect-latency SLO (< 500ms p95) is impossible to meet. Capping the timer to
# ~50ms initial resends the dropped flight quickly: connect drops to ~56ms.
#
# This edits the crate source under CARGO_HOME in place (datachannel-sys ships
# libdatachannel inside the published crate; there is no upstream knob and no
# lighter override point). It is idempotent — re-runs detect the marker and skip
# — and only forces a datachannel-sys rebuild when it actually applies the patch.
#
# Invoked by build_wheel_artefacts.sh before building neuracore_webrtc. See
# reports/PR2-data-path.md "DTLS retransmit on loopback".

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
manifest="$workspace_root/Cargo.toml"

# Make sure the datachannel-sys source is fetched/extracted so we can find it.
cargo fetch --manifest-path "$manifest" >/dev/null 2>&1 || true

cargo_home="${CARGO_HOME:-$HOME/.cargo}"
target=""
for d in "$cargo_home"/registry/src/*/datachannel-sys-0.23.*; do
  candidate="$d/libdatachannel/src/impl/dtlstransport.cpp"
  if [[ -f "$candidate" ]]; then
    target="$candidate"
    break
  fi
done

if [[ -z "$target" ]]; then
  echo "warn: datachannel-sys source not found under $cargo_home; skipping DTLS patch" >&2
  exit 0
fi

if grep -q "NEURACORE PATCH" "$target"; then
  echo "==> libdatachannel DTLS retransmit patch already applied"
  exit 0
fi

python3 - "$target" <<'PY'
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    src = fh.read()

anchor = "\t\tSSL_set_ex_data(mSsl, TransportExIndex, this);\n"
if anchor not in src:
    sys.exit("error: DTLS patch anchor not found; libdatachannel layout changed")

block = (
    "\n"
    "\t\t// NEURACORE PATCH: cap the DTLS handshake retransmit timer. On fast\n"
    "\t\t// loopback the responder's first flight is dropped while its ICE is not\n"
    "\t\t// yet Connected (icetransport.cpp gates outgoing on ICE state); OpenSSL's\n"
    "\t\t// default 1s initial retransmit then dominates connection latency. Cap it\n"
    "\t\t// so a dropped flight is resent in tens of ms, not a full second. See\n"
    "\t\t// reports/PR2-data-path.md \"DTLS retransmit on loopback\".\n"
    "\t\tDTLS_set_timer_cb(mSsl, [](SSL *, unsigned int timer_us) -> unsigned int {\n"
    "\t\t\tunsigned int first = 50000;   /* 50 ms initial */\n"
    "\t\t\tunsigned int cap = 1000000;   /* 1 s backoff cap */\n"
    "\t\t\tunsigned int next = (timer_us == 0) ? first : timer_us * 2;\n"
    "\t\t\tif (next < first) next = first;\n"
    "\t\t\tif (next > cap) next = cap;\n"
    "\t\t\treturn next;\n"
    "\t\t});\n"
)

src = src.replace(anchor, anchor + block, 1)
with open(path, "w", encoding="utf-8") as fh:
    fh.write(src)
print(f"==> applied libdatachannel DTLS retransmit patch to {path}")
PY

# Force a datachannel-sys rebuild so the patched C++ is recompiled. cargo's
# fingerprint does not track edits to the crate's bundled C source, so drop the
# build artefacts explicitly. (On a fresh checkout there is nothing to remove.)
rm -rf "$workspace_root"/target/*/.fingerprint/datachannel-sys-* \
       "$workspace_root"/target/*/build/datachannel-sys-* \
       "$workspace_root"/target/*/deps/libdatachannel_sys-* \
       "$workspace_root"/target/*/.fingerprint/neuracore_webrtc-* \
       "$workspace_root"/target/*/libneuracore_webrtc.so 2>/dev/null || true
