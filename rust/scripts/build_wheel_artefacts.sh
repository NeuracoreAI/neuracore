#!/usr/bin/env bash
# Build the standalone data-daemon binary and place it inside the neuracore
# package tree where the root pyproject.toml's [tool.maturin] include expects
# it:
#
#   neuracore/data_daemon/bin/data-daemon
#
# Re-exec'd by neuracore/data_daemon/__main__.py when NCD_RUST_DAEMON is truthy.
#
# The producer cdylib (_data_bridge) is NOT built here — maturin builds it from
# rust/data_daemon_bridge and names it correctly per platform. This script
# only handles the daemon binary, which lives in a *different* crate
# (data-daemon) that maturin does not build. Run it before `maturin build` at
# the repo root.
#
# Usage:
#   ./rust/scripts/build_wheel_artefacts.sh                 # native host target
#   ./rust/scripts/build_wheel_artefacts.sh --target <triple>
#     Linux: --target x86_64-unknown-linux-gnu inside the manylinux container.
#     macOS: --target aarch64-apple-darwin (Apple Silicon only; Intel Macs are
#            not supported) on a native arm64 runner, before `maturin build`.
#            This also pins MACOSX_DEPLOYMENT_TARGET and ad-hoc-signs the binary
#            so it matches the wheel tag and launches under Gatekeeper.
#
# See docs/rust_data_daemon_development.md#packaging-the-wheel for the pipeline.

set -euo pipefail

target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:?--target requires a triple}"
      shift 2
      ;;
    --target=*)
      target="${1#--target=}"
      shift
      ;;
    *)
      echo "error: unknown argument '$1'" >&2
      echo "usage: $0 [--target <triple>]" >&2
      exit 1
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$workspace_root/.." && pwd)"

package_dir="$repo_root/neuracore/data_daemon"
bin_dst="$package_dir/bin/data-daemon"

# For the Apple target, pin the same macOS deployment floor the maturin build
# uses so the cargo-built binary's `LC_BUILD_VERSION`/`minos` agrees with the
# wheel's platform tag — a mismatch causes confusing "incompatible architecture"
# or version load failures. arm64 only exists from 11.0. Honour a caller-provided
# MACOSX_DEPLOYMENT_TARGET if already set.
case "$target" in
  aarch64-apple-darwin)
    export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
    ;;
esac
if [[ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
  echo "==> MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
fi

cargo_args=(build --release --manifest-path "$workspace_root/Cargo.toml" -p data-daemon)
if [[ -n "$target" ]]; then
  cargo_args+=(--target "$target")
  bin_src="$workspace_root/target/$target/release/data-daemon"
else
  bin_src="$workspace_root/target/release/data-daemon"
fi

echo "==> cargo ${cargo_args[*]}"
cargo "${cargo_args[@]}"

if [[ ! -f "$bin_src" ]]; then
  echo "error: data-daemon binary not found at $bin_src" >&2
  exit 1
fi

mkdir -p "$(dirname "$bin_dst")"
install -m 0755 "$bin_src" "$bin_dst"
echo "    wrote $bin_dst"

# Apple Silicon refuses to exec an unsigned (or signature-invalidated) binary.
# rustc ad-hoc-signs arm64 Mach-O at link time, but `install` rewrites the file
# and the runtime `chmod` in __main__.py mutates it again — either can drop the
# signature. Re-apply an ad-hoc signature at the final on-disk path so the wheel
# ships a launchable binary. Best-effort: skip cleanly when not on a Mac host.
case "$target" in
  *-apple-darwin)
    if command -v codesign >/dev/null 2>&1; then
      codesign --force --sign - "$bin_dst"
      echo "    ad-hoc signed $bin_dst"
    else
      echo "    warning: codesign not found; daemon binary left unsigned" >&2
    fi
    ;;
esac
