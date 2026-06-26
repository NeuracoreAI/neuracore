#!/usr/bin/env bash
# Build the standalone data-daemon binary and place it inside the
# neuracore-data-daemon package tree where `maturin build`'s [tool.maturin]
# include expects it:
#
#   packaging/neuracore-data-daemon/neuracore/data_daemon/bin/data-daemon
#
# Re-exec'd by neuracore/data_daemon/__main__.py when NCD_RUST_DAEMON is truthy.
#
# The producer cdylib (_data_bridge) is NOT built here — maturin builds it from
# rust/data_daemon_bridge and names it correctly per platform. This script
# only handles the daemon binary, which lives in a *different* crate
# (data-daemon) that maturin does not build. Run it before `maturin build` of
# the neuracore-data-daemon package.
#
# Usage:
#   ./rust/scripts/build_wheel_artefacts.sh                 # native host target
#   ./rust/scripts/build_wheel_artefacts.sh --target <triple>
#     e.g. --target x86_64-unknown-linux-gnu inside the manylinux container.
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

package_dir="$repo_root/packaging/neuracore-data-daemon/neuracore/data_daemon"
bin_dst="$package_dir/bin/data-daemon"

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
