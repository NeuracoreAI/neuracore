"""Guard that the rust workspace version tracks the python package version.

The daemon version check compares the two at runtime, so a drift between
`pyproject.toml` and `rust/Cargo.toml` would make every install raise
`DaemonVersionMismatchError` after the next release. `bump2version` rewrites
both files together; this test moves a drift failure from release time to CI.

Parsed by hand because `tomllib` needs python 3.11 and this suite also runs
on 3.10.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _version_in_section(toml_path: Path, section: str) -> str:
    in_section = False
    for line in toml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = stripped == f"[{section}]"
            continue
        if in_section:
            match = re.fullmatch(r'version\s*=\s*"([^"]+)"', stripped)
            if match:
                return match.group(1)
    raise AssertionError(f"No version found in [{section}] of {toml_path}")


def test_rust_workspace_version_matches_python_package_version() -> None:
    python_version = _version_in_section(REPO_ROOT / "pyproject.toml", "project")
    rust_version = _version_in_section(
        REPO_ROOT / "rust" / "Cargo.toml", "workspace.package"
    )
    assert rust_version == python_version, (
        f"rust/Cargo.toml workspace version {rust_version} does not match "
        f"pyproject.toml version {python_version}; the daemon version check "
        "compares these at runtime, so keep them in lockstep"
    )
