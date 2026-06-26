# Rust Data Daemon — Developer Guide
<!-- cspell:disable -->
This guide is for developers working on the Rust data daemon under [rust/](../rust/). For the end-user CLI (profiles, launch, stop, troubleshooting) see [data_daemon.md](data_daemon.md).

---

## Workspace layout

The [rust/](../rust/) directory is a Cargo workspace with three members declared in [rust/Cargo.toml](../rust/Cargo.toml):

| Crate | Path | What it is |
|---|---|---|
| `data-daemon` | [rust/data_daemon/](../rust/data_daemon/) | The daemon binary — CLI, lifecycle, SQLite state, IPC listener, per-trace pipeline, encoding. |
| `data_daemon_shared` | [rust/data_daemon_shared/](../rust/data_daemon_shared/) | Shared library — IPC envelope types and service-name constants, plus the daemon configuration model and filesystem-path resolution the two processes must compute identically. Linked by both the daemon and the producer crate. |
| `data_daemon_bridge` | [rust/data_daemon_bridge/](../rust/data_daemon_bridge/) | PyO3 `cdylib` — producer-side IPC client exposed to Python as `neuracore.data_daemon._data_bridge`. |

Shared workspace dependencies and the Rust edition (`2021`) are pinned in [rust/Cargo.toml](../rust/Cargo.toml); individual crates inherit them via `.workspace = true`.

---

## Architecture

The producer is a *thin shipper*: it publishes source/sensor/timestamp-tagged
data and fire-and-forget lifecycle events, and the daemon owns all recording
identity and routing. Pixel data never travels the IPC bus — the producer spools
NUT chunks to disk and only announces them.

```mermaid
flowchart LR
    subgraph SDK["Python SDK"]
      LOG["log_joints / log_json / log_frame<br/>start / stop / cancel_recording"]
    end
    subgraph PROD["data_daemon_bridge (PyO3 cdylib)"]
      LOG -->|GIL released| PUBT["publisher thread"]
      LOG -->|RGB frames| WRT["writer thread → NUT spool on disk"]
    end
    PUBT -->|postcard envelopes| BUS[("iceoryx2 commands")]
    WRT -. VideoChunkReady .-> PUBT
    BUS --> LIS["ipc::listener (adaptive poll)"]
    LIS --> DISP["dispatcher<br/>single task · holdback · publish-clock routing"]
    DISP --> ACT["per-trace actors"]
    ACT -->|fire-and-forget| TW["trace_writer<br/>batched DB write-behind"]
    ACT -->|fire-and-forget| JW["json_writer (IO thread)"]
    ACT -->|spawn_blocking| FF["ffmpeg chunk encode"]
    TW --> DB[("SQLite WAL")]
    LIS -->|recording-id queries| DB
    subgraph CLOUD["cloud coordinators"]
      REG["registration"] --> UP["uploader"] --> STA["status"]
      PR["progress"]
      NOT["start / stop / cancel notifiers"]
      RP["reaper"]
    end
    DB --> CLOUD
    UP -->|PUT chunks| GCS[("GCS / backend")]
```

A recording's lifecycle — the daemon assigns the local `recording_index`
immediately and the cloud `recording_id` is backfilled asynchronously:

```mermaid
sequenceDiagram
    participant SDK
    participant Producer
    participant Dispatcher
    participant Store as SQLite
    participant StartN as start-notifier
    participant Backend

    SDK->>Producer: start_recording(source)
    Producer->>Dispatcher: StartRecording envelope
    Dispatcher->>Store: insert recording (recording_index)
    Dispatcher-->>StartN: RecordingStarted (event bus)
    StartN->>Backend: POST /recording/start
    Backend-->>StartN: cloud recording_id
    StartN->>Store: persist recording_id
    SDK->>Producer: log_* (data / frames)
    Producer->>Dispatcher: Data / VideoChunkReady (routed by publish ts)
    SDK->>Producer: stop_recording
    Producer->>Dispatcher: StopRecording envelope
    Dispatcher->>Store: mark stopped
    Note over StartN,Backend: stop-notifier POSTs /recording/stop
    Note over StartN,Backend: uploader PUTs traces, reaper reclaims once fully uploaded
```

---

## Prerequisites

### Rust toolchain

The pre-commit hooks and CI ([.github/workflows/rust-data-daemon.yaml](../.github/workflows/rust-data-daemon.yaml)) invoke `cargo` from your `PATH`. Install via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup component add rustfmt clippy
```

CI uses `stable` (via `dtolnay/rust-toolchain@stable`), so any recent stable toolchain works locally.

### System dependencies

- **ffmpeg + ffprobe** — required by the video-encoder subprocess and the `encoding::video_encoder` / `encoding::nut_writer` test suites (tests that need ffmpeg self-skip if it's missing, but the daemon itself depends on it at runtime):

    ```bash
    sudo apt-get update && sudo apt-get install -y ffmpeg
    ```

- **maturin** (only when working on the `data_daemon_bridge` PyO3 crate):

    ```bash
    pip install maturin
    ```

---

## Common commands

Run all `cargo` commands from the workspace root [rust/](../rust/) unless noted otherwise. Targeting a specific crate uses `-p <crate-name>` (the workspace member names from the table above).

### Build

```bash
# Whole workspace (debug)
cargo build --workspace

# Release binary, daemon only
cargo build --release -p data-daemon

# Producer cdylib only
cargo build -p data_daemon_bridge
```

The release binary lands at [rust/target/release/data-daemon](../rust/target/release/data-daemon).

### Test

```bash
# Whole workspace
cargo test --workspace

# A specific crate
cargo test -p data-daemon
cargo test -p data_daemon_shared

# A specific module or test name (partial match)
cargo test -p data-daemon pipeline::dispatcher
cargo test -p data-daemon encoding::metadata::fixture_matches_python_video_trace_output
```

Tests that shell out to `ffmpeg` / `ffprobe` self-skip on hosts without those binaries — install them (see above) to exercise the full encoding suite.

### Format and lint

These are the gates the pre-commit hooks and CI enforce; run them before pushing.

```bash
cargo fmt --check                         # passive — fails if anything is unformatted
cargo fmt                                 # apply formatting
cargo clippy --all-targets -- -D warnings # workspace-wide, warnings denied
```

The pre-commit hooks in [.pre-commit-config.yaml](../.pre-commit-config.yaml) run `cargo fmt` and `cargo clippy` against [rust/data_daemon/](../rust/data_daemon/) only — `language: system`, so they need the local toolchain. If you don't touch any file under `rust/data_daemon/`, the cargo hooks are skipped.

### Documentation

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

The `-D warnings` flag matches CI; broken intra-doc links and missing items fail the build. Generated HTML lands at [rust/target/doc/](../rust/target/doc/).

---

## Running the daemon locally

Once built, run the CLI directly through cargo. The command tree mirrors the user-facing CLI documented in [data_daemon.md#cli-reference](data_daemon.md#cli-reference):

```bash
cargo run -p data-daemon -- profile list
cargo run -p data-daemon -- profile create dev
cargo run -p data-daemon -- launch --profile dev
cargo run -p data-daemon -- status
cargo run -p data-daemon -- stop
```

### Pointing the daemon at scratch paths

To avoid polluting your real `~/.neuracore`, override the runtime paths (also documented in [data_daemon.md#runtime-path-environment-variables](data_daemon.md#runtime-path-environment-variables)):

```bash
export NEURACORE_DAEMON_PID_PATH=/tmp/ndd-dev/daemon.pid
export NEURACORE_DAEMON_DB_PATH=/tmp/ndd-dev/state.db
export NEURACORE_DAEMON_RECORDINGS_ROOT=/tmp/ndd-dev/recordings
cargo run -p data-daemon -- launch
```

### Foreground vs background

- **Foreground** (default): logs stream to stderr, Ctrl-C triggers graceful shutdown. Use this for almost everything during development.
- **Background** (`launch --background`): double-forks via [lifecycle::daemonize](../rust/data_daemon/src/lifecycle/daemonize.rs); logs go to a `daemon.log` sibling of the SQLite DB. Use this when you specifically need to test the daemonized path or PID-file handling.

### Debug logging

The daemon defaults to the `warn` tracing level. `--debug` (or `NDD_DEBUG=1`) bumps it to `debug`. `RUST_LOG` overrides both — for example:

```bash
RUST_LOG=data_daemon=trace,iceoryx2=warn cargo run -p data-daemon -- launch
```

---

## Working on the PyO3 producer

The `data_daemon_bridge` crate compiles to a `cdylib` that Python imports as `neuracore.data_daemon._data_bridge` (shipped by the separate `neuracore-data-daemon` package). During development, use `maturin develop` from that package directory — its `pyproject.toml` carries the `module-name` and `manifest-path` that point maturin at the crate:

```bash
cd packaging/neuracore-data-daemon
maturin develop
python -c "import neuracore.data_daemon._data_bridge as p; print(p)"
```

To route the Python SDK through the data bridge instead of the legacy zmq one, set the rollout flag:

```bash
export NCD_RUST_DAEMON=1
python your_script.py
```

Selection logic lives in [neuracore/data_daemon/rust_selection.py](../neuracore/data_daemon/rust_selection.py); both the daemon binary handoff and the SDK's `DataStream` construction read it. A small shim bridges the data bridge to the Python `ProducerChannel` contract.

---

## Packaging: two distributions

The Rust daemon is shipped as a **separate, Linux-only PyPI distribution** so that
`neuracore` itself stays pure-Python and installs on every platform.

| Distribution | Build | Installs on | Ships |
|---|---|---|---|
| `neuracore` | pure-Python (setuptools, [pyproject.toml](../pyproject.toml)) | everywhere (`py3-none-any`) | all the Python SDK + the `daemon` extra |
| `neuracore-data-daemon` | maturin ([packaging/neuracore-data-daemon/pyproject.toml](../packaging/neuracore-data-daemon/pyproject.toml)) | Linux x86_64 only | the `_data_bridge` PyO3 extension + the `data-daemon` binary |

`neuracore-data-daemon` is a base dependency of `neuracore` carrying the PEP 508
marker `; sys_platform == 'linux'`, so `pip install neuracore` pulls the daemon
**automatically on Linux** and is a no-op on macOS/Windows (where the SDK falls
back to the Python daemon). The `daemon` extra is kept as a backward-compatible
explicit alias — `pip install neuracore[daemon]` resolves to the same thing.

The two artefacts and how `neuracore` finds them:

| Artefact | Location (in `neuracore-data-daemon`) | Source crate | Reached from `neuracore` via |
|---|---|---|---|
| Daemon binary | `neuracore/data_daemon/bin/data-daemon` | `data-daemon` (bin) | `rust_selection.rust_daemon_binary_path()` → `files("neuracore.data_daemon")/"bin"/"data-daemon"` |
| Producer extension | `neuracore/data_daemon/_data_bridge*.so` | `data_daemon_bridge` (cdylib) | `recording_context._load_native()` → `import neuracore.data_daemon._data_bridge` |

Both lookups degrade gracefully when `neuracore-data-daemon` is absent (binary path → `None`, extension import → a helpful `RuntimeError`).

The daemon wheel installs its files **into** the `neuracore/data_daemon/` import path the `neuracore` wheel owns, but ships **no** `__init__.py` for those dirs — so the two distributions share the directory with no file collision: `neuracore` owns the `.py` + `__init__.py`, `neuracore-data-daemon` owns the `.so` + binary. (Installed alone, the daemon wheel's `neuracore/data_daemon/` imports as a PEP 420 namespace package; installed alongside `neuracore`, it's a normal package — both ways `import neuracore.data_daemon._data_bridge` resolves.)

### Local build

The daemon binary is a *separate* crate that maturin doesn't build, so the helper
script compiles it into the daemon package tree first; then maturin builds the
extension:

```bash
./rust/scripts/build_wheel_artefacts.sh   # cargo build -p data-daemon -> packaging/.../neuracore/data_daemon/bin/data-daemon
cd packaging/neuracore-data-daemon
maturin develop                            # builds + editable-installs _data_bridge into the active env
```

`maturin develop` builds and installs the **extension only** — it doesn't run the
binary build or evaluate `include`, so run the helper script first for an
end-to-end daemon. For the SDK itself, `pip install -e .` (or `.[dev]`) at the repo
root installs `neuracore` pure-Python with no Rust.

### Building wheels

```bash
# neuracore (pure Python, py3-none-any)
python -m build

# neuracore-data-daemon (Linux only)
./rust/scripts/build_wheel_artefacts.sh
cd packaging/neuracore-data-daemon
maturin build --release --out dist --interpreter python3.11
```

The daemon wheel is platform-specific: pyo3 uses `extension-module` without `abi3`,
so each is one Python minor × one platform and `--interpreter pythonX.Y` must be
passed. `include` bundles the daemon binary; no sdist is shipped for the daemon
package (its Rust sources live outside the package dir).

> **Daemon is Linux only.** The daemon stack uses `iceoryx2` shared-memory IPC and
> Linux-only syscalls (`sync_file_range`, `gettid`). `neuracore-data-daemon` builds
> and ships `linux-x86_64` wheels only; `neuracore` is unaffected on other platforms.

### CI

[.github/workflows/build-wheels.yaml](../.github/workflows/build-wheels.yaml) builds
the `neuracore-data-daemon` wheels — a `linux-x86_64` × py3.10/3.11 `PyO3/maturin-action`
matrix that builds the daemon binary inside the manylinux container (glibc match,
plus a `clang` install for iceoryx2's bindgen), runs `maturin build`, and smoke-tests
`import neuracore.data_daemon._data_bridge` + the executable binary. The pure-Python
`neuracore` wheel needs no special CI — it's built with `python -m build`.

### Release path

The [release workflow](../.github/workflows/release.yaml) bumps the version (in both
pyprojects + the `daemon` extra pin, kept in lockstep by [.bumpversion.cfg](../.bumpversion.cfg)),
pushes the tag, then: the `release` job builds + `twine`-publishes the pure-Python
`neuracore` wheel + sdist (`python -m build`), and a `publish-daemon-wheels`
linux-x86_64 matrix checks out the tag, builds the `neuracore-data-daemon` wheels,
and publishes them. `twine --skip-existing` makes re-running any leg idempotent.

---

## SQLite state inspection

The daemon stores its state at `NEURACORE_DAEMON_DB_PATH` (default `~/.neuracore/data_daemon/state.db`), opened in WAL mode. Migrations live in [rust/data_daemon/migrations/](../rust/data_daemon/migrations/) and run automatically on startup via `sqlx::migrate!`. To poke at the live DB:

```bash
sqlite3 "$NEURACORE_DAEMON_DB_PATH" ".tables"
sqlite3 "$NEURACORE_DAEMON_DB_PATH" "SELECT trace_id, write_status, registration_status, upload_status FROM traces;"
```

The schema is defined by the `sqlx` migrations under [rust/data_daemon/migrations/](../rust/data_daemon/migrations/).

---

## Before committing

The pre-commit hooks cover formatting and lint, but CI also runs the test suite and builds the release binary and the docs. To catch failures locally before pushing:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo build --release -p data-daemon
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

Run `pre-commit run --all-files` from the repo root to exercise the full hook chain (including the Python checks).

---

## Further reading

- [data_daemon.md](data_daemon.md) — end-user CLI, profiles, environment variables, troubleshooting.
- [contribution_guide.md](contribution_guide.md) — repo-wide contribution flow, release process, PR conventions.
