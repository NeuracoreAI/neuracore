# Neuracore Data Daemon

The Neuracore Data Daemon is a small background service that runs on your machine and takes care of storing recordings locally and uploading them.

You can use it in two ways:
- **CLI first**: launch the daemon, then run your scripts
- **Script first**: run your script and let it start the daemon automatically

Profiles are optional. If you do not use a named profile, the daemon uses the default profile (and any environment variable overrides you set).

---

## What this README covers

- How to run the daemon (CLI or from a script)
- How profiles work (optional) and where they are stored
- The configuration fields you can set
- Environment variables that control DB path, recordings root, and other runtime settings
- The order of precedence (defaults, profile, environment variables, CLI)
- What happens to old daemon databases at startup (automatic schema migration)
- A full CLI reference for the commands currently in use

It does not explain internal implementation details.

---

## Quick start

### 1) Install (from repo root)

```bash
pip install -e .
```

Recommended for video recording, and **required** when running the Rust daemon
(`NCD_RUST_DAEMON=1`):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Both daemons encode video with the `ffmpeg` CLI, but they differ when `ffmpeg`
is missing or fails to initialise:
- The **Rust daemon** shells out to `ffmpeg` and runs a preflight at startup; if
  the binary is missing or the build is incompatible it fails fast with a clear
  message rather than starting and dropping every video recording. Install
  `ffmpeg` before launching.
- The **legacy Python daemon** prefers the `ffmpeg` CLI but automatically falls
  back to PyAV when `ffmpeg` is unavailable or the encoder fails to initialise.

### 2) Launch the daemon

With the default profile:

```bash
neuracore data-daemon launch
```

With a named profile:

```bash
neuracore data-daemon profile create recording
neuracore data-daemon profile update recording --storage-limit 2gb --bandwidth-limit 50mb --storage-path /data/records --num-threads 4
neuracore data-daemon launch --profile recording
```

Background (runs quietly):

```bash
neuracore data-daemon launch --profile recording --background
```

### 3) Check status and stop

```bash
neuracore data-daemon status
neuracore data-daemon stop
```

---

## Run your script without launching the daemon first

You do not have to use `neuracore data-daemon launch` beforehand. The daemon will automatically start in the background if it is not already running when your script needs it.

It will:
- check if the daemon is already running
- start it in the background if it is not running
- wait until it is ready before continuing

Example:

```python
import neuracore as nc

def main():
    nc.login()

    # The daemon starts automatically when needed
    nc.start_recording()
    # ...
    nc.stop_recording()
```

Choosing a profile when using auto-start:

```bash
export NEURACORE_DAEMON_PROFILE=recording
python your_script.py --record
```

When to use which approach:
- Use **CLI launch** if you want to start the daemon once and then run many scripts.
- Use **auto-start** if you want each script to be self contained.

---

## How it works (high level)

When you run:

```bash
neuracore data-daemon launch
```

the CLI launches the daemon as a separate background process. There are two
daemon implementations and the launcher picks one based on the `NCD_RUST_DAEMON`
flag (see [rust_data_daemon_development.md](rust_data_daemon_development.md)):

- **Rust daemon** — when `NCD_RUST_DAEMON` is truthy,
  the launcher `exec`s the native binary bundled in the `neuracore` wheel
  (Linux x86_64 and Apple-Silicon macOS) at
  `neuracore/data_daemon/bin/data-daemon`. This is the implementation described
  throughout this guide.
- **Legacy Python daemon (default)** — when `NCD_RUST_DAEMON` is unset or not
  truthy, the launcher runs the Python implementation instead.

Either daemon process:
- boots the internal components it needs
- starts its main loop
- stays running until you stop it (or the machine shuts down)

You may see simple messages when it stops:
- Daemon exited.
- Daemon stopped.

### Startup and schema migration

On startup the daemon opens its SQLite store (WAL mode) and applies any pending
schema migrations before serving requests.

The Rust daemon's schema is defined by the SQL migrations under
[rust/data_daemon/migrations/](../rust/data_daemon/migrations/) and applied
automatically with `sqlx::migrate!`. A fresh database is created and migrated on
first launch; an existing one has only the not-yet-applied migrations run. There
is no legacy single-table conversion — the migrations are the single source of
truth for the schema. To inspect the live database see
[rust_data_daemon_development.md#sqlite-state-inspection](rust_data_daemon_development.md#sqlite-state-inspection).

---

## Configuration

### Profiles

A profile is a YAML file that stores daemon settings you want to reuse.

Profiles are stored here:

```text
~/.neuracore/data_daemon/profiles/<name>.yaml
```

Manage profiles with:

```bash
neuracore data-daemon profile create <name>
neuracore data-daemon profile update [profile_name] [options...]
neuracore data-daemon profile get [profile_name]
neuracore data-daemon profile list
```

Notes:
- Profile names are positional arguments, not `--name` flags.
- `profile update` can be run without a profile name to update the default profile.
- `profile get` can be run without a profile name to read the default profile.
- The default profile is protected and cannot be deleted.

Delete a named profile with:

```bash
neuracore data-daemon profile delete <name>
```

If you do not use a named profile, the daemon uses the default profile.

---

### Config fields

These are the supported settings:

| Field | What it controls |
|---|---|
| `storage_limit` | Maximum local disk space the daemon should use for recordings (bytes). |
| `bandwidth_limit` | Maximum upload speed the daemon should use (bytes per second). |
| `spool_limit` | Cap on the producer's on-disk video spool backlog (bytes). When the un-encoded backlog reaches this size the producer applies backpressure to video logging instead of letting the spool fill the disk. `0` disables the bound. Defaults to 2 GiB. |
| `path_to_store_record` | Folder where recordings are stored. |
| `num_threads` | Number of worker threads used by the daemon. |
| `keep_wakelock_while_upload` | Whether to keep the machine awake during uploads (where supported). |
| `offline` | If enabled, uploading is disabled and data is only stored locally. |
| `api_key` | API key used for authenticating the daemon. |
| `current_org_id` | Which organisation the daemon should operate under. |
| `video_codec` |  Video encoding for RGB cameras. `h264_lossless` (default)/ `h264_medium` . |

---

### Byte units (for storage and bandwidth)

For `storage_limit` and `bandwidth_limit`, you can pass a raw number (bytes) or a unit suffixed value.

Supported units:
- b
- k or kb
- m or mb
- g or gb

Examples:

```bash
--storage-limit 500000000
--storage-limit 2gb
--bandwidth-limit 50mb
```

---

### Configuration precedence (which value wins)

When the daemon resolves its configuration, this is the order:

1. Built in defaults (used if nothing is provided)
2. Profile YAML (if you choose a profile)
3. Environment variables (optional overrides)
4. CLI values (explicit values you pass on the command line)

---

### Environment variables (optional)

You can override settings using environment variables. This is useful in CI, containers, or when you do not want to edit a profile file.

Supported environment variables:

| Setting | Environment variable |
|---|---|
| `storage_limit` | `NCD_STORAGE_LIMIT` |
| `bandwidth_limit` | `NCD_BANDWIDTH_LIMIT` |
| `spool_limit` | `NCD_SPOOL_LIMIT` |
| `path_to_store_record` | `NCD_PATH_TO_STORE_RECORD` |
| `num_threads` | `NCD_NUM_THREADS` |
| `keep_wakelock_while_upload` | `NCD_KEEP_WAKELOCK_WHILE_UPLOAD` |
| `offline` | `NCD_OFFLINE` |
| `api_key` | `NCD_API_KEY` |
| `current_org_id` | `NEURACORE_ORG_ID` (shared with the SDK) |
| `video_codec` | `NCD_VIDEO_CODEC` |

Note on `NEURACORE_ORG_ID`: the daemon uses it only as a fallback while
`~/.neuracore/config.json` has no organization set. The config file is watched
live, so an org selected with `neuracore select-org` mid-run always takes
precedence. This differs from the SDK, where `NEURACORE_ORG_ID` overrides the
config file.

Boolean values treat these as true:
- `1`
- `true`
- `yes`
- `y`

Examples:

```bash
export NCD_STORAGE_LIMIT=3gb
export NCD_OFFLINE=true
neuracore data-daemon launch
```

```bash
export NCD_PATH_TO_STORE_RECORD=/mnt/data/records
export NCD_NUM_THREADS=4
neuracore data-daemon launch --background
```

### Runtime path environment variables

These variables control where the daemon runtime artifacts live:

| Purpose | Environment variable | Default |
|---|---|---|
| PID file path | `NEURACORE_DAEMON_PID_PATH` | `~/.neuracore/daemon.pid` |
| SQLite DB path | `NEURACORE_DAEMON_DB_PATH` | `~/.neuracore/data_daemon/state.db` |
| Recordings root | `NEURACORE_DAEMON_RECORDINGS_ROOT` | sibling of DB path (`<db_dir>/recordings`) |
| Profile for launch/auto-start | `NEURACORE_DAEMON_PROFILE` | unset |
| Enable debug mode | `NDD_DEBUG` | `false` |

Recommended for containers/dev environments:

```bash
export NEURACORE_DAEMON_DB_PATH=/workspaces/neuracore/data_daemon_state.db
export NEURACORE_DAEMON_RECORDINGS_ROOT=/workspaces/neuracore/recordings
```

---

## CLI reference

### `neuracore data-daemon profile create`

```bash
neuracore data-daemon profile create <name>
```

Example:

```bash
neuracore data-daemon profile create laptop
```

### `neuracore data-daemon profile update`

Update a named profile:

```bash
neuracore data-daemon profile update <name> [--storage-limit <bytes|unit>] [--bandwidth-limit <bytes|unit>] [--spool-limit <bytes|unit>] [--storage-path <path>] [--num-threads <n>] [--wakelock|--no-wakelock] [--offline|--online] [--api-key <key>] [--current-org-id <org_id>] [--video-codec <h264_lossless|h264_medium>]
```

Update the default profile:

```bash
neuracore data-daemon profile update [--storage-limit <bytes|unit>] [--bandwidth-limit <bytes|unit>] [--spool-limit <bytes|unit>] [--storage-path <path>] [--num-threads <n>] [--wakelock|--no-wakelock] [--offline|--online] [--api-key <key>] [--current-org-id <org_id>] [--video-codec <h264_lossless|h264_medium>]
```

Example:

```bash
neuracore data-daemon profile update laptop --storage-limit 2gb --offline
neuracore data-daemon profile update laptop --video-codec h264_medium  # lossy-only uploads
neuracore data-daemon profile update laptop --video-codec h264_lossless # back to the default
```

### `neuracore data-daemon profile get`

Describe a profile:

```bash
neuracore data-daemon profile get [profile_name]
```

Examples:

```bash
neuracore data-daemon profile get high-bandwidth
neuracore data-daemon profile get low-bandwidth
neuracore data-daemon profile get
```

### `neuracore data-daemon profile list`

```bash
neuracore data-daemon profile list
```

### `neuracore data-daemon profile delete`

```bash
neuracore data-daemon profile delete <name>
```

Notes:
- The profile name is required.
- The default profile cannot be deleted.

### `neuracore data-daemon launch`

```bash
neuracore data-daemon launch [--profile <name>] [--background] [--debug]
```

`--debug` raises the log level (equivalent to setting `NDD_DEBUG=1`).

Examples:

```bash
neuracore data-daemon launch
```

```bash
neuracore data-daemon launch --profile laptop
```

```bash
neuracore data-daemon launch --profile laptop --background
```

### `neuracore data-daemon status`

```bash
neuracore data-daemon status
```

### `neuracore data-daemon stop`

```bash
neuracore data-daemon stop
```

### `neuracore data-daemon reset`

Stops the daemon (if running) and removes **all** of its local state: the
recordings tree, the SQLite database, the PID file, and the shared-memory
artefacts. This is destructive and cannot be undone — use it to return a wedged
host to a clean slate.

```bash
neuracore data-daemon reset          # prompts for confirmation
neuracore data-daemon reset --yes    # skip the prompt (for scripts)
```

---

## Offline Recordings

### Single node

Set `offline: true` in your profile, then launch the daemon with that profile as usual. Record normally, all data is stored locally. When you have internet access again, relaunch the daemon without offline mode and it will automatically upload your recordings to Neuracore.

```bash
# Set offline mode
neuracore data-daemon profile update my_profile --offline

# Record offline
neuracore data-daemon launch --profile my_profile

# Back online, disable offline mode and relaunch
neuracore data-daemon profile update my_profile --online
neuracore data-daemon launch --profile my_profile
```

### Multi node

For multi-node offline setups, collect your data using a data distribution system like ROS across multiple nodes, then use a single node to import your collected data into Neuracore.

---

## Troubleshooting

### Daemon already running
You tried to launch it while it is already running.

Try:

```bash
neuracore data-daemon status
neuracore data-daemon stop
neuracore data-daemon launch
```

### Daemon failed to start
Run it in the foreground so you can see the output:

```bash
neuracore data-daemon launch
```

If it still fails, check your profiles:

```bash
neuracore data-daemon profile list
neuracore data-daemon profile get
neuracore data-daemon profile get <name>
```

A common cause is trying to launch with `offline: false` and no valid `api_key`.

### Background launch reports success but daemon is not running

`neuracore data-daemon launch --background` currently confirms that the subprocess started, but it may still exit shortly afterward during bootstrap, for example if authentication fails.

If background launch appears successful but `status` later shows the daemon is not running, rerun in the foreground:

```bash
neuracore data-daemon launch
```

### Which video encoder backend is being used

Both daemons encode video with `ffmpeg`, but they handle a missing or broken
`ffmpeg` differently:
- **Rust daemon** (`NCD_RUST_DAEMON=1`) — verifies `ffmpeg` at startup. If the
  binary is missing from `PATH`, or the local build cannot run the encode the
  daemon needs, the preflight fails and the daemon refuses to start (rather than
  starting and silently dropping every video recording).
- **Legacy Python daemon** (default) — uses the `ffmpeg` CLI when it is on
  `PATH` and falls back to PyAV when `ffmpeg` is unavailable or fails to
  initialise.

Confirm `ffmpeg` is installed and runnable:

```bash
ffmpeg -version
```

If that command fails, install `ffmpeg` (see [Quick start](#1-install-from-repo-root)) for the Rust daemon, or rely on the PyAV fallback under the Python daemon.

### Migration issues on startup

If startup logs mention migration failures:

1. Verify the daemon is using the DB you expect:

```bash
echo "$NEURACORE_DAEMON_DB_PATH"
```

2. Ensure the process has write permission to DB directory and recordings root.

3. Start in foreground and read migration logs:

```bash
neuracore data-daemon launch
```

4. If migration fails repeatedly, stop daemon and keep a backup copy of the DB before retrying.

### Shutdown hangs or noisy `KeyboardInterrupt` traces

Repeated `Ctrl+C` while shutdown is already in progress can interrupt cleanup.

Recommended:
- Press `Ctrl+C` once, then wait for shutdown to complete
- For normal operation, use:

```bash
neuracore data-daemon stop
```
