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
- Environment variables that control DB path, recordings root, and upload concurrency
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

Optional, but recommended for video performance:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

The data daemon prefers the `ffmpeg` CLI encoder for recording. If the binary is not installed or encoder init fails, it automatically falls back to PyAV.

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
    dataset = nc.connect_dataset("My Robot Dataset")

    # The daemon starts automatically when needed
    dataset.start_recording()
    # ...
    dataset.stop_recording()
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

the CLI starts the daemon as a separate Python process by running:

```text
python -m neuracore.data_daemon.runner_entry
```

That daemon process:
- boots the internal components it needs
- starts its main loop
- stays running until you stop it (or the machine shuts down)

You may see simple messages when it stops:
- Daemon exited.
- Daemon stopped.

### Startup and schema migration

On startup, the daemon initializes the SQLite store and ensures schema compatibility.

If an older single-table schema is detected (legacy `traces.status` format), the daemon
automatically migrates data to the current schema:

- `traces` rows are transformed into lifecycle fields:
  - `write_status`
  - `registration_status`
  - `upload_status`
- `recordings` rows are generated per unique `recording_id`
- Existing trace metadata/bytes/error fields are preserved
- Migration runs before normal startup reconciliation

Migration runs once per DB file. After a successful migration, startup continues normally.

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
| `path_to_store_record` | Folder where recordings are stored. |
| `num_threads` | Number of worker threads used by the daemon. |
| `keep_wakelock_while_upload` | Whether to keep the machine awake during uploads (where supported). |
| `offline` | If enabled, uploading is disabled and data is only stored locally. |
| `api_key` | API key used for authenticating the daemon. |
| `current_org_id` | Which organisation the daemon should operate under. |

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
| `path_to_store_record` | `NCD_PATH_TO_STORE_RECORD` |
| `num_threads` | `NCD_NUM_THREADS` |
| `keep_wakelock_while_upload` | `NCD_KEEP_WAKELOCK_WHILE_UPLOAD` |
| `offline` | `NCD_OFFLINE` |
| `api_key` | `NCD_API_KEY` |
| `current_org_id` | `NCD_CURRENT_ORG_ID` |

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

Recommended upload concurrency:
- Most machines: `5-10`
- Start at `5`, increase only if CPU/network/disk are stable
- Very high values can increase retries, memory pressure, and shutdown latency

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
neuracore data-daemon profile update <name> [--storage-limit <bytes|unit>] [--bandwidth-limit <bytes|unit>] [--storage-path <path>] [--num-threads <n>] [--max-concurrent-uploads <n>] [--wakelock|--no-wakelock] [--offline|--online] [--api-key <key>] [--current-org-id <org_id>]
```

Update the default profile:

```bash
neuracore data-daemon profile update [--storage-limit <bytes|unit>] [--bandwidth-limit <bytes|unit>] [--storage-path <path>] [--num-threads <n>] [--max-concurrent-uploads <n>] [--wakelock|--no-wakelock] [--offline|--online] [--api-key <key>] [--current-org-id <org_id>]
```

Example:

```bash
neuracore data-daemon profile update laptop --storage-limit 2gb --offline
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
neuracore data-daemon launch [--profile <name>] [--background]
```

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

The recording encoder selects backend at runtime:
- Uses `ffmpeg` CLI when `ffmpeg` is available on `PATH`
- Falls back to PyAV when `ffmpeg` is unavailable or fails to initialize

Quick check:

```bash
ffmpeg -version
```

If this command succeeds, the daemon will use the FFmpeg backend for new recordings.

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
