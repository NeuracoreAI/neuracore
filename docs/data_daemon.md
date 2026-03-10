# Neuracore Data Daemon

The Neuracore Data Daemon is a small background service that runs on your machine and takes care of storing recordings locally and uploading them.

You can use it in two ways:
- **CLI first**: launch the daemon, then run your scripts
- **Script first**: run your script and let it start the daemon automatically

Profiles are optional. If you do not use a profile, the daemon runs with built in defaults (and any environment variable overrides you set).

---

## What this README covers

- How to run the daemon (CLI or from a script)
- How profiles work (optional) and where they are stored
- The configuration fields you can set
- The order of precedence (defaults, profile, environment variables, CLI)
- A full CLI reference for the commands currently in use

It does not explain internal implementation details.

---

## Quick start

### 1) Install (from repo root)

```bash
pip install -e .
```

### 2) Launch the daemon

With defaults (no profile):

```bash
nc-data-daemon launch
```

With a profile:

```bash
nc-data-daemon profile create --name recording
nc-data-daemon profile update --name recording --storage-limit 2gb --bandwidth-limit 50mb --storage-path /data/records --num-threads 4
nc-data-daemon launch --profile recording
```

Background (runs quietly):

```bash
nc-data-daemon launch --profile recording --background
```

### 3) Check status and stop

```bash
nc-data-daemon status
nc-data-daemon stop
```

---

## Run your script without launching the daemon first

You do not have to use `nc-data-daemon launch` beforehand. The daemon will automatically start in the background if it is not already running when your script needs it.

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
nc-data-daemon launch
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

---

## Configuration

### Profiles (optional)

A profile is a YAML file that stores daemon settings you want to reuse.

Profiles are stored here:

```text
~/.neuracore/data_daemon/profiles/<name>.yaml
```

Manage profiles with:

```bash
nc-data-daemon profile create --name <name>
nc-data-daemon profile update --name <name> [options...]
nc-data-daemon profile show --name <name>
nc-data-daemon list-profiles
```

If you do not use a profile, the daemon runs with defaults.

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
- 1, true, yes, y (case insensitive)

Examples:

```bash
export NCD_STORAGE_LIMIT=3gb
export NCD_OFFLINE=true
nc-data-daemon launch
```

```bash
export NCD_PATH_TO_STORE_RECORD=/mnt/data/records
export NCD_NUM_THREADS=4
nc-data-daemon launch --background
```

---

## CLI reference

### nc-data-daemon profile create

```bash
nc-data-daemon profile create --name <name>
```

Example:

```bash
nc-data-daemon profile create --name laptop
```

### nc-data-daemon profile update

```bash
nc-data-daemon profile update --name <name>   [--storage-limit <bytes|unit>]   [--bandwidth-limit <bytes|unit>]   [--storage-path <path>]   [--num-threads <n>]   [--keep-wakelock-while-upload]   [--offline]   [--api-key <key>]   [--current-org-id <org_id>]
```

Example:

```bash
nc-data-daemon profile update --name laptop --storage-limit 2gb --offline
```

### nc-data-daemon profile show

```bash
nc-data-daemon profile show --name <name>
```

Example:

```bash
nc-data-daemon profile show --name laptop
```

### nc-data-daemon list-profiles

```bash
nc-data-daemon list-profiles
```

### nc-data-daemon launch

```bash
nc-data-daemon launch [--profile <name>] [--background]
```

Examples:

```bash
nc-data-daemon launch
```

```bash
nc-data-daemon launch --profile laptop
```

```bash
nc-data-daemon launch --profile laptop --background
```

### nc-data-daemon status

```bash
nc-data-daemon status
```

### nc-data-daemon stop

```bash
nc-data-daemon stop
```

### nc-data-daemon update

Resolve and print the effective configuration using:
- built in defaults
- the selected profile (if any)
- environment variables
- any CLI values you pass to this command

```bash
nc-data-daemon update   [--storage-limit <bytes|unit>]   [--bandwidth-limit <bytes|unit>]   [--storage-path <path>]   [--num-threads <n>]   [--keep-wakelock-while-upload]   [--offline]   [--api-key <key>]   [--current-org-id <org_id>]
```

---

## Troubleshooting

### Daemon already running
You tried to launch it while it is already running.

Try:

```bash
nc-data-daemon status
nc-data-daemon stop
nc-data-daemon launch
```

### Daemon failed to start
Run it in the foreground so you can see the output:

```bash
nc-data-daemon launch
```

If it still fails, check your profiles:

```bash
nc-data-daemon list-profiles
nc-data-daemon profile show --name <name>
```