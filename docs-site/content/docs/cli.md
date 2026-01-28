---
title: Command Line Interface
weight: 3
---

The `neuracore` command helps you authenticate, choose an organization, monitor training runs (cloud and local), and launch a local policy server for quick validation.

## Configure Once

```bash
# Login and save an API key
neuracore login

# Non-interactive login
neuracore login --email you@example.com --password '<PASSWORD>'

# Select your organization
neuracore select-org --org-name "<ORG_NAME>"
```

{{< callout type="info" >}}
Config is stored at `~/.neuracore/config.json`. Override the backend with `NEURACORE_API_URL=https://staging.api.neuracore.com/api` if needed.
{{< /callout >}}

## Quick Start: Sign In → Monitor → Inspect → Serve

{{< steps >}}

### Sign in and pick your org

Run this once per machine; re-run if you switch orgs.

```bash
neuracore login
neuracore select-org --org-name "<ORG_NAME>"
```

### List recent training runs

Cloud by default:

```bash
neuracore training list --cloud --limit 5
# Expected: a table plus plain-text rows like:
# run-2024-08-12 | Yes | diffusion_policy | <DATASET_ID>
```

### Inspect a run

```bash
neuracore training inspect --training-name <RUN_NAME_OR_ID>
# Add --config to print algorithm config, or --json for raw JSON.
```

### Launch a local policy server

Sanity-check inference:

```bash
neuracore launch-server \
  --model_input_order '{"RGB_IMAGES": ["front_cam"]}' \
  --model_output_order '{"JOINT_TARGET_POSITIONS": ["arm"]}' \
  --job_id <RUN_ID> \
  --org_id <ORG_ID> \
  --port 8080
```

### Clean up

```bash
neuracore training delete --training-name <RUN_NAME_OR_ID> --yes

# Delete a local run directory instead:
neuracore training delete --local --root <RUNS_DIR> --training-name <RUN_NAME> --yes
```

{{< /steps >}}

## Core Commands Reference

| Command | Description |
|---------|-------------|
| `neuracore login [--email <EMAIL>] [--password <PASSWORD>]` | Saves your API key to `~/.neuracore/config.json` |
| `neuracore select-org [--org-name <NAME> \| --org-id <ID>]` | Sets the active organization |
| `neuracore training list [--cloud \| --local \| --all] [--status <STATUS>] [--limit <N>] [--root <DIR>]` | List training runs. Default is both cloud and local (`~/.neuracore/training/runs`). Status: `PENDING`, `RUNNING`, `COMPLETED` |
| `neuracore training inspect --training-name <RUN> [--cloud \| --local] [--root <DIR>] [--config] [--json]` | Cloud is default; add `--local` to read `training_run.json` from a local run directory |
| `neuracore training delete --training-name <RUN> [--cloud \| --local] [--root <DIR>] [--yes]` | Cloud is default; add `--local` to remove a run directory under `--root`. Prompts unless `--yes` |
| `neuracore training start` | Placeholder; use the SDK to launch training and return here to monitor |
| `neuracore launch-server --model_input_order '<JSON>' --model_output_order '<JSON>' [--job_id <RUN_ID>] [--org_id <ORG_ID>] [--host <HOST>] [--port <PORT>]` | JSON must map DataType strings to ordered name lists |

## Troubleshooting

{{< callout type="warning" title="Authentication failed" >}}
Re-run `neuracore login`; confirm `~/.neuracore/config.json` exists and contains an API key.
{{< /callout >}}

**No runs show up but you know they exist:**
- Ensure you selected the right org (`neuracore select-org`) and used the correct `--status` filter

**Invalid status filter when listing runs:**
- Use one of `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`

**Local inspect fails with missing `training_run.json`:**
- Point `--root` to the directory where your Hydra runs live (default `~/.neuracore/training/runs`)

**`launch-server` JSON errors:**
- Ensure the JSON strings use double quotes and DataType keys match your model (`"RGB_IMAGES"`, `"JOINT_TARGET_POSITIONS"`, etc.)

## FAQ

| Question | Answer |
|----------|--------|
| Where is CLI config stored? | `~/.neuracore/config.json` (API key and current org) |
| How do I switch API environments? | Set `NEURACORE_API_URL` before running commands |
| Where are local runs written? | `~/.neuracore/training/runs` (override with `--root`) |
| Can the CLI upload datasets or start training? | Not yet. Use the Python API; `neuracore training start` is a placeholder |
