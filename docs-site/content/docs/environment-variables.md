---
title: Environment Variables
weight: 5
---

Configure Neuracore behavior with environment variables. All variables are case insensitive and prefixed with `NEURACORE_`.

## Available Variables

| Variable | Function | Valid Values | Default |
|----------|----------|--------------|---------|
| `NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED` | Allow remote recording triggers | `true`/`false` | `true` |
| `NEURACORE_PROVIDE_LIVE_DATA` | Enable live data streaming from this node | `true`/`false` | `true` |
| `NEURACORE_CONSUME_LIVE_DATA` | Enable live data consumption for inference | `true`/`false` | `true` |
| `NEURACORE_API_URL` | Base URL for Neuracore platform | URL string | `https://api.neuracore.com/api` |
| `NEURACORE_API_KEY` | An override to the API key to access Neuracore | `nrc_XXXX` | Configured with `neuracore login` |
| `NEURACORE_ORG_ID` | An override to select the organization to use | A valid UUID | Configured with `neuracore select-org` |
| `TMPDIR` | Specifies a directory used for storing temporary files | Filepath | System default |

## Usage Examples

### Set API URL for Staging Environment

```bash
export NEURACORE_API_URL=https://staging.api.neuracore.com/api
```

### Disable Live Data Streaming

```bash
export NEURACORE_PROVIDE_LIVE_DATA=false
python my_robot_script.py
```

### Use a Specific API Key

```bash
export NEURACORE_API_KEY=nrc_your_api_key_here
python my_robot_script.py
```

### Override Organization

```bash
export NEURACORE_ORG_ID=your-org-uuid-here
python my_robot_script.py
```

## Configuration File

Most settings can also be configured via the config file at `~/.neuracore/config.json`, which is created when you run:

```bash
neuracore login
neuracore select-org --org-name "<ORG_NAME>"
```

Environment variables take precedence over the config file.
