# Command Line Tools

Neuracore provides several command-line utilities:

To get all available commands:
```bash
neuracore --help
```

## Authentication
```bash
# Interactive login to save API key
neuracore login
```

Use the `--email` and `--password` option if you wish to login non-interactively.

## Organization Management
```bash
# Select your current organization
neuracore select-org
```

Use the `--org-name` option if you wish to select the org non-interactively.

## Server Operations
```bash
# Launch local policy server for inference
neuracore launch-server --job_id <job_id> --org_id <org_id> [--host <host>] [--port <port>]

# Example:
neuracore launch-server --job_id my_job_123 --org_id my_org_456 --host 0.0.0.0 --port 8080
```

**Parameters:**
- `--job_id`: Required. The job ID to run
- `--org_id`: Required. Your organization ID
- `--host`: Optional. Host address (default: 0.0.0.0)
- `--port`: Optional. Port number (default: 8080)

## Algorithm Validation

```bash
# Validate custom algorithms before upload
neuracore-validate /path/to/your/algorithm
```