# Environment Variables

Configure Neuracore behavior with environment variables (case insensitive, prefixed with `NEURACORE_`):

| Variable                                     | Function                                               | Valid Values   | Default                                                                 |
| -------------------------------------------- | ------------------------------------------------------ | -------------- | ----------------------------------------------------------------------- |
| `NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED` | Allow remote recording triggers                        | `true`/`false` | `true`                                                                  |
| `NEURACORE_PROVIDE_LIVE_DATA`                | Enable live data streaming from this node              | `true`/`false` | `true`                                                                  |
| `NEURACORE_CONSUME_LIVE_DATA`                | Enable live data consumption for inference             | `true`/`false` | `true`                                                                  |
| `NEURACORE_API_URL`                          | Base URL for Neuracore platform                        | URL string     | `https://api.neuracore.com/api`                                         |
| `NEURACORE_API_KEY`                          | An override to the api-key to access the neuracore     | `nrc_XXXX`     | Configured with the [`neuracore login`](#authentication) command        |
| `NEURACORE_ORG_ID`                           | An override to select the organization to use.         | A valid UUID   | Configured with the [`neuracore select-org`](#organization-management) command |
| `TMPDIR`                                     | Specifies a directory used for storing temporary files | Filepath       | An appropriate folder for your system                                   |
