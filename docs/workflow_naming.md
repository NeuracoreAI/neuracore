# Workflow Naming Convention

## Integration Workflows

Integration workflow `name:` fields follow this pattern:

```
Integration <Category> - <Test Name> (<Environment>)
```

| Part | Values | Example |
|---|---|---|
| Category | `ML` or `PF` | `ML` |
| Test Name | Short description of what is tested | `Resume Training` |
| Environment | `Production` or `Staging` | `(Staging)` |

### Examples

```
Integration ML - Resume Training (Production)
Integration ML - Algorithm Validation (Staging)
Integration PF - Data Flow (Production)
Integration PF - Importer (Staging)
```

### Category Definitions

- **ML** — tests ML training, inference, or algorithm behaviour (e.g. training flow, auto batch size, back-to-back training)
- **PF** — tests client/SDK functionality against the live API (e.g. data daemon, stream consumption, dataset importer)

### File Naming

Workflow files use kebab-case with environment suffix:

```
integration-<category>-<test-name>-<prod|staging>.yaml
```

Example: `integration-ml-resume-training-prod.yaml`
