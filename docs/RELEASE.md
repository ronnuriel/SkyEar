# Release

## Versioning Policy

- `pyproject.toml` `project.version` must be bumped for package releases.
- Release tag format: `v0.2.0-field-alpha`
- Patch = bugfix/docs.
- Minor = new station capability.
- Major = breaking config/API.

## Preflight

```bash
skyear release preflight
```

This runs:

```text
git status --short
python -m compileall station server dashboard shared tools tests
pytest -q
bash scripts/release_field_alpha_check.sh
python -m build
```

The existing Field Alpha check remains available:

```bash
bash scripts/release_field_alpha_check.sh
```

## Tag

```bash
skyear release tag v0.2.0-field-alpha
```

The command verifies a clean worktree, verifies the `pyproject.toml` version matches the tag version, creates an annotated tag, and prints the push command.

Push automatically only when intended:

```bash
skyear release tag v0.2.0-field-alpha --push
```
