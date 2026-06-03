# SkyEar Field Alpha Release Checklist

Release name: `v0.1.0-field-alpha`

SkyEar Field Alpha is an engineering dry-run release for passive acoustic warning, visual confirmation, dataset benchmarking, and field-test data collection. It is not an operational deployment.

## Preflight

Run from the repository root.

1. Check the worktree is clean:

```bash
git status --short
```

2. Check raw audio is not tracked:

```bash
git ls-files | grep -Ei '\.(wav|mp3|flac|m4a|ogg)$' && exit 1 || true
```

3. Check raw dataset storage remains ignored:

```bash
git check-ignore data/datasets/example.wav
```

`runtime/` and `reports/` can exist locally during development. They are not required for the release tag. Do not commit raw recordings or downloaded datasets.

## Release Checks

Run the one-command release check:

```bash
bash scripts/release_field_alpha_check.sh
```

That script runs:

```bash
python -m pip install build
python -m build
PYTHONPATH=. pytest -q
bash scripts/release_smoke_test.sh
bash scripts/map_smoke_test.sh
bash scripts/dataset_hub_smoke_test.sh
```

If you already have a server running for the map smoke test, set:

```bash
export SKYEAR_BASE_URL=http://127.0.0.1:8080
```

Otherwise the release script starts a temporary local server for the map smoke test.

## Tag

After the release check passes and the worktree contains only intended release files:

```bash
git tag -a v0.1.0-field-alpha -m "SkyEar Field Alpha v0.1.0"
git push origin v0.1.0-field-alpha
```

## Install From Tag

Base install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "skyear @ git+https://github.com/ronnuriel/SkyEar.git@v0.1.0-field-alpha"
```

Install with Hugging Face and dataset tooling:

```bash
pip install "skyear[all] @ git+https://github.com/ronnuriel/SkyEar.git@v0.1.0-field-alpha"
```

Copy configs and verify:

```bash
skyear-copy-configs configs
skyear-check-server --url http://127.0.0.1:8080
```
