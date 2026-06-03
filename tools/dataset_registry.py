from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REGISTRY = Path("data/dataset_registry.yaml")
REQUIRED_FIELDS = {
    "dataset_id",
    "display_name",
    "source_type",
    "source_ref",
    "download_method",
    "local_dir",
    "license",
    "label_schema",
    "label_mapping",
    "metadata_parser",
    "expected_audio_format",
    "source_priority",
    "use_for",
    "safety_notes",
    "status",
}
VALID_STATUSES = {"active", "needs_license_review", "manual_download_required", "deprecated"}


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    elif path == DEFAULT_REGISTRY:
        text = resources.files("data").joinpath("dataset_registry.yaml").read_text(encoding="utf-8")
    else:
        raise FileNotFoundError(path)
    payload = yaml.safe_load(text) or {}
    payload.setdefault("datasets", [])
    return payload


def list_datasets(path: str | Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    return list(load_registry(path).get("datasets", []))


def dataset_by_id(dataset_id: str, path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    for dataset in list_datasets(path):
        if dataset.get("dataset_id") == dataset_id:
            return dataset
    raise KeyError(f"dataset not found: {dataset_id}")


def validate_registry(path: str | Path = DEFAULT_REGISTRY) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for idx, dataset in enumerate(list_datasets(path)):
        prefix = str(dataset.get("dataset_id") or f"entry[{idx}]")
        missing = sorted(field for field in REQUIRED_FIELDS if field not in dataset)
        for field in missing:
            errors.append(f"{prefix}: missing {field}")
        dataset_id = str(dataset.get("dataset_id") or "")
        if dataset_id in seen:
            errors.append(f"{prefix}: duplicate dataset_id")
        if dataset_id:
            seen.add(dataset_id)
        if dataset.get("status") not in VALID_STATUSES:
            errors.append(f"{prefix}: invalid status {dataset.get('status')!r}")
        if not isinstance(dataset.get("use_for"), list):
            errors.append(f"{prefix}: use_for must be a list")
    return errors


def resolve_local_dir(dataset: dict[str, Any], base_dir: str | Path | None = None) -> Path:
    path = Path(str(dataset.get("local_dir") or ""))
    if path.is_absolute() or base_dir is None:
        return path
    return Path(base_dir) / path


def print_dataset_summary(dataset: dict[str, Any]) -> str:
    use_for = ", ".join(str(item) for item in dataset.get("use_for", []))
    return (
        f"{dataset.get('dataset_id')} | {dataset.get('display_name')} | "
        f"{dataset.get('source_type')}:{dataset.get('source_ref')} | "
        f"status={dataset.get('status')} | license={dataset.get('license')} | use_for={use_for}"
    )


def license_status(dataset: dict[str, Any]) -> str:
    status = str(dataset.get("status") or "unknown")
    license_name = str(dataset.get("license") or "unknown")
    if status in {"needs_license_review", "manual_download_required"} or license_name == "unknown":
        return "needs_review"
    return "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="skyear-datasets", description="Inspect the SkyEar dataset registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("validate")
    info = subparsers.add_parser("info")
    info.add_argument("dataset_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        for dataset in sorted(list_datasets(args.registry), key=lambda item: int(item.get("source_priority") or 0)):
            print(print_dataset_summary(dataset))
        return
    if args.command == "validate":
        errors = validate_registry(args.registry)
        if errors:
            for error in errors:
                print(f"FAILED: {error}")
            raise SystemExit(1)
        print("registry OK")
        return
    if args.command == "info":
        dataset = dataset_by_id(args.dataset_id, args.registry)
        print(yaml.safe_dump(dataset, sort_keys=False).strip())


if __name__ == "__main__":
    main()
