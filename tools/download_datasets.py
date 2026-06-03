from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.dataset_registry import DEFAULT_REGISTRY, dataset_by_id, list_datasets, resolve_local_dir


def _github_url(source_ref: str) -> str:
    if source_ref.startswith("http://") or source_ref.startswith("https://"):
        return source_ref
    return f"https://github.com/{source_ref}.git"


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def download_huggingface(dataset: dict[str, Any], target: Path, *, dry_run: bool = False) -> None:
    print(f"Hugging Face dataset: {dataset['source_ref']} -> {target}")
    if dry_run:
        return
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install optional dependency: pip install 'skyear[datasets]'") from exc
    target.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(str(dataset["source_ref"]))
    ds.save_to_disk(str(target))


def download_github(dataset: dict[str, Any], target: Path, *, dry_run: bool = False) -> None:
    cmd = ["git", "clone"]
    if dataset.get("download_method") == "git_clone_branch_develop":
        cmd.extend(["--branch", "develop"])
    cmd.extend(["--depth", "1", _github_url(str(dataset["source_ref"])), str(target)])
    _run(cmd, dry_run=dry_run)


def download_kaggle(dataset: dict[str, Any], target: Path, *, dry_run: bool = False) -> None:
    print(f"Kaggle dataset requires credentials: {dataset['source_ref']} -> {target}")
    if dry_run:
        return
    try:
        import kaggle  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install kaggle and configure credentials, or download manually.") from exc
    target.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_files(str(dataset["source_ref"]), path=str(target), unzip=True)


def download_dataset(dataset: dict[str, Any], *, force: bool = False, dry_run: bool = False) -> Path:
    target = resolve_local_dir(dataset)
    if target.exists() and not force:
        print(f"exists, skipping: {target}")
        return target
    if force and target.exists() and not dry_run:
        shutil.rmtree(target)
    method = str(dataset.get("download_method") or "")
    status = str(dataset.get("status") or "")
    if status == "manual_download_required" or method == "manual_download_required":
        print(f"manual download required for {dataset['dataset_id']}: {dataset.get('license_notes') or dataset.get('source_ref')}")
        return target
    if method.startswith("huggingface"):
        download_huggingface(dataset, target, dry_run=dry_run)
    elif method.startswith("git_clone"):
        download_github(dataset, target, dry_run=dry_run)
    elif method.startswith("kaggle"):
        download_kaggle(dataset, target, dry_run=dry_run)
    else:
        print(f"No adapter for {dataset['dataset_id']} download_method={method}; prepare {target} manually.")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or prepare datasets from the SkyEar registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list_datasets(args.registry)
    if args.dataset:
        selected = [dataset_by_id(dataset_id, args.registry) for dataset_id in args.dataset]
    elif args.all:
        selected = datasets
    else:
        raise SystemExit("Pass --dataset DATASET_ID or --all")
    for dataset in selected:
        download_dataset(dataset, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
