from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.dataset_registry import DEFAULT_REGISTRY, available_dataset_ids_and_aliases, dataset_by_id, list_datasets, resolve_local_dir


def _github_url(source_ref: str) -> str:
    if source_ref.startswith("http://") or source_ref.startswith("https://"):
        return source_ref
    return f"https://github.com/{source_ref}.git"


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _json_safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key in {"array", "bytes"}:
                continue
            cleaned[key] = _json_safe_metadata(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        if len(value) > 20:
            return f"<list len={len(value)}>"
        return [_json_safe_metadata(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return f"<array shape={tuple(value.shape)} dtype={value.dtype}>"
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _write_jsonl_rows(rows: Any, path: Path, max_examples: int | None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if max_examples is not None and count >= max_examples:
                break
            handle.write(json.dumps(_json_safe_metadata(row), sort_keys=True) + "\n")
            count += 1
    return count


def download_huggingface(
    dataset: dict[str, Any],
    target: Path,
    *,
    hf_config: str | None = None,
    split: str | None = None,
    max_examples: int | None = None,
    streaming_export: bool = False,
    force_large: bool = False,
    metadata_only: bool = False,
    dry_run: bool = False,
) -> None:
    source_ref = str(dataset["source_ref"])
    print(f"Hugging Face dataset: {source_ref} -> {target}")
    if not force_large and max_examples is None and not streaming_export and not metadata_only:
        raise SystemExit(
            "Refusing to materialize a full Hugging Face dataset without --force-large. "
            "Use --split and --max-examples for a small smoke test, or --metadata-only/--streaming-export."
        )
    if dry_run:
        print(
            "dry-run HF options:",
            f"config={hf_config}",
            f"split={split}",
            f"max_examples={max_examples}",
            f"streaming_export={streaming_export}",
            f"metadata_only={metadata_only}",
            f"force_large={force_large}",
        )
        return
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("Install optional dependency: pip install 'skyear[datasets]'") from exc
    target.mkdir(parents=True, exist_ok=True)
    load_kwargs: dict[str, Any] = {}
    if hf_config:
        load_kwargs["name"] = hf_config
    if split:
        load_kwargs["split"] = split
    if streaming_export:
        load_kwargs["streaming"] = True
    ds = load_dataset(source_ref, **load_kwargs)

    info = {
        "dataset_id": dataset.get("dataset_id"),
        "source_ref": source_ref,
        "hf_config": hf_config,
        "split": split,
        "max_examples": max_examples,
        "streaming_export": streaming_export,
        "metadata_only": metadata_only,
        "force_large": force_large,
    }
    (target / "download_options.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if streaming_export or metadata_only:
        if isinstance(ds, dict):
            rows = []
            for split_name, split_rows in ds.items():
                for row in split_rows:
                    item = dict(row)
                    item["_split"] = split_name
                    rows.append(item)
        else:
            rows = ds
        count = _write_jsonl_rows(rows, target / "metadata.jsonl", max_examples)
        print(f"wrote metadata rows: {count} -> {target / 'metadata.jsonl'}")
        return

    if max_examples is not None:
        if isinstance(ds, dict):
            raise SystemExit("Pass --split when using --max-examples for Hugging Face subset export.")
        count = min(int(max_examples), len(ds))
        subset = ds.select(range(count))
        subset.save_to_disk(str(target / "dataset"))
        print(f"saved HF subset examples={count} -> {target / 'dataset'}")
        return

    ds.save_to_disk(str(target / "dataset"))


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


def download_dataset(
    dataset: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
    hf_config: str | None = None,
    split: str | None = None,
    max_examples: int | None = None,
    streaming_export: bool = False,
    force_large: bool = False,
    metadata_only: bool = False,
) -> Path:
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
        download_huggingface(
            dataset,
            target,
            hf_config=hf_config,
            split=split,
            max_examples=max_examples,
            streaming_export=streaming_export,
            force_large=force_large,
            metadata_only=metadata_only,
            dry_run=dry_run,
        )
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
    parser.add_argument("--hf-config")
    parser.add_argument("--split")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--streaming-export", action="store_true")
    parser.add_argument("--force-large", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list_datasets(args.registry)
    if args.dataset:
        try:
            selected = [dataset_by_id(dataset_id, args.registry) for dataset_id in args.dataset]
        except KeyError as exc:
            raise SystemExit(str(exc).strip("'")) from None
    elif args.all:
        selected = datasets
    else:
        raise SystemExit("Pass --dataset DATASET_ID or --all\n" + available_dataset_ids_and_aliases(args.registry))
    for dataset in selected:
        download_dataset(
            dataset,
            force=args.force,
            dry_run=args.dry_run,
            hf_config=args.hf_config,
            split=args.split,
            max_examples=args.max_examples,
            streaming_export=args.streaming_export,
            force_large=args.force_large,
            metadata_only=args.metadata_only,
        )


if __name__ == "__main__":
    main()
