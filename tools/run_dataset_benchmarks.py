from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from station.hf_detector import DEFAULT_MODEL_ID
from tools.build_audio_manifest import FIELDNAMES, iter_manifest_rows, write_csv
from tools.combine_benchmark_summaries import combine_summaries, write_combined_outputs
from tools.dataset_registry import DEFAULT_REGISTRY, dataset_by_id, resolve_dataset_id, resolve_local_dir
from tools.stream_manifest_dataset import run_manifest
from tools.summarize_benchmark import load_report, summarize_report


def _dataset_values(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for item in args.dataset or []:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    for item in args.datasets or []:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    return values


def _root_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            dataset_id, path = value.split("=", 1)
            overrides[resolve_dataset_id(dataset_id.strip())] = Path(path.strip())
        else:
            overrides["__default__"] = Path(value)
    return overrides


def _write_empty_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()


def _manifest_for_dataset(dataset_id: str, args: argparse.Namespace, output_dir: Path, root_overrides: dict[str, Path]) -> tuple[Path, str]:
    resolved_id = resolve_dataset_id(dataset_id)
    dataset: dict[str, Any] | None = None
    root = root_overrides.get(resolved_id) or root_overrides.get(dataset_id) or root_overrides.get("__default__")
    if root is None:
        dataset = dataset_by_id(dataset_id, args.registry)
        resolved_id = str(dataset["dataset_id"])
        root = resolve_local_dir(dataset)
    else:
        dataset = {
            "dataset_id": resolved_id,
            "display_name": resolved_id,
            "license": "synthetic_or_local",
            "license_notes": "Provided through --root override.",
            "use_for": ["benchmark"],
            "metadata_parser": "generic",
            "label_mapping": {},
        }

    if not root.exists():
        raise FileNotFoundError(f"dataset root not found for {dataset_id}: {root}")
    rows = list(
        iter_manifest_rows(
            [root],
            dataset=dataset,
            verify_audio=bool(args.verify_audio),
            skip_audio_hash=True,
            max_files=args.max_files,
        )
    )
    manifest = output_dir / resolved_id / "manifest.csv"
    if rows:
        write_csv(manifest, rows)
    else:
        _write_empty_manifest(manifest)
    return manifest, resolved_id


def run_one_dataset(dataset_id: str, args: argparse.Namespace, root_overrides: dict[str, Path]) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    manifest, resolved_id = _manifest_for_dataset(dataset_id, args, output_dir, root_overrides)
    dataset_dir = output_dir / resolved_id
    report = dataset_dir / "eval.csv"
    summary_path = dataset_dir / "summary.json"
    stream_args = argparse.Namespace(
        manifest=manifest,
        config=args.config,
        mode="offline",
        window_sec=args.window_sec,
        save_report=report,
        server=None,
        dataset=[],
        max_files=None,
        max_windows=args.max_windows,
        hf=args.hf,
        model_id=args.model_id,
    )
    events = run_manifest(stream_args)
    summary = summarize_report(load_report(report)) if report.exists() else summarize_report([])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "dataset_id": resolved_id,
        "ok": True,
        "manifest": str(manifest),
        "report": str(report),
        "summary": str(summary_path),
        "events": len(events),
        "files": int((summary.get("overall") or {}).get("files") or 0),
    }


def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary_dir = output_dir / "all_benchmarks"
    run_summary_dir.mkdir(parents=True, exist_ok=True)
    root_overrides = _root_overrides(args.root or [])
    dataset_ids = _dataset_values(args)
    if not dataset_ids:
        raise SystemExit("Pass at least one --dataset or --datasets value.")

    results = []
    for dataset_id in dataset_ids:
        try:
            result = run_one_dataset(dataset_id, args, root_overrides)
            print(f"[OK] {dataset_id}: files={result['files']} events={result['events']}")
        except Exception as exc:
            result = {"dataset_id": dataset_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[FAILED] {dataset_id}: {result['error']}")
        results.append(result)

    combined_rows = combine_summaries(output_dir)
    write_combined_outputs(
        combined_rows,
        output_dir / "combined_summary.json",
        output_dir / "combined_summary.csv",
    )
    payload = {"output_dir": str(output_dir), "results": results}
    (run_summary_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run_summary={run_summary_dir / 'run_summary.json'}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkyEar offline benchmarks across one or more datasets.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--datasets", action="append", default=[])
    parser.add_argument("--root", action="append", default=[], help="Override dataset root. Use DATASET_ID=PATH or PATH.")
    parser.add_argument("--config", type=Path, default=Path("configs/config_station.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--verify-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_benchmarks(parse_args())


if __name__ == "__main__":
    main()
