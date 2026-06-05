from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
import yaml

from station.hf_detector import DEFAULT_MODEL_ID, HFDetector
from tools.stream_hf_dataset import format_detection_log, mono_to_simulated_channels, resample_mono
from tools.stream_local_dataset import (
    REPORT_FIELDNAMES,
    _report_rows,
    apply_local_eval_guards,
    build_event,
    detector_state_from_args,
    iter_audio_windows_with_padding,
    read_audio_mono,
)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def station_args_from_config(config_path: str | Path, *, window_sec: float | None = None) -> argparse.Namespace:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    audio_cfg = cfg.get("audio", {})
    station_cfg = cfg.get("station", {})
    return argparse.Namespace(
        station_id=str(station_cfg.get("station_id") or "manifest_offline_station"),
        channels=int(audio_cfg.get("channels", 1)),
        sample_rate=int(audio_cfg.get("sample_rate", 44100)),
        window_sec=float(window_sec if window_sec is not None else audio_cfg.get("window_sec", 1.0)),
        reset_state_per_file=True,
        skip_tail_padding=True,
        eval_mode=True,
    )


def enrich_event_metadata(event, manifest_row: dict[str, Any], window_idx: int, window_rms: float, padding_ratio: float) -> None:
    metadata = dict(event.metadata or {})
    for key, value in manifest_row.items():
        metadata[key] = value
    metadata["source"] = "manifest_offline_dataset"
    metadata["window_idx"] = window_idx
    metadata["window_rms"] = window_rms
    metadata["padding_ratio"] = padding_ratio
    event.metadata = metadata


def run_manifest(args: argparse.Namespace) -> list:
    station_args = station_args_from_config(args.config, window_sec=args.window_sec)
    rows = load_manifest(args.manifest)
    if args.dataset:
        wanted = set(args.dataset)
        rows = [row for row in rows if row.get("dataset_id") in wanted]
    if args.max_files is not None:
        rows = rows[: int(args.max_files)]

    hf_detector = HFDetector(model_id=args.model_id) if args.hf else None
    events = []
    for row in rows:
        path = Path(row.get("audio_path") or row.get("file_path") or row.get("wav_path") or "")
        if not path.exists():
            print(f"[WARN] missing audio: {path}")
            continue
        detector_state = detector_state_from_args(station_args)
        mono, source_sr = read_audio_mono(path)
        mono = resample_mono(mono, source_sr, station_args.sample_rate)
        emitted = 0
        for window_idx, (window, padding_ratio) in enumerate(
            iter_audio_windows_with_padding(mono, station_args.sample_rate, station_args.window_sec, skip_tail_padding=True)
        ):
            if args.max_windows is not None and emitted >= int(args.max_windows):
                break
            hf_result = hf_detector.predict(window, station_args.sample_rate) if hf_detector is not None else None
            hf_p = getattr(hf_result, "p_drone", None) if hf_result is not None else None
            audio = mono_to_simulated_channels(window, station_args.channels)
            timestamp = time.time()
            event = build_event(
                station_id=station_args.station_id,
                root=path.parent,
                file_path=path,
                label=str(row.get("label") or row.get("label_from_markers") or "unknown"),
                distance_category=row.get("distance_category"),
                audio=audio,
                sample_rate=station_args.sample_rate,
                timestamp=timestamp,
                detector_state=detector_state,
                hf_result=hf_result,
                hf_p_drone=hf_p,
            )
            event = apply_local_eval_guards(event)
            enrich_event_metadata(event, row, window_idx, float((window * window).mean() ** 0.5), float(padding_ratio))
            events.append(event)
            emitted += 1
            print(format_detection_log(f"{path.name} {row.get('label')}", event))
            if args.server:
                requests.post(args.server, json=event.model_dump(mode="json"), timeout=2.0)
    if args.save_report:
        write_manifest_report(args.save_report, events)
    return events


def write_manifest_report(path: Path, events: list) -> None:
    manifest_fields: list[str] = []
    for event in events:
        for key in (event.metadata or {}).keys():
            if key not in REPORT_FIELDNAMES and key not in manifest_fields:
                manifest_fields.append(key)
    fieldnames = list(REPORT_FIELDNAMES) + manifest_fields
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for base, event in zip(_report_rows(events), events):
            row = dict(event.metadata or {})
            row.update(base)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SkyEar station detector offline over a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/config_station.yaml"))
    parser.add_argument("--mode", choices=["offline"], default="offline")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--save-report", type=Path)
    parser.add_argument("--server")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    return parser.parse_args()


def main() -> None:
    run_manifest(parse_args())


def run_single_wav(
    *,
    wav: Path,
    config: Path,
    label: str = "unknown",
    save_report: Path | None = None,
    window_sec: float = 1.0,
) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "single_manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_path", "file_path", "dataset_id", "source_dataset", "label"])
            writer.writeheader()
            writer.writerow(
                {
                    "audio_path": str(wav),
                    "file_path": str(wav),
                    "dataset_id": "single_wav",
                    "source_dataset": "single_wav",
                    "label": label,
                }
            )
        args = argparse.Namespace(
            manifest=manifest,
            config=config,
            mode="offline",
            window_sec=window_sec,
            save_report=save_report,
            server=None,
            dataset=[],
            max_files=None,
            max_windows=None,
            hf=False,
            model_id=DEFAULT_MODEL_ID,
        )
        return run_manifest(args)


if __name__ == "__main__":
    main()
