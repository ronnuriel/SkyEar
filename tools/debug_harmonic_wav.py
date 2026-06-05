from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import wavfile

from station.audio_capture import select_mono_channel
from station.harmonic import harmonic_score
from station.station_agent import _detector_config


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _float_audio(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.floating):
        return array.astype(np.float32)
    max_value = float(np.iinfo(array.dtype).max)
    return (array.astype(np.float32) / max(max_value, 1.0)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print SkyEar harmonic debug metrics for a local WAV.")
    parser.add_argument("wav_path")
    parser.add_argument("--config", default="configs/config_station.yaml")
    parser.add_argument("--mono-mode", default=None)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    det_cfg = cfg.get("detector", {}) or {}
    stability_cfg = cfg.get("stability", {}) or {}
    hf_cfg = cfg.get("hf", {}) or {}
    detection_cfg = cfg.get("detection", {}) or {}
    harmonic_cfg = cfg.get("harmonic", {}) or {}
    detector_config = _detector_config(det_cfg, stability_cfg, hf_cfg, detection_cfg, harmonic_cfg)

    sample_rate, data = wavfile.read(args.wav_path)
    audio = _float_audio(data)
    if args.max_seconds and args.max_seconds > 0:
        audio = audio[: int(float(args.max_seconds) * int(sample_rate))]

    channel_scores: list[float] = []
    if audio.ndim == 2 and audio.shape[1] > 1:
        for idx in range(audio.shape[1]):
            score, _, _ = harmonic_score(
                audio[:, idx],
                int(sample_rate),
                detector_config.f0_min,
                detector_config.f0_max,
                detector_config.max_freq,
                detector_config.min_harmonics,
                detector_config.harmonic_min_ridge_prominence_db,
            )
            channel_scores.append(float(score))

    mode = args.mono_mode or str((cfg.get("audio", {}) or {}).get("mono_mix_mode") or "strongest_harmonic")
    mono, selected_channel, channel_rms = select_mono_channel(audio, mode=mode, channel_scores=channel_scores)
    score, f0, details = harmonic_score(
        mono,
        int(sample_rate),
        detector_config.f0_min,
        detector_config.f0_max,
        detector_config.max_freq,
        detector_config.min_harmonics,
        detector_config.harmonic_min_ridge_prominence_db,
    )
    present = [item for item in details if not isinstance(item, dict) or item.get("present")]
    payload = {
        "wav_path": str(Path(args.wav_path)),
        "sample_rate": int(sample_rate),
        "channels": 1 if audio.ndim == 1 else int(audio.shape[1]),
        "mono_mode": mode,
        "selected_channel": selected_channel,
        "channel_rms": channel_rms,
        "channel_harmonic_score": channel_scores,
        "harmonic_score": float(score),
        "best_f0_hz": f0,
        "present_ridge_count": len(present),
        "details": details,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
