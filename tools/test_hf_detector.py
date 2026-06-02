from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from station.hf_detector import DEFAULT_MODEL_ID, HFDetectionResult, HFDetector


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("soundfile is required to read WAV files") from exc

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optional Hugging Face drone detector on a WAV file.")
    parser.add_argument("--wav", required=True, type=Path, help="Path to a WAV file.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model ID.")
    parser.add_argument("--fallback-drone-label-idx", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = HFDetector(
        model_id=args.model_id,
        fallback_drone_label_idx=args.fallback_drone_label_idx,
        threshold=args.threshold,
    )

    try:
        audio, sample_rate = _read_wav(args.wav)
        result = detector.predict(audio, sample_rate)
    except Exception as exc:
        result = HFDetectionResult(error=str(exc))

    print(f"model loaded: {'yes' if detector.model_loaded else 'no'}")
    print(f"p_drone: {result.p_drone}")
    print(f"label: {result.label}")
    print(f"class_probs: {json.dumps(result.class_probs, sort_keys=True)}")
    if result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    main()
