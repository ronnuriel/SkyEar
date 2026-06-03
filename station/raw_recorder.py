from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


class RawRingBufferRecorder:
    def __init__(
        self,
        directory: str | Path,
        sample_rate: int,
        channels: int,
        buffer_seconds: float = 20.0,
        cooldown_seconds: float = 5.0,
    ):
        self.directory = Path(directory)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.buffer_seconds = float(buffer_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self._chunks: deque[np.ndarray] = deque()
        self._samples = 0
        self._max_samples = max(1, int(self.sample_rate * self.buffer_seconds))
        self._last_saved_at = 0.0

    def append(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        if chunk.ndim != 2:
            raise ValueError(f"audio must be 1D or 2D, got shape {chunk.shape}")
        self._chunks.append(chunk.copy())
        self._samples += chunk.shape[0]
        while self._samples > self._max_samples and self._chunks:
            removed = self._chunks.popleft()
            self._samples -= removed.shape[0]

    def should_save(self, now: float | None = None) -> bool:
        now = time.time() if now is None else float(now)
        return now - self._last_saved_at >= self.cooldown_seconds

    def save_candidate(self, station_id: str, metadata: dict[str, Any], now: float | None = None) -> tuple[Path, Path] | None:
        now = time.time() if now is None else float(now)
        if not self.should_save(now) or not self._chunks:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        stem = f"{station_id}_{stamp}_{int((now % 1) * 1000):03d}"
        wav_path = self.directory / f"{stem}.wav"
        json_path = self.directory / f"{stem}.json"
        audio = np.concatenate(list(self._chunks), axis=0)
        _write_wav(wav_path, audio, self.sample_rate)
        sidecar = dict(metadata)
        sidecar.update(
            {
                "station_id": station_id,
                "sample_rate": self.sample_rate,
                "channels": int(audio.shape[1]),
                "duration_sec": float(audio.shape[0] / self.sample_rate),
                "wav_path": str(wav_path),
                "saved_unix": now,
            }
        )
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
        self._last_saved_at = now
        return wav_path, json_path


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf

        sf.write(str(path), audio, int(sample_rate))
        return
    except Exception:
        from scipy.io import wavfile

        clipped = np.clip(audio, -1.0, 1.0)
        wavfile.write(str(path), int(sample_rate), (clipped * 32767.0).astype(np.int16))
