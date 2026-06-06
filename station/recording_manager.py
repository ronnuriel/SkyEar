from __future__ import annotations

import csv
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MARKER_FIELDS = [
    "session_id",
    "chunk_index_expected",
    "source",
    "timestamp_unix",
    "offset_sec",
    "label",
    "note",
    "distance_m",
    "bearing_deg",
    "drone_model",
    "current_wav_path",
]


@dataclass
class RecordingConfig:
    enabled: bool = True
    root: str = "runtime/recordings"
    chunk_sec: float = 60.0
    format: str = "wav"
    auto_record_on_candidate: bool = False
    pre_roll_sec: float = 10.0
    max_session_sec: float = 3600.0
    max_disk_gb: float = 20.0


class RecordingManager:
    def __init__(
        self,
        *,
        station_id: str,
        sample_rate: int,
        channels: int,
        config: RecordingConfig | dict[str, Any] | None = None,
        station_config: dict[str, Any] | None = None,
    ):
        if isinstance(config, dict) or config is None:
            cfg = config or {}
            config = RecordingConfig(
                enabled=bool(cfg.get("enabled", True)),
                root=str(cfg.get("root", "runtime/recordings")),
                chunk_sec=float(cfg.get("chunk_sec", 60.0)),
                format=str(cfg.get("format", "wav")),
                auto_record_on_candidate=bool(cfg.get("auto_record_on_candidate", False)),
                pre_roll_sec=float(cfg.get("pre_roll_sec", 10.0)),
                max_session_sec=float(cfg.get("max_session_sec", 3600.0)),
                max_disk_gb=float(cfg.get("max_disk_gb", 20.0)),
            )
        self.station_id = str(station_id)
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.config = config
        self.station_config = station_config or {}
        self.root = Path(config.root)
        self._lock = threading.RLock()
        self._recording = False
        self._session_id: str | None = None
        self._session_name: str | None = None
        self._session_dir: Path | None = None
        self._started_unix: float | None = None
        self._stopped_unix: float | None = None
        self._label: str | None = None
        self._note: str | None = None
        self._chunk_index = 0
        self._chunk_audio: list[tuple[np.ndarray, float]] = []
        self._chunk_samples = 0
        self._current_file: Path | None = None
        self._wav_files: list[dict[str, Any]] = []
        self._last_error: str | None = None
        self._last_sample_end_unix: float | None = None
        self._discontinuities: list[dict[str, float]] = []
        self._overflow_timestamps: list[float] = []
        self._recording_blocks_written = 0
        self._marker_count = 0

    def start_recording(self, session_name: str | None = None, label: str | None = None, note: str | None = None) -> dict[str, Any]:
        with self._lock:
            if not self.config.enabled:
                self._last_error = "recording_disabled"
                return self.state()
            if self._recording:
                return self.state()
            if self._disk_usage_bytes(self.root) >= self._max_disk_bytes():
                self._last_error = "recording_disk_limit_reached"
                return self.state()
            now = time.time()
            safe_name = _safe_name(session_name or "session")
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
            self._session_id = f"{self.station_id}_{stamp}_{safe_name}"
            self._session_name = session_name or safe_name
            self._session_dir = self.root / self._session_id
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._started_unix = now
            self._stopped_unix = None
            self._label = label
            self._note = note
            self._chunk_index = 0
            self._chunk_audio = []
            self._chunk_samples = 0
            self._current_file = None
            self._wav_files = []
            self._last_error = None
            self._last_sample_end_unix = None
            self._discontinuities = []
            self._overflow_timestamps = []
            self._recording_blocks_written = 0
            self._marker_count = 0
            self._recording = True
            self._write_markers_header()
            self._write_station_config_snapshot()
            if label:
                self._append_marker_locked(
                    label=str(label),
                    note=note,
                    source="start",
                    timestamp_unix=now,
                    offset_sec=0.0,
                )
            self._write_metadata()
            return self.state()

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            if self._recording:
                self._flush_chunk_locked()
                self._stopped_unix = time.time()
                self._recording = False
                self._write_metadata()
            return self.state()

    def mark_event(
        self,
        label: str,
        note: str | None = None,
        distance_m: float | None = None,
        bearing_deg: float | None = None,
        drone_model: str | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        with self._lock:
            if self._session_dir is None:
                self._last_error = "no_active_recording_session"
                return self.state()
            now = time.time()
            offset = 0.0 if self._started_unix is None else max(0.0, now - self._started_unix)
            self._append_marker_locked(
                label=str(label or ""),
                note=note,
                distance_m=distance_m,
                bearing_deg=bearing_deg,
                drone_model=drone_model,
                source=source,
                timestamp_unix=now,
                offset_sec=offset,
            )
            self._write_metadata()
            return self.state()

    def record_overflow(self, timestamp_unix: float | None = None) -> None:
        with self._lock:
            if not self._recording or self._session_dir is None:
                return
            self._overflow_timestamps.append(time.time() if timestamp_unix is None else float(timestamp_unix))
            self._write_metadata()

    def append_audio(self, audio: np.ndarray, timestamp: float | None = None) -> None:
        with self._lock:
            if not self._recording or self._session_dir is None:
                return
            now = time.time() if timestamp is None else float(timestamp)
            if self._started_unix is not None and now - self._started_unix >= float(self.config.max_session_sec):
                self.stop_recording()
                return
            array = _as_channels(audio, self.channels)
            block_start = now
            block_end = block_start + float(array.shape[0]) / float(self.sample_rate)
            if self._last_sample_end_unix is not None:
                gap = float(block_start) - float(self._last_sample_end_unix)
                tolerance = max(0.01, 4.0 / float(self.sample_rate))
                if gap > tolerance:
                    self._discontinuities.append(
                        {
                            "gap_start_unix": float(self._last_sample_end_unix),
                            "gap_end_unix": float(block_start),
                            "missing_sec": float(gap),
                        }
                    )
            self._last_sample_end_unix = block_end
            self._chunk_audio.append((array.copy(), block_start))
            self._chunk_samples += int(array.shape[0])
            self._recording_blocks_written += 1
            while self._chunk_samples >= self._chunk_sample_limit():
                self._flush_chunk_locked(target_samples=self._chunk_sample_limit())

    def state(self) -> dict[str, Any]:
        with self._lock:
            duration = 0.0
            if self._started_unix is not None:
                end = time.time() if self._recording else (self._stopped_unix or time.time())
                duration = max(0.0, float(end) - float(self._started_unix))
            return {
                "enabled": bool(self.config.enabled),
                "recording": bool(self._recording),
                "session_id": self._session_id,
                "session_name": self._session_name,
                "session_dir": None if self._session_dir is None else str(self._session_dir),
                "duration_sec": duration,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "chunk_sec": float(self.config.chunk_sec),
                "current_file": None if self._current_file is None else str(self._current_file),
                "chunk_index": int(self._chunk_index),
                "disk_usage_bytes": self._disk_usage_bytes(self.root),
                "disk_usage_gb": self._disk_usage_bytes(self.root) / (1024.0**3),
                "max_disk_gb": float(self.config.max_disk_gb),
                "wav_files": list(self._wav_files),
                "wav_file_count": len(self._wav_files),
                "marker_count": int(self._marker_count),
                "recording_blocks_written": int(self._recording_blocks_written),
                "discontinuities": list(self._discontinuities),
                "overflow_count": len(self._overflow_timestamps),
                "overflow_timestamps": list(self._overflow_timestamps),
                "recording_continuity_ok": self._recording_continuity_ok_locked(),
                "last_error": self._last_error,
                "privacy_notice": "Recording may capture voices. Use only where permitted.",
            }

    def _chunk_sample_limit(self) -> int:
        return max(1, int(round(float(self.config.chunk_sec) * self.sample_rate)))

    def _max_disk_bytes(self) -> int:
        return max(1, int(float(self.config.max_disk_gb) * (1024**3)))

    def _flush_chunk_locked(self, target_samples: int | None = None) -> None:
        if not self._chunk_audio or self._session_dir is None:
            return
        target = self._chunk_samples if target_samples is None else int(target_samples)
        write_blocks = self._pop_chunk_blocks_locked(target)
        if not write_blocks:
            return
        write_audio = np.concatenate([block for block, _ in write_blocks], axis=0)
        chunk_start = float(write_blocks[0][1])
        last_audio, last_start = write_blocks[-1]
        chunk_end = float(last_start) + float(last_audio.shape[0]) / float(self.sample_rate)
        wav_path = self._session_dir / f"chunk_{self._chunk_index:04d}.wav"
        _write_wav(wav_path, write_audio, self.sample_rate)
        self._current_file = wav_path
        self._wav_files.append(
            {
                "wav_path": str(wav_path),
                "chunk_index": int(self._chunk_index),
                "start_time": chunk_start,
                "end_time": chunk_end,
                "duration_sec": float(write_audio.shape[0] / self.sample_rate),
                "sample_rate": self.sample_rate,
                "channels": int(write_audio.shape[1]),
            }
        )
        self._chunk_index += 1
        self._write_metadata()

    def _pop_chunk_blocks_locked(self, target_samples: int) -> list[tuple[np.ndarray, float]]:
        remaining_target = max(0, int(target_samples))
        write_blocks: list[tuple[np.ndarray, float]] = []
        new_blocks: list[tuple[np.ndarray, float]] = []
        for block, start_unix in self._chunk_audio:
            if remaining_target <= 0:
                new_blocks.append((block, start_unix))
                continue
            if block.shape[0] <= remaining_target:
                write_blocks.append((block, start_unix))
                remaining_target -= int(block.shape[0])
                continue
            write_blocks.append((block[:remaining_target], start_unix))
            new_start = float(start_unix) + float(remaining_target) / float(self.sample_rate)
            new_blocks.append((block[remaining_target:], new_start))
            remaining_target = 0
        self._chunk_audio = new_blocks
        self._chunk_samples = int(sum(block.shape[0] for block, _ in self._chunk_audio))
        return write_blocks

    def _append_marker_locked(
        self,
        *,
        label: str,
        note: str | None = None,
        distance_m: float | None = None,
        bearing_deg: float | None = None,
        drone_model: str | None = None,
        source: str = "manual",
        timestamp_unix: float,
        offset_sec: float,
    ) -> None:
        assert self._session_dir is not None
        chunk_index_expected = int(max(0.0, float(offset_sec)) // max(float(self.config.chunk_sec), 1e-6))
        row = {
            "session_id": str(self._session_id or ""),
            "chunk_index_expected": str(chunk_index_expected),
            "source": str(source or "manual"),
            "timestamp_unix": f"{float(timestamp_unix):.6f}",
            "offset_sec": f"{float(offset_sec):.3f}",
            "label": str(label or ""),
            "note": str(note or ""),
            "distance_m": "" if distance_m is None else f"{float(distance_m):.3f}",
            "bearing_deg": "" if bearing_deg is None else f"{float(bearing_deg) % 360.0:.3f}",
            "drone_model": str(drone_model or ""),
            "current_wav_path": "" if self._current_file is None else str(self._current_file),
        }
        with (self._session_dir / "markers.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MARKER_FIELDS)
            writer.writerow(row)
        self._marker_count += 1

    def _write_markers_header(self) -> None:
        assert self._session_dir is not None
        with (self._session_dir / "markers.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MARKER_FIELDS)
            writer.writeheader()

    def _write_station_config_snapshot(self) -> None:
        if self._session_dir is None:
            return
        (self._session_dir / "station_config_snapshot.json").write_text(
            json.dumps(self.station_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_metadata(self) -> None:
        if self._session_dir is None:
            return
        payload = {
            "station_id": self.station_id,
            "session_id": self._session_id,
            "session_name": self._session_name,
            "label": self._label,
            "note": self._note,
            "started_unix": self._started_unix,
            "stopped_unix": self._stopped_unix,
            "recording": self._recording,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_sec": float(self.config.chunk_sec),
            "format": self.config.format,
            "wav_files": self._wav_files,
            "marker_count": int(self._marker_count),
            "recording_blocks_written": int(self._recording_blocks_written),
            "discontinuities": self._discontinuities,
            "overflow_count": len(self._overflow_timestamps),
            "overflow_timestamps": self._overflow_timestamps,
            "recording_continuity_ok": self._recording_continuity_ok_locked(),
            "privacy_notice": "Recording may capture voices. Use only where permitted.",
        }
        (self._session_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _recording_continuity_ok_locked(self) -> bool:
        return not self._discontinuities and not self._overflow_timestamps and self._last_error is None

    @staticmethod
    def _disk_usage_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("._") or "session"


def _as_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"audio must be mono or 2D, got shape {array.shape}")
    if array.shape[1] != int(channels):
        raise ValueError(f"expected {channels} audio channels, got {array.shape[1]}")
    return array


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), audio, int(sample_rate))
        return
    except Exception:
        from scipy.io import wavfile

        clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        wavfile.write(str(path), int(sample_rate), (clipped * 32767.0).astype(np.int16))


def load_recording_state(session_dir: str | Path) -> dict[str, Any]:
    path = Path(session_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def available_recording_space_gb(root: str | Path) -> float:
    usage = shutil.disk_usage(Path(root).resolve().anchor or ".")
    return float(usage.free / (1024.0**3))
