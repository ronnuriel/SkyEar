from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass
from collections import deque
from typing import Any, Callable, Iterator, Optional

import numpy as np

def _sounddevice():
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sounddevice is required for audio capture. Install it with `pip install -r requirements.txt`."
        ) from exc
    return sd

def _default_input_device_id(sd) -> int | None:
    try:
        default_device = getattr(getattr(sd, "default", None), "device", None)
    except Exception:
        return None

    if default_device is None:
        return None
    if isinstance(default_device, (tuple, list)):
        candidate = default_device[0] if default_device else None
    elif hasattr(default_device, "input"):
        candidate = getattr(default_device, "input")
    else:
        try:
            candidate = default_device[0]
        except Exception:
            candidate = default_device

    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None

def list_input_devices():
    sd = _sounddevice()
    default_input = _default_input_device_id(sd)
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []
    out = []
    for idx, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            hostapi_index = d.get("hostapi")
            hostapi_name = None
            if hostapi_index is not None:
                try:
                    hostapi_name = hostapis[int(hostapi_index)].get("name")
                except Exception:
                    hostapi_name = None
            out.append({
                "id": idx,
                "name": d["name"],
                "max_input_channels": int(d["max_input_channels"]),
                "default_samplerate": int(d.get("default_samplerate", 44100)),
                "hostapi": hostapi_name,
                "is_default_input": default_input is not None and default_input == idx,
            })
    return out

@dataclass
class CapturedAudioBlock:
    audio: np.ndarray
    start_unix: float
    end_unix: float
    input_overflow: bool = False


def copy_audio_block(block: CapturedAudioBlock) -> CapturedAudioBlock:
    return CapturedAudioBlock(
        audio=np.asarray(block.audio, dtype=np.float32).copy(),
        start_unix=float(block.start_unix),
        end_unix=float(block.end_unix),
        input_overflow=bool(block.input_overflow),
    )


def audio_block_stream(
    device_id: Optional[int],
    sample_rate: int,
    channels: int,
    window_sec: float,
    latency: str | float | None = None,
) -> Iterator[CapturedAudioBlock]:
    sd = _sounddevice()
    block = int(sample_rate * window_sec)
    stream_kwargs: dict[str, Any] = {
        "device": device_id,
        "channels": channels,
        "samplerate": sample_rate,
        "blocksize": block,
        "dtype": "float32",
    }
    if latency not in (None, ""):
        stream_kwargs["latency"] = latency
    with sd.InputStream(**stream_kwargs) as stream:
        while True:
            audio, overflowed = stream.read(block)
            end_unix = time.time()
            array = np.asarray(audio, dtype=np.float32)
            duration = float(array.shape[0]) / float(sample_rate)
            yield CapturedAudioBlock(
                audio=array,
                start_unix=end_unix - duration,
                end_unix=end_unix,
                input_overflow=bool(overflowed),
            )


def audio_blocks(device_id: Optional[int], sample_rate: int, channels: int, window_sec: float) -> Iterator[np.ndarray]:
    for block in audio_block_stream(device_id, sample_rate, channels, window_sec):
        if block.input_overflow:
            print("[WARN] audio input overflow")
        yield block.audio


class ThreadedAudioCapture:
    def __init__(
        self,
        *,
        device_id: Optional[int],
        sample_rate: int,
        channels: int,
        window_sec: float,
        capture_block_sec: float | None = None,
        latency: str | float | None = None,
        queue_size: int = 4,
        on_block: Callable[[CapturedAudioBlock], None] | None = None,
        on_block_async: bool = True,
        on_block_queue_size: int = 256,
        overflow_recent_sec: float = 5.0,
        source: Callable[[], Iterator[CapturedAudioBlock]] | None = None,
    ):
        self.device_id = device_id
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.window_sec = float(window_sec)
        self.capture_block_sec = float(capture_block_sec or window_sec)
        self.latency = latency
        self.queue_size = max(1, int(queue_size))
        self.on_block = on_block
        self.on_block_async = bool(on_block_async)
        self.on_block_queue_size = max(1, int(on_block_queue_size))
        self.overflow_recent_sec = max(0.1, float(overflow_recent_sec))
        self._source = source
        self._queue: queue.Queue[CapturedAudioBlock] = queue.Queue(maxsize=self.queue_size)
        self._on_block_queue: queue.Queue[CapturedAudioBlock] | None = (
            queue.Queue(maxsize=self.on_block_queue_size) if self.on_block is not None and self.on_block_async else None
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._on_block_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._audio_input_overflow_count = 0
        self._overflow_timestamps: deque[float] = deque(maxlen=200)
        self._detection_blocks_dropped = 0
        self._on_block_dropped = 0
        self._capture_blocks_seen = 0
        self._last_error: str | None = None
        self._detect_audio_parts: list[np.ndarray] = []
        self._detect_start_unix: float | None = None
        self._detect_end_unix: float | None = None
        self._detect_overflow = False
        self._detect_samples = 0
        self._detection_window_samples = max(1, int(round(self.sample_rate * self.window_sec)))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        if self._on_block_queue is not None and (self._on_block_thread is None or not self._on_block_thread.is_alive()):
            self._on_block_thread = threading.Thread(target=self._run_on_block_worker, name="skyear-recording-writer", daemon=True)
            self._on_block_thread.start()
        self._thread = threading.Thread(target=self._run, name="skyear-audio-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.window_sec * 2.0))
        if self._on_block_thread is not None:
            self._on_block_thread.join(timeout=max(1.0, self.window_sec * 2.0))

    def blocks(self) -> Iterator[CapturedAudioBlock]:
        while not self._stop.is_set() or not self._queue.empty() or self._thread_alive():
            try:
                yield self._queue.get(timeout=0.25)
            except queue.Empty:
                if not self._thread_alive():
                    break

    def stats(self, now: float | None = None) -> dict[str, int | float | bool | list[float] | str | None]:
        now = time.time() if now is None else float(now)
        with self._lock:
            recent = [ts for ts in self._overflow_timestamps if now - float(ts) <= self.overflow_recent_sec]
            return {
                "audio_input_overflow_count": int(self._audio_input_overflow_count),
                "overflow_recent": bool(recent),
                "overflow_recent_count": len(recent),
                "overflow_timestamps": list(self._overflow_timestamps),
                "detection_blocks_dropped": int(self._detection_blocks_dropped),
                "on_block_dropped": int(self._on_block_dropped),
                "capture_queue_depth": int(self._queue.qsize()),
                "on_block_queue_depth": 0 if self._on_block_queue is None else int(self._on_block_queue.qsize()),
                "capture_blocks_seen": int(self._capture_blocks_seen),
                "capture_last_error": self._last_error,
            }

    def _thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            stream = (
                self._source()
                if self._source is not None
                else audio_block_stream(
                    self.device_id,
                    self.sample_rate,
                    self.channels,
                    self.capture_block_sec,
                    latency=self.latency,
                )
            )
            for block in stream:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._capture_blocks_seen += 1
                    if block.input_overflow:
                        self._audio_input_overflow_count += 1
                        self._overflow_timestamps.append(float(block.end_unix))
                if block.input_overflow:
                    print("[WARN] audio input overflow")
                self._dispatch_on_block(block)
                self._put_detection_block(block)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] audio capture stopped: {type(exc).__name__}: {exc}")

    def _dispatch_on_block(self, block: CapturedAudioBlock) -> None:
        if self.on_block is None:
            return
        if self._on_block_queue is None:
            self.on_block(copy_audio_block(block))
            return
        try:
            self._on_block_queue.put_nowait(copy_audio_block(block))
        except queue.Full:
            with self._lock:
                self._on_block_dropped += 1

    def _run_on_block_worker(self) -> None:
        while not self._stop.is_set() or (self._on_block_queue is not None and not self._on_block_queue.empty()):
            if self._on_block_queue is None:
                return
            try:
                block = self._on_block_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if self.on_block is not None:
                    self.on_block(block)
            except Exception as exc:
                with self._lock:
                    self._last_error = f"on_block {type(exc).__name__}: {exc}"
                print(f"[WARN] recording block handler failed: {type(exc).__name__}: {exc}")

    def _put_detection_block(self, block: CapturedAudioBlock) -> None:
        array = np.asarray(block.audio, dtype=np.float32)
        if self._detect_start_unix is None:
            self._detect_start_unix = float(block.start_unix)
        self._detect_end_unix = float(block.end_unix)
        self._detect_overflow = bool(self._detect_overflow or block.input_overflow)
        self._detect_audio_parts.append(array)
        self._detect_samples += int(array.shape[0])
        while self._detect_samples >= self._detection_window_samples:
            self._put_detection_window_locked(self._detection_window_samples)

    def _put_detection_window_locked(self, target_samples: int) -> None:
        remaining = int(target_samples)
        out_parts: list[np.ndarray] = []
        new_parts: list[np.ndarray] = []
        for part in self._detect_audio_parts:
            if remaining <= 0:
                new_parts.append(part)
                continue
            if part.shape[0] <= remaining:
                out_parts.append(part)
                remaining -= int(part.shape[0])
                continue
            out_parts.append(part[:remaining])
            new_parts.append(part[remaining:])
            remaining = 0
        if not out_parts:
            return
        audio = np.concatenate(out_parts, axis=0)
        duration = float(audio.shape[0]) / float(self.sample_rate)
        start_unix = float(self._detect_start_unix or time.time())
        end_unix = start_unix + duration
        self._detect_audio_parts = new_parts
        self._detect_samples = int(sum(part.shape[0] for part in self._detect_audio_parts))
        self._detect_start_unix = end_unix if self._detect_samples else None
        self._detect_end_unix = self._detect_end_unix if self._detect_samples else None
        out = CapturedAudioBlock(
            audio=audio,
            start_unix=start_unix,
            end_unix=end_unix,
            input_overflow=bool(self._detect_overflow),
        )
        self._detect_overflow = False if not self._detect_samples else self._detect_overflow
        self._put_detection_queue(out)

    def _put_detection_queue(self, block: CapturedAudioBlock) -> None:
        while True:
            try:
                self._queue.put_nowait(block)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue
                with self._lock:
                    self._detection_blocks_dropped += 1

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def per_channel_rms(audio: np.ndarray) -> list[float]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        value = float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2))) if array.size else 0.0
        return [value]
    if array.ndim != 2:
        raise ValueError(f"audio must be mono or 2D multi-channel, got shape {array.shape}")
    out: list[float] = []
    for idx in range(array.shape[1]):
        channel = np.asarray(array[:, idx], dtype=np.float64)
        out.append(float(np.sqrt(np.mean(channel * channel))) if channel.size else 0.0)
    return out


def select_mono_channel(
    audio: np.ndarray,
    *,
    mode: str = "mean",
    channel_scores: list[float] | None = None,
) -> tuple[np.ndarray, int | None, list[float]]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        return array.astype(np.float32), 0, per_channel_rms(array)
    if array.ndim != 2:
        raise ValueError(f"audio must be mono or 2D multi-channel, got shape {array.shape}")
    if array.shape[1] == 1:
        return array[:, 0].astype(np.float32), 0, per_channel_rms(array)

    normalized = str(mode or "mean").strip().lower()
    rms = per_channel_rms(array)
    if normalized in {"channel0", "ch0", "left"}:
        selected = 0
    elif normalized in {"channel1", "ch1", "right"}:
        selected = min(1, array.shape[1] - 1)
    elif normalized in {"max_rms", "strongest_rms"}:
        selected = int(np.argmax(np.asarray(rms, dtype=np.float64)))
    elif normalized in {"strongest_harmonic", "max_harmonic"} and channel_scores:
        selected = int(np.argmax(np.asarray(channel_scores, dtype=np.float64)))
    else:
        return array.mean(axis=1).astype(np.float32), None, rms
    return array[:, selected].astype(np.float32), selected, rms
