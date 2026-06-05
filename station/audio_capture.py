from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

def _sounddevice():
    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sounddevice is required for audio capture. Install it with `pip install -r requirements.txt`."
        ) from exc
    return sd

def list_input_devices():
    sd = _sounddevice()
    out = []
    for idx, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({
                "id": idx,
                "name": d["name"],
                "max_input_channels": int(d["max_input_channels"]),
                "default_samplerate": int(d.get("default_samplerate", 44100)),
            })
    return out

@dataclass
class CapturedAudioBlock:
    audio: np.ndarray
    start_unix: float
    end_unix: float
    input_overflow: bool = False


def audio_block_stream(
    device_id: Optional[int],
    sample_rate: int,
    channels: int,
    window_sec: float,
) -> Iterator[CapturedAudioBlock]:
    sd = _sounddevice()
    block = int(sample_rate * window_sec)
    with sd.InputStream(device=device_id, channels=channels, samplerate=sample_rate, blocksize=block, dtype="float32") as stream:
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
        queue_size: int = 4,
        on_block: Callable[[CapturedAudioBlock], None] | None = None,
        source: Callable[[], Iterator[CapturedAudioBlock]] | None = None,
    ):
        self.device_id = device_id
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.window_sec = float(window_sec)
        self.queue_size = max(1, int(queue_size))
        self.on_block = on_block
        self._source = source
        self._queue: queue.Queue[CapturedAudioBlock] = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._audio_input_overflow_count = 0
        self._detection_blocks_dropped = 0
        self._capture_blocks_seen = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="skyear-audio-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.window_sec * 2.0))

    def blocks(self) -> Iterator[CapturedAudioBlock]:
        while not self._stop.is_set() or not self._queue.empty() or self._thread_alive():
            try:
                yield self._queue.get(timeout=0.25)
            except queue.Empty:
                if not self._thread_alive():
                    break

    def stats(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "audio_input_overflow_count": int(self._audio_input_overflow_count),
                "detection_blocks_dropped": int(self._detection_blocks_dropped),
                "capture_queue_depth": int(self._queue.qsize()),
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
                else audio_block_stream(self.device_id, self.sample_rate, self.channels, self.window_sec)
            )
            for block in stream:
                if self._stop.is_set():
                    break
                with self._lock:
                    self._capture_blocks_seen += 1
                    if block.input_overflow:
                        self._audio_input_overflow_count += 1
                if block.input_overflow:
                    print("[WARN] audio input overflow")
                if self.on_block is not None:
                    self.on_block(block)
                self._put_detection_block(block)
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            print(f"[WARN] audio capture stopped: {type(exc).__name__}: {exc}")

    def _put_detection_block(self, block: CapturedAudioBlock) -> None:
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
