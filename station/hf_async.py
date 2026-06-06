from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from station.hf_detector import HFDetectionResult, HFDetector


@dataclass
class AsyncHFState:
    latest_result: HFDetectionResult | None = None
    latest_started_unix: float | None = None
    latest_completed_unix: float | None = None
    in_flight: bool = False
    dropped_requests: int = 0


class AsyncHFDetectorRunner:
    def __init__(self, detector: HFDetector):
        self.detector = detector
        self._queue: queue.Queue[tuple[np.ndarray, int, float]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = AsyncHFState()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="skyear-hf-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def submit(self, audio_mono: np.ndarray, sample_rate: int, timestamp_unix: float) -> bool:
        payload = (np.asarray(audio_mono, dtype=np.float32).reshape(-1).copy(), int(sample_rate), float(timestamp_unix))
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            with self._lock:
                self._state.dropped_requests += 1
            return False

    def state(self) -> AsyncHFState:
        with self._lock:
            return AsyncHFState(
                latest_result=self._state.latest_result,
                latest_started_unix=self._state.latest_started_unix,
                latest_completed_unix=self._state.latest_completed_unix,
                in_flight=self._state.in_flight,
                dropped_requests=self._state.dropped_requests,
            )

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                audio, sample_rate, started = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            with self._lock:
                self._state.in_flight = True
                self._state.latest_started_unix = float(started)
            result = self.detector.predict(audio, sample_rate)
            with self._lock:
                self._state.latest_result = result
                self._state.latest_completed_unix = time.time()
                self._state.in_flight = False
