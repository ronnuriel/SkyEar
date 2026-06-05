from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


class HighPassFilter:
    def __init__(self, sample_rate: int, cutoff_hz: float, channels: int, order: int = 4):
        self.sample_rate = int(sample_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.channels = int(channels)
        self.order = int(order)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        nyquist = self.sample_rate / 2.0
        if self.cutoff_hz <= 0.0 or self.cutoff_hz >= nyquist:
            raise ValueError("cutoff_hz must be between 0 and Nyquist")
        self._sos = butter(
            max(1, self.order),
            self.cutoff_hz,
            btype="highpass",
            fs=self.sample_rate,
            output="sos",
        )
        self._zi = [np.zeros((self._sos.shape[0], 2), dtype=np.float64) for _ in range(self.channels)]

    def process(self, audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            filtered, self._zi[0] = sosfilt(self._sos, array, zi=self._zi[0])
            return filtered.astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"audio must be mono or 2D multi-channel, got shape {array.shape}")
        if array.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {array.shape[1]}")
        out = np.empty_like(array, dtype=np.float32)
        for channel_idx in range(self.channels):
            filtered, self._zi[channel_idx] = sosfilt(
                self._sos,
                array[:, channel_idx],
                zi=self._zi[channel_idx],
            )
            out[:, channel_idx] = filtered.astype(np.float32)
        return out
