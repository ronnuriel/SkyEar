from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from station.harmonic import harmonic_score

@dataclass
class DetectorConfig:
    f0_min: int = 500
    f0_max: int = 2200
    max_freq: int = 7000
    suspect_threshold: float = 14.0
    alert_threshold: float = 22.0

@dataclass
class DetectionResult:
    harmonic_score: float
    best_f0_hz: Optional[int]
    confidence: float
    status: str

def sigmoid_score(score: float, threshold: float) -> float:
    return float(1 / (1 + np.exp(-(score - threshold) / 3.0)))

def detect_drone_like(audio_mono: np.ndarray, sr: int, cfg: DetectorConfig) -> DetectionResult:
    score, best_f0, _ = harmonic_score(audio_mono, sr, cfg.f0_min, cfg.f0_max, cfg.max_freq)
    confidence = sigmoid_score(score, cfg.suspect_threshold)
    if score >= cfg.alert_threshold:
        status = "alert"
    elif score >= cfg.suspect_threshold:
        status = "suspect"
    else:
        status = "background"
    return DetectionResult(float(score), best_f0, float(confidence), status)
