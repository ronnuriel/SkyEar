from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


@dataclass
class HFDetectionResult:
    p_drone: Optional[float] = None
    label: Optional[str] = None
    class_probs: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class HFDetector:
    def __init__(
        self,
        model_id: str,
        fallback_drone_label_idx: int = 1,
        threshold: float = 0.70,
    ):
        self.model_id = model_id
        self.fallback_drone_label_idx = fallback_drone_label_idx
        self.threshold = threshold
        self._extractor = None
        self._model = None
        self._torch = None
        self._load_error: Optional[str] = None

    def predict(self, audio_mono: np.ndarray, sr: int) -> HFDetectionResult:
        if not self._ensure_loaded():
            return HFDetectionResult(error=self._load_error)

        try:
            audio = np.asarray(audio_mono, dtype=np.float32).reshape(-1)
            target_sr = int(getattr(self._extractor, "sampling_rate", 16000) or 16000)
            if sr != target_sr:
                audio = resample_poly(audio, target_sr, sr).astype(np.float32)

            features = self._extractor(audio, sampling_rate=target_sr, return_tensors="pt")
            with self._torch.no_grad():
                output = self._model(**features)
                probs = self._torch.softmax(output.logits[0], dim=-1).detach().cpu().numpy()

            id2label = getattr(self._model.config, "id2label", {}) or {}
            class_probs = {}
            for idx, prob in enumerate(probs):
                label = str(id2label.get(idx, idx))
                class_probs[label] = float(prob)

            best_idx = int(np.argmax(probs))
            label = str(id2label.get(best_idx, best_idx))
            p_drone = self._drone_probability(probs, id2label)
            return HFDetectionResult(
                p_drone=p_drone,
                label=label,
                class_probs=class_probs,
                error=None,
            )
        except Exception as exc:
            return HFDetectionResult(error=str(exc))

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._extractor is not None:
            return True
        if self._load_error:
            return False

        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self._torch = torch
            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_id)
            self._model.eval()
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def _drone_probability(self, probs: np.ndarray, id2label: dict) -> float:
        drone_indices = [
            idx
            for idx, label in id2label.items()
            if "drone" in str(label).lower() or "uav" in str(label).lower()
        ]
        if drone_indices:
            return float(max(probs[int(idx)] for idx in drone_indices))
        if 0 <= self.fallback_drone_label_idx < len(probs):
            return float(probs[self.fallback_drone_label_idx])
        return 0.0
