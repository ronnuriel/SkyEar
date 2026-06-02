from __future__ import annotations

import builtins
import os

import numpy as np
import pytest

from station.hf_detector import HFDetectionResult, HFDetector


def test_hf_detector_missing_dependencies_fails_gracefully(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    detector = HFDetector("example/missing-model")

    result = detector.predict(np.zeros(16000, dtype=np.float32), 16000)

    assert result.error
    assert result.p_drone is None
    assert result.class_probs == {}


def test_hf_detector_transformers_missing_fails_gracefully(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError("transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    detector = HFDetector("example/missing-model")

    result = detector.predict(np.zeros(16000, dtype=np.float32), 16000)

    assert result.error
    assert result.p_drone is None
    assert result.label is None
    assert result.class_probs == {}


def test_hf_detection_result_has_expected_fields():
    result = HFDetectionResult()

    assert hasattr(result, "p_drone")
    assert hasattr(result, "label")
    assert hasattr(result, "class_probs")
    assert hasattr(result, "error")
    assert result.class_probs == {}


@pytest.mark.parametrize(
    ("id2label", "probs", "fallback_idx", "expected"),
    [
        ({0: "drone", 1: "not_drone"}, [0.2, 0.8], 1, 0.2),
        ({0: "not_drone", 1: "drone"}, [0.9, 0.1], 1, 0.1),
        ({0: "background", 1: "drone"}, [0.7, 0.3], 1, 0.3),
        ({0: "LABEL_0", 1: "LABEL_1"}, [0.4, 0.6], 1, 0.6),
    ],
)
def test_drone_probability_ignores_negative_drone_labels(id2label, probs, fallback_idx, expected):
    detector = HFDetector("example/model", fallback_drone_label_idx=fallback_idx)

    p_drone = detector._drone_probability(np.asarray(probs, dtype=np.float32), id2label)

    assert p_drone == pytest.approx(expected)


@pytest.mark.skipif(
    os.environ.get("SKYEAR_RUN_HF_TESTS") != "1",
    reason="Set SKYEAR_RUN_HF_TESTS=1 to download and run the Hugging Face model.",
)
def test_hf_detector_real_model_opt_in():
    detector = HFDetector()

    result = detector.predict(np.zeros(16000, dtype=np.float32), 16000)

    assert hasattr(result, "p_drone")
    assert hasattr(result, "label")
    assert hasattr(result, "class_probs")
    assert hasattr(result, "error")
