from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from station.two_mic_direction import TwoMicDirectionResult


@dataclass
class TwoMicDirectionTrackerConfig:
    smoothing_windows: int = 5
    min_stable_windows: int = 3
    max_angle_std_deg: float = 25.0
    max_side_flip_rate: float = 0.35
    min_confidence: float = 0.45
    unstable_sector_width_deg: float = 120.0


class TwoMicDirectionTracker:
    def __init__(self, config: TwoMicDirectionTrackerConfig):
        self.config = config
        self._history: deque[tuple[float, TwoMicDirectionResult]] = deque(maxlen=max(1, int(config.smoothing_windows)))

    def update(self, timestamp: float, result: TwoMicDirectionResult) -> TwoMicDirectionResult:
        if _valid(result):
            self._history.append((float(timestamp), result))
        valid = [item for _, item in self._history if _valid(item)]
        result.tracker_window_count = len(valid)
        if not valid:
            return self._unstable(result, stable_count=0)

        sides = [item.side for item in valid]
        dominant_side = max(set(sides), key=sides.count)
        stable_count = sum(1 for side in sides if side == dominant_side)
        confidence_ok = float(result.confidence or 0.0) >= float(self.config.min_confidence)
        same_side_ok = stable_count >= max(1, int(self.config.min_stable_windows))
        flip_rate = _side_flip_rate(sides)
        flip_ok = flip_rate <= float(self.config.max_side_flip_rate)
        angles = [float(item.angle_from_center_deg) for item in valid if item.angle_from_center_deg is not None]
        angle_std = float(np.std(np.asarray(angles, dtype=np.float64))) if len(angles) >= 2 else 0.0
        angle_ok = angle_std <= float(self.config.max_angle_std_deg)

        result.stable_window_count = stable_count
        if confidence_ok and same_side_ok and flip_ok and angle_ok:
            result.stable = True
            return result
        return self._unstable(result, stable_count=stable_count)

    def _unstable(self, result: TwoMicDirectionResult, *, stable_count: int) -> TwoMicDirectionResult:
        result.stable = False
        result.stable_window_count = int(stable_count)
        result.tracker_window_count = len([item for _, item in self._history if _valid(item)])
        result.look_label = "unknown"
        result.look_hint = "DIRECTION UNSTABLE - scan left and right, front/back ambiguous"
        result.sector_width_deg = float(self.config.unstable_sector_width_deg)
        result.front_back_ambiguous = True
        return result


def two_mic_tracker_config_from_dict(config: dict[str, Any]) -> TwoMicDirectionTrackerConfig:
    return TwoMicDirectionTrackerConfig(
        smoothing_windows=int(config.get("smoothing_windows", 5)),
        min_stable_windows=int(config.get("min_stable_windows", 3)),
        max_angle_std_deg=float(config.get("max_angle_std_deg", 25.0)),
        max_side_flip_rate=float(config.get("max_side_flip_rate", 0.35)),
        min_confidence=float(config.get("min_confidence", 0.45)),
        unstable_sector_width_deg=float(config.get("unstable_sector_width_deg", 120.0)),
    )


def _valid(result: TwoMicDirectionResult) -> bool:
    return (
        result.side in {"left", "right", "center"}
        and result.angle_from_center_deg is not None
        and result.confidence is not None
    )


def _side_flip_rate(sides: list[str]) -> float:
    directional = [side for side in sides if side in {"left", "right"}]
    if len(directional) < 2:
        return 0.0
    flips = sum(1 for prev, cur in zip(directional, directional[1:]) if prev != cur)
    return float(flips / max(1, len(directional) - 1))
