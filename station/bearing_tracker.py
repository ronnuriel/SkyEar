from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def normalize_bearing_deg(value: float) -> float:
    return float(value) % 360.0


def circular_delta_deg(new: float, old: float) -> float:
    return (float(new) - float(old) + 180.0) % 360.0 - 180.0


def circular_distance_deg(left: float, right: float) -> float:
    return abs(circular_delta_deg(left, right))


def circular_smooth_deg(previous: float, current: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    prev_rad = math.radians(float(previous))
    curr_rad = math.radians(float(current))
    x = (1.0 - alpha) * math.cos(prev_rad) + alpha * math.cos(curr_rad)
    y = (1.0 - alpha) * math.sin(prev_rad) + alpha * math.sin(curr_rad)
    return normalize_bearing_deg(math.degrees(math.atan2(y, x)))


@dataclass
class BearingTrackerConfig:
    bearing_smoothing_alpha: float = 0.25
    bearing_flip_hysteresis_windows: int = 3
    max_bearing_jump_deg: float = 60.0
    coast_hold_sec: float = 5.0
    show_unreliable_bearing: bool = False
    poor_bearing_sector_width_deg: float = 60.0
    good_bearing_sector_width_deg: float = 18.0


@dataclass
class BearingTrackFrame:
    raw_bearing_deg: float | None = None
    tracked_bearing_deg: float | None = None
    bearing_velocity_deg_per_sec: float = 0.0
    bearing_track_age_sec: float = 0.0
    bearing_track_stable: bool = False
    bearing_track_status: str = "lost"
    bearing_flip_suppressed: bool = False
    bearing_reject_reason: str | None = None
    bearing_used_for_geo: bool = False
    bearing_uncertainty_deg: float | None = None


class BearingTracker:
    """Track geographic bearings: 0 north, 90 east, 180 south, 270 west."""

    def __init__(self, config: BearingTrackerConfig | None = None):
        self.config = config or BearingTrackerConfig()
        self._tracked_bearing_deg: float | None = None
        self._track_started_at: float | None = None
        self._last_update_at: float | None = None
        self._last_seen_at: float | None = None
        self._tracked_peak_ratio: float | None = None
        self._pending_flip_bearing_deg: float | None = None
        self._pending_flip_count = 0

    def update(
        self,
        *,
        timestamp: float,
        raw_bearing_deg: float | None,
        bearing_quality: str | None,
        bearing_reliable: bool | None = None,
        beam_confidence_pct: float | None = None,
        peak_ratio: float | None = None,
        second_peak_ratio: float | None = None,
        bearing_reject_reason: str | None = None,
    ) -> BearingTrackFrame:
        raw = None if raw_bearing_deg is None else normalize_bearing_deg(float(raw_bearing_deg))
        quality = str(bearing_quality or "").lower()
        reliable = self._is_reliable(quality, bearing_reliable, beam_confidence_pct)
        reason = bearing_reject_reason
        if raw is None or not reliable:
            return self._coast_or_lost(float(timestamp), raw, reason)

        if self._tracked_bearing_deg is None:
            self._tracked_bearing_deg = raw
            self._track_started_at = float(timestamp)
            self._last_update_at = float(timestamp)
            self._last_seen_at = float(timestamp)
            self._tracked_peak_ratio = None if peak_ratio is None else float(peak_ratio)
            self._pending_flip_bearing_deg = None
            self._pending_flip_count = 0
            return self._frame(float(timestamp), raw, "acquiring", reason)

        delta = circular_distance_deg(raw, self._tracked_bearing_deg)
        flip_like = self._is_flip_like(delta, peak_ratio, second_peak_ratio, quality)
        if flip_like:
            self._pending_flip_count = self._next_pending_count(raw)
            self._pending_flip_bearing_deg = raw
            if self._pending_flip_count < max(1, int(self.config.bearing_flip_hysteresis_windows)):
                reason = "ambiguous_lobe_flip" if delta > 90.0 else "bearing_flip_suppressed"
                self._last_seen_at = float(timestamp)
                return self._frame(float(timestamp), raw, "tracking", reason, flip_suppressed=True)
        else:
            self._pending_flip_bearing_deg = None
            self._pending_flip_count = 0

        previous = self._tracked_bearing_deg
        previous_update = self._last_update_at
        alpha = float(self.config.bearing_smoothing_alpha)
        if quality == "poor":
            alpha *= 0.5
        if flip_like and self._pending_flip_count >= max(1, int(self.config.bearing_flip_hysteresis_windows)):
            alpha = max(alpha, 0.65)
            self._track_started_at = float(timestamp)
            reason = reason or "bearing_lobe_switch_confirmed"
        self._tracked_bearing_deg = circular_smooth_deg(previous, raw, alpha)
        self._last_update_at = float(timestamp)
        self._last_seen_at = float(timestamp)
        if peak_ratio is not None:
            self._tracked_peak_ratio = float(peak_ratio)
        velocity = 0.0
        if previous_update is not None:
            dt = max(1e-6, float(timestamp) - float(previous_update))
            velocity = circular_delta_deg(self._tracked_bearing_deg, previous) / dt
        frame = self._frame(float(timestamp), raw, "tracking", reason)
        frame.bearing_velocity_deg_per_sec = float(velocity)
        return frame

    def _is_reliable(
        self,
        quality: str,
        bearing_reliable: bool | None,
        beam_confidence_pct: float | None,
    ) -> bool:
        if bearing_reliable is False or quality == "unreliable":
            return bool(self.config.show_unreliable_bearing)
        if bearing_reliable is True:
            return True
        if quality in {"good", "fair", "poor"}:
            return True
        return beam_confidence_pct is not None and float(beam_confidence_pct) >= 0.35

    def _is_flip_like(
        self,
        delta: float,
        peak_ratio: float | None,
        second_peak_ratio: float | None,
        quality: str,
    ) -> bool:
        if delta <= float(self.config.max_bearing_jump_deg):
            return False
        if delta > 90.0:
            return True
        return quality in {"poor", "unreliable", ""}

    def _next_pending_count(self, raw_bearing_deg: float) -> int:
        if self._pending_flip_bearing_deg is None:
            return 1
        if circular_distance_deg(raw_bearing_deg, self._pending_flip_bearing_deg) <= float(self.config.max_bearing_jump_deg):
            return self._pending_flip_count + 1
        return 1

    def _coast_or_lost(
        self,
        timestamp: float,
        raw_bearing_deg: float | None,
        reason: str | None,
    ) -> BearingTrackFrame:
        if self._tracked_bearing_deg is None or self._last_seen_at is None:
            return BearingTrackFrame(raw_bearing_deg=raw_bearing_deg, bearing_reject_reason=reason, bearing_track_status="lost")
        age_since_seen = float(timestamp) - float(self._last_seen_at)
        flip_suppressed = False
        if raw_bearing_deg is not None and circular_distance_deg(raw_bearing_deg, self._tracked_bearing_deg) > 90.0:
            flip_suppressed = True
            reason = reason or "ambiguous_lobe_flip"
        if age_since_seen <= float(self.config.coast_hold_sec):
            return self._frame(timestamp, raw_bearing_deg, "coasting", reason, flip_suppressed=flip_suppressed)
        self._tracked_bearing_deg = None
        self._track_started_at = None
        self._last_update_at = None
        self._last_seen_at = None
        self._pending_flip_bearing_deg = None
        self._pending_flip_count = 0
        return BearingTrackFrame(raw_bearing_deg=raw_bearing_deg, bearing_reject_reason=reason, bearing_track_status="lost")

    def _frame(
        self,
        timestamp: float,
        raw_bearing_deg: float | None,
        status: str,
        reason: str | None,
        *,
        flip_suppressed: bool = False,
    ) -> BearingTrackFrame:
        age = 0.0
        if self._track_started_at is not None:
            age = max(0.0, float(timestamp) - float(self._track_started_at))
        stable = status == "tracking" and age >= 1.0 and not flip_suppressed
        used_for_geo = status in {"tracking", "acquiring"} and not flip_suppressed and self._tracked_bearing_deg is not None
        uncertainty = None
        if used_for_geo:
            uncertainty = (
                float(self.config.good_bearing_sector_width_deg)
                if stable
                else float(self.config.poor_bearing_sector_width_deg)
            )
        return BearingTrackFrame(
            raw_bearing_deg=raw_bearing_deg,
            tracked_bearing_deg=self._tracked_bearing_deg,
            bearing_track_age_sec=age,
            bearing_track_stable=stable,
            bearing_track_status=status,
            bearing_flip_suppressed=bool(flip_suppressed),
            bearing_reject_reason=reason,
            bearing_used_for_geo=used_for_geo,
            bearing_uncertainty_deg=uncertainty,
        )


def bearing_tracker_config_from_direction(direction_cfg: dict[str, Any]) -> BearingTrackerConfig:
    return BearingTrackerConfig(
        bearing_smoothing_alpha=float(direction_cfg.get("bearing_smoothing_alpha", 0.25)),
        bearing_flip_hysteresis_windows=int(direction_cfg.get("bearing_flip_hysteresis_windows", 3)),
        max_bearing_jump_deg=float(direction_cfg.get("max_bearing_jump_deg", 60.0)),
        coast_hold_sec=float(direction_cfg.get("coast_hold_sec", 5.0)),
        show_unreliable_bearing=bool(direction_cfg.get("show_unreliable_bearing", False)),
        poor_bearing_sector_width_deg=float(direction_cfg.get("poor_bearing_sector_width_deg", 60.0)),
        good_bearing_sector_width_deg=float(direction_cfg.get("good_bearing_sector_width_deg", 18.0)),
    )
