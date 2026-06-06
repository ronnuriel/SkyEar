from __future__ import annotations

import math
import time
from itertools import combinations
from typing import Any


EARTH_RADIUS_M = 6_371_000.0
CANDIDATE_LABELS = {"ml_drone_candidate", "local_drone_candidate", "strong_local_candidate", "drone_like", "alert"}


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> dict[str, float]:
    bearing = math.radians(float(bearing_deg))
    phi1 = math.radians(float(lat))
    lambda1 = math.radians(float(lon))
    delta = float(distance_m) / EARTH_RADIUS_M
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(bearing))
    lambda2 = lambda1 + math.atan2(
        math.sin(bearing) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return {"latitude": math.degrees(phi2), "longitude": ((math.degrees(lambda2) + 540.0) % 360.0) - 180.0}


def bearing_line_segment(
    lat: float,
    lon: float,
    bearing_deg: float,
    min_range_m: float,
    max_range_m: float,
) -> list[dict[str, float]]:
    return [
        destination_point(lat, lon, bearing_deg, min_range_m),
        destination_point(lat, lon, bearing_deg, max_range_m),
    ]


def bearing_sector_polygon(
    lat: float,
    lon: float,
    bearing_deg: float,
    uncertainty_deg: float,
    min_range_m: float,
    max_range_m: float,
    steps: int = 16,
) -> list[dict[str, float]]:
    steps = max(2, int(steps))
    start = float(bearing_deg) - float(uncertainty_deg)
    end = float(bearing_deg) + float(uncertainty_deg)
    outer = [
        destination_point(lat, lon, start + (end - start) * idx / (steps - 1), max_range_m)
        for idx in range(steps)
    ]
    inner = [
        destination_point(lat, lon, end - (end - start) * idx / (steps - 1), min_range_m)
        for idx in range(steps)
    ]
    return outer + inner


def _latlon_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    x = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(math.radians(origin_lat))
    y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return x, y


def _xy_to_latlon(x: float, y: float, origin_lat: float, origin_lon: float) -> dict[str, float]:
    lat = origin_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(x / (EARTH_RADIUS_M * max(0.1, math.cos(math.radians(origin_lat)))))
    return {"latitude": lat, "longitude": lon}


def _line_intersection(a: dict[str, Any], b: dict[str, Any], origin_lat: float, origin_lon: float) -> dict[str, Any] | None:
    ax, ay = _latlon_to_xy(a["latitude"], a["longitude"], origin_lat, origin_lon)
    bx, by = _latlon_to_xy(b["latitude"], b["longitude"], origin_lat, origin_lon)
    theta_a = math.radians(float(a["bearing_deg"]))
    theta_b = math.radians(float(b["bearing_deg"]))
    da = (math.sin(theta_a), math.cos(theta_a))
    db = (math.sin(theta_b), math.cos(theta_b))
    det = da[0] * (-db[1]) - da[1] * (-db[0])
    if abs(det) < 1e-6:
        return None
    rhs = (bx - ax, by - ay)
    ta = (rhs[0] * (-db[1]) - rhs[1] * (-db[0])) / det
    tb = (da[0] * rhs[1] - da[1] * rhs[0]) / det
    if ta < 0 or tb < 0:
        return None
    point = _xy_to_latlon(ax + ta * da[0], ay + ta * da[1], origin_lat, origin_lon)
    point["range_a_m"] = ta
    point["range_b_m"] = tb
    return point


def intersect_bearings(station_bearings: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in station_bearings if item.get("latitude") is not None and item.get("longitude") is not None and item.get("bearing_deg") is not None]
    if len(valid) < 2:
        return None
    origin_lat = sum(float(item["latitude"]) for item in valid) / len(valid)
    origin_lon = sum(float(item["longitude"]) for item in valid) / len(valid)
    intersections = []
    for a, b in combinations(valid, 2):
        point = _line_intersection(a, b, origin_lat, origin_lon)
        if point is not None:
            intersections.append(point)
    if not intersections:
        return None
    lat = sum(point["latitude"] for point in intersections) / len(intersections)
    lon = sum(point["longitude"] for point in intersections) / len(intersections)
    residuals = [haversine_distance_m(lat, lon, point["latitude"], point["longitude"]) for point in intersections]
    radius = max(50.0, sorted(residuals)[len(residuals) // 2] if residuals else 100.0)
    return {
        "latitude": lat,
        "longitude": lon,
        "radius_m": radius,
        "bearing_residual_deg": min(180.0, radius / 20.0),
        "source_station_ids": [str(item.get("station_id")) for item in valid],
        "bearing_cross_angle_deg": bearing_cross_angle_deg(valid),
        "bearing_geometry_quality": bearing_geometry_quality(valid),
    }


def _bearing_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def bearing_cross_angle_deg(station_bearings: list[dict[str, Any]]) -> float | None:
    bearings = [float(item["bearing_deg"]) % 360.0 for item in station_bearings if item.get("bearing_deg") is not None]
    if len(bearings) < 2:
        return None
    angles = []
    for a, b in combinations(bearings, 2):
        angles.append(_bearing_delta_deg(a, b))
    return min(angles, key=lambda value: min(value, 180.0 - value))


def bearing_geometry_quality(station_bearings: list[dict[str, Any]]) -> str:
    angle = bearing_cross_angle_deg(station_bearings)
    if angle is None:
        return "poor"
    if angle < 20.0 or angle > 160.0:
        return "poor"
    if angle < 40.0 or angle > 140.0:
        return "fair"
    return "good"


def confidence_for_bearing_geometry(confidence: float, quality: str) -> float:
    confidence = float(confidence)
    if quality == "poor":
        return max(0.1, min(confidence, confidence * 0.55))
    if quality == "fair":
        return max(0.1, min(confidence, confidence * 0.85))
    return confidence


def _event_location(event: Any) -> tuple[float | None, float | None, float | None]:
    loc = getattr(event, "station_location", None)
    metadata = getattr(event, "metadata", {}) or {}
    lat = getattr(event, "station_latitude", None) or metadata.get("latitude") or (getattr(loc, "latitude", None) if loc else None)
    lon = getattr(event, "station_longitude", None) or metadata.get("longitude") or (getattr(loc, "longitude", None) if loc else None)
    alt = getattr(event, "station_altitude_m", None) or metadata.get("altitude_m") or (getattr(loc, "altitude_m", None) if loc else None)
    return (None if lat is None else float(lat), None if lon is None else float(lon), None if alt is None else float(alt))


def _event_bearing(event: Any) -> float | None:
    metadata = getattr(event, "metadata", {}) or {}
    possible_front = getattr(event, "possible_front_azimuth_deg", None)
    if possible_front is None:
        possible_front = metadata.get("possible_front_azimuth_deg")
    if metadata.get("two_mic_direction_enabled") and possible_front is None:
        return None
    used_for_geo = getattr(event, "bearing_used_for_geo", None)
    if used_for_geo is None:
        used_for_geo = metadata.get("bearing_used_for_geo")
    if used_for_geo is False or str(used_for_geo).strip().lower() == "false":
        return None
    reliable = getattr(event, "bearing_reliable", None)
    if reliable is None:
        reliable = metadata.get("bearing_reliable")
    if reliable is False or str(reliable).strip().lower() == "false":
        return None
    for value in (
        getattr(event, "tracked_bearing_deg", None),
        getattr(event, "estimated_azimuth_deg", None),
        metadata.get("tracked_bearing_deg"),
        metadata.get("beam_bearing_deg"),
        metadata.get("bearing_deg"),
        metadata.get("estimated_azimuth_deg"),
    ):
        if value is not None:
            return float(value) % 360.0
    return None


def _is_candidate_event(event: Any) -> bool:
    metadata = getattr(event, "metadata", {}) or {}
    label = str(getattr(event, "operator_label", None) or metadata.get("operator_label") or "")
    return label in CANDIDATE_LABELS


def latest_candidate_bearings(events: list[Any], max_age_sec: float = 10.0, now: float | None = None) -> list[dict[str, Any]]:
    now = time.time() if now is None else float(now)
    latest: dict[str, Any] = {}
    for event in events:
        received = getattr(event, "server_received_unix", None) or (getattr(event, "metadata", {}) or {}).get("server_received_unix") or getattr(event, "timestamp_unix", 0.0)
        if now - float(received) <= float(max_age_sec):
            latest[getattr(event, "station_id")] = event
    bearings = []
    for event in latest.values():
        if not _is_candidate_event(event):
            continue
        lat, lon, _alt = _event_location(event)
        bearing = _event_bearing(event)
        if lat is None or lon is None or bearing is None:
            continue
        metadata = getattr(event, "metadata", {}) or {}
        uncertainty = getattr(event, "bearing_uncertainty_deg", None) or metadata.get("bearing_uncertainty_deg") or 25.0
        reliable = getattr(event, "bearing_reliable", None)
        if reliable is None:
            reliable = metadata.get("bearing_reliable")
        quality = getattr(event, "bearing_quality", None) or metadata.get("bearing_quality")
        if str(quality or "").lower() == "poor":
            uncertainty = max(float(uncertainty), 60.0)
        reject_reason = getattr(event, "bearing_reject_reason", None) or metadata.get("bearing_reject_reason")
        raw_bearing = getattr(event, "raw_bearing_deg", None)
        if raw_bearing is None:
            raw_bearing = metadata.get("raw_bearing_deg")
        tracked_bearing = getattr(event, "tracked_bearing_deg", None)
        if tracked_bearing is None:
            tracked_bearing = metadata.get("tracked_bearing_deg")
        bearings.append(
            {
                "station_id": getattr(event, "station_id"),
                "latitude": lat,
                "longitude": lon,
                "bearing_deg": bearing,
                "raw_bearing_deg": raw_bearing,
                "tracked_bearing_deg": tracked_bearing,
                "uncertainty_deg": float(uncertainty),
                "beam_confidence_pct": getattr(event, "beam_confidence_pct", None) or metadata.get("beam_confidence_pct"),
                "bearing_stable": getattr(event, "bearing_stable", None) or metadata.get("bearing_stable"),
                "bearing_reliable": reliable,
                "bearing_reject_reason": reject_reason,
                "bearing_quality": quality,
                "bearing_used_for_geo": True,
            }
        )
    return bearings


def estimate_from_recent_bearings(events: list[Any], max_age_sec: float = 10.0, now: float | None = None) -> dict[str, Any]:
    bearings = [item for item in latest_candidate_bearings(events, max_age_sec=max_age_sec, now=now) if item.get("bearing_deg") is not None]
    if not bearings:
        return {
            "estimate_type": "none",
            "confidence": 0.0,
            "source_station_ids": [],
            "reason": "no recent candidate station with location and bearing",
        }
    if len(bearings) == 1:
        item = bearings[0]
        quality = str(item.get("bearing_quality") or "")
        confidence = 0.14 if quality == "poor" else 0.25
        return {
            "estimate_type": "station_sector",
            "confidence": confidence,
            "source_station_ids": [item["station_id"]],
            "area_polygon": bearing_sector_polygon(item["latitude"], item["longitude"], item["bearing_deg"], item["uncertainty_deg"], 25.0, 800.0),
            "reason": "single-station bearing cue; range unknown",
        }
    intersection = intersect_bearings(bearings)
    if intersection is None:
        cross_angle = bearing_cross_angle_deg(bearings)
        quality = bearing_geometry_quality(bearings)
        return {
            "estimate_type": "multi_station_area",
            "confidence": confidence_for_bearing_geometry(0.35, quality),
            "source_station_ids": [item["station_id"] for item in bearings],
            "bearing_cross_angle_deg": cross_angle,
            "bearing_geometry_quality": quality,
            "reason": "multiple bearing cues did not intersect cleanly",
        }
    radius = float(intersection.get("radius_m") or 150.0)
    confidence = max(0.35, min(0.9, 1.0 - radius / 1000.0))
    quality = str(intersection.get("bearing_geometry_quality") or bearing_geometry_quality(bearings))
    confidence = confidence_for_bearing_geometry(confidence, quality)
    if any(str(item.get("bearing_quality") or "") == "poor" for item in bearings):
        confidence *= 0.65
    return {
        "estimate_type": "bearing_intersection" if len(bearings) == 2 else "multi_station_area",
        "latitude": intersection["latitude"],
        "longitude": intersection["longitude"],
        "radius_m": radius,
        "confidence": confidence,
        "source_station_ids": intersection["source_station_ids"],
        "bearing_residual_deg": intersection.get("bearing_residual_deg"),
        "bearing_cross_angle_deg": intersection.get("bearing_cross_angle_deg"),
        "bearing_geometry_quality": quality,
        "area_polygon": _circle_polygon(intersection["latitude"], intersection["longitude"], radius),
        "reason": "approximate acoustic candidate location from multiple passive bearing cues",
    }


def _circle_polygon(lat: float, lon: float, radius_m: float, steps: int = 24) -> list[dict[str, float]]:
    return [destination_point(lat, lon, 360.0 * idx / steps, radius_m) for idx in range(steps)]
