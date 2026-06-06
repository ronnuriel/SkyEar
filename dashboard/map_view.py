from __future__ import annotations

import math
from typing import Any

import pandas as pd


def normalize_map_state(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return {
        "server_time": payload.get("server_time"),
        "stations": payload.get("stations") or [],
        "bearing_cues": payload.get("bearing_cues") or [],
        "geo_estimates": payload.get("geo_estimates") or [],
        "tracks": payload.get("tracks") or [],
    }


def stations_missing_location(map_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        station
        for station in normalize_map_state(map_state)["stations"]
        if station.get("latitude") is None or station.get("longitude") is None
    ]


def bearing_ray_rows(events: list[dict], ray_length_m: float = 500.0) -> list[dict]:
    rows = []
    for event in events:
        loc = event.get("station_location") or {}
        if event.get("bearing_reliable") is False or event.get("bearing_used_for_geo") is False:
            continue
        bearing = (
            event.get("tracked_bearing_deg")
            if event.get("tracked_bearing_deg") is not None
            else event.get("estimated_azimuth_deg")
            if event.get("estimated_azimuth_deg") is not None
            else event.get("bearing_deg")
        )
        if bearing is None or loc.get("latitude") is None or loc.get("longitude") is None:
            continue
        lat = float(loc["latitude"])
        lon = float(loc["longitude"])
        bearing_rad = math.radians(float(bearing))
        d_north = math.cos(bearing_rad) * float(ray_length_m)
        d_east = math.sin(bearing_rad) * float(ray_length_m)
        end_lat = lat + d_north / 111_320.0
        end_lon = lon + d_east / (111_320.0 * max(0.1, math.cos(math.radians(lat))))
        rows.append(
            {
                "station_id": event.get("station_id"),
                "lat": lat,
                "lon": lon,
                "bearing_deg": float(bearing),
                "raw_bearing_deg": event.get("raw_bearing_deg"),
                "tracked_bearing_deg": event.get("tracked_bearing_deg"),
                "ray_end_lat": end_lat,
                "ray_end_lon": end_lon,
                "beam_score": event.get("beam_score"),
                "bearing_stable": event.get("bearing_stable"),
            }
        )
    return rows


def station_marker_rows(map_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for station in normalize_map_state(map_state)["stations"]:
        if station.get("latitude") is None or station.get("longitude") is None:
            continue
        status = str(station.get("last_status") or "background")
        health = str(station.get("health") or "offline")
        rows.append(
            {
                **station,
                "lat": float(station["latitude"]),
                "lon": float(station["longitude"]),
                "color": _station_color(status, health),
            }
        )
    return rows


def sector_polygon_rows(map_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cue in normalize_map_state(map_state)["bearing_cues"]:
        polygon = cue.get("sector_polygon") or []
        if len(polygon) >= 3:
            quality = str(cue.get("bearing_quality") or "")
            fill_alpha = 28 if quality == "poor" else 48
            line_alpha = 90 if quality == "poor" else 170
            rows.append(
                {
                    "station_id": cue.get("station_id"),
                    "bearing_deg": cue.get("bearing_deg"),
                    "raw_bearing_deg": cue.get("raw_bearing_deg"),
                    "tracked_bearing_deg": cue.get("tracked_bearing_deg"),
                    "uncertainty_deg": cue.get("uncertainty_deg"),
                    "bearing_quality": cue.get("bearing_quality"),
                    "bearing_reject_reason": cue.get("bearing_reject_reason"),
                    "bearing_used_for_geo": cue.get("bearing_used_for_geo"),
                    "fill_color": [255, 120, 20, fill_alpha],
                    "line_color": [255, 80, 20, line_alpha],
                    "polygon": [[point["longitude"], point["latitude"]] for point in polygon],
                }
            )
    return rows


def estimate_rows(map_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for estimate in normalize_map_state(map_state)["geo_estimates"]:
        if estimate.get("latitude") is not None and estimate.get("longitude") is not None:
            rows.append(
                {
                    **estimate,
                    "lat": float(estimate["latitude"]),
                    "lon": float(estimate["longitude"]),
                }
            )
    return rows


def track_rows(map_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for track in normalize_map_state(map_state)["tracks"]:
        latitude = track.get("latitude")
        longitude = track.get("longitude")
        if latitude is None or longitude is None:
            source = track.get("estimated_source") or {}
            latitude = source.get("latitude")
            longitude = source.get("longitude")
        if latitude is None or longitude is None:
            continue
        rows.append(
            {
                **track,
                "lat": float(latitude),
                "lon": float(longitude),
                "label": f"{track.get('track_id', 'track')} L{track.get('level', 0)}",
                "color": _track_color(int(track.get("level") or 0)),
            }
        )
    return rows


def _station_color(status: str, health: str) -> list[int]:
    if health in {"offline", "stale"}:
        return [130, 130, 130, 180]
    if status in {"alert", "drone_like"}:
        return [220, 40, 30, 220]
    if status in {"suspect", "calibrating"}:
        return [240, 190, 40, 220]
    return [40, 170, 80, 220]


def _track_color(level: int) -> list[int]:
    if level >= 3:
        return [220, 40, 30, 230]
    if level == 2:
        return [245, 145, 35, 220]
    if level == 1:
        return [240, 190, 40, 210]
    return [75, 130, 210, 190]


def render_passive_map(st, map_state: dict[str, Any]) -> None:
    st.subheader("Map / Passive Acoustic Situation")
    st.caption("Map estimates are approximate acoustic cues and not targeting-grade.")
    state = normalize_map_state(map_state)
    missing = stations_missing_location(state)
    if missing:
        st.warning("Stations missing location")
        st.dataframe(pd.DataFrame(missing), width="stretch")

    markers = station_marker_rows(state)
    sectors = sector_polygon_rows(state)
    estimates = estimate_rows(state)
    tracks = track_rows(state)
    if not markers:
        st.info("No station coordinates available for map rendering.")
        return
    try:
        import pydeck as pdk
    except Exception:
        st.dataframe(pd.DataFrame(markers), width="stretch")
        if sectors:
            st.dataframe(pd.DataFrame(sectors), width="stretch")
        if estimates:
            st.dataframe(pd.DataFrame(estimates), width="stretch")
        if tracks:
            st.dataframe(pd.DataFrame(tracks), width="stretch")
        return

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=35,
            pickable=True,
        )
    ]
    if sectors:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=sectors,
                get_polygon="polygon",
                get_fill_color="fill_color",
                get_line_color="line_color",
                line_width_min_pixels=1,
                pickable=True,
            )
        )
    if estimates:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=estimates,
                get_position="[lon, lat]",
                get_fill_color=[220, 30, 30, 120],
                get_radius="radius_m || 100",
                pickable=True,
            )
        )
    if tracks:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=tracks,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=65,
                pickable=True,
            )
        )
        layers.append(
            pdk.Layer(
                "TextLayer",
                data=tracks,
                get_position="[lon, lat]",
                get_text="label",
                get_color=[20, 20, 20, 230],
                get_size=14,
                get_alignment_baseline="'bottom'",
                pickable=True,
            )
        )
    center = markers[0]
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=center["lat"], longitude=center["lon"], zoom=13),
            tooltip={
                "text": "{station_id}\n{health}\n{last_status}\nML {ml_drone_pct}\nCombined {combined_drone_evidence_pct}\nBearing {bearing_deg}"
            },
        )
    )
