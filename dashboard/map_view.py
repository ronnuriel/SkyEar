import pandas as pd
import streamlit as st
import math


def bearing_ray_rows(events: list[dict], ray_length_m: float = 500.0) -> list[dict]:
    rows = []
    for event in events:
        loc = event.get("station_location") or {}
        bearing = event.get("estimated_azimuth_deg")
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
                "ray_end_lat": end_lat,
                "ray_end_lon": end_lon,
                "beam_score": event.get("beam_score"),
                "bearing_stable": event.get("bearing_stable"),
            }
        )
    return rows

def show_station_table(events: list[dict]):
    if not events:
        st.info("No events yet.")
        return
    rows = []
    for e in events:
        loc = e.get("station_location") or {}
        rows.append({
            "station_id": e.get("station_id"),
            "status": e.get("status"),
            "confidence": e.get("confidence"),
            "harmonic_score": e.get("harmonic_score"),
            "best_f0_hz": e.get("best_f0_hz"),
            "rms": e.get("rms"),
            "calibrated": e.get("calibrated"),
            "channel_agreement_count": e.get("channel_agreement_count"),
            "strongest_channel": e.get("strongest_channel"),
            "channel_count": e.get("channel_count"),
            "station_mode": e.get("station_mode"),
            "azimuth": e.get("estimated_azimuth_deg"),
            "lat": loc.get("latitude"),
            "lon": loc.get("longitude"),
        })
    st.dataframe(pd.DataFrame(rows).tail(100).iloc[::-1], width="stretch")
