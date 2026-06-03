import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from dashboard.map_view import bearing_ray_rows
from dashboard.station_view import (
    external_spectrum_app_url,
    health_badge_label,
    is_event_stale_for_fusion,
    operator_action_label,
    plot_spectrum_figure,
    render_decision_bars,
    status_label,
    timing_summary,
)


FUSION_WINDOW_SEC = 8.0


def _draw_status(status: str):
    label, kind = status_label(status)
    getattr(st, kind)(label)


def _draw_health_badge(health: dict | None):
    label = health_badge_label(health)
    if label == "ONLINE":
        st.success(label)
    elif label == "STALE":
        st.warning(label)
    elif label == "NO HEARTBEAT":
        st.warning(label)
    else:
        st.error(label)


def _render_station_card(
    station_id: str,
    event: dict,
    health: dict | None,
    fusion_level: int,
    server_url: str,
    spectrum_app_url: str,
    show_inline_mini_spectrum: bool,
):
    metadata = event.get("metadata") or {}
    station_id = event.get("station_id") or station_id
    station_name = event.get("station_name")
    status = event.get("status", "background")
    demo_phase = metadata.get("demo_phase")

    with st.container(border=True):
        header_cols = st.columns([2, 1])
        header_cols[0].markdown(f"### {station_id}")
        with header_cols[1]:
            _draw_health_badge(health)
        if station_name:
            header_cols[0].caption(station_name)
        _draw_status(status)
        timing = timing_summary(event, health)
        timing_cols = st.columns(4)
        timing_cols[0].metric("Last heartbeat", timing["heartbeat_age"])
        timing_cols[1].metric("Last event", timing["event_age"])
        timing_cols[2].metric("Latency", timing["latency"])
        timing_cols[3].metric("Received", timing["received"])
        st.caption(
            f"Generated: {timing['generated']} | Received: {timing['received']} | "
            f"Fusion window: {FUSION_WINDOW_SEC:.0f}s"
        )
        if is_event_stale_for_fusion(event, FUSION_WINDOW_SEC):
            st.warning("Latest station event is stale for fusion")
        if demo_phase:
            st.caption(f"Demo phase: {demo_phase}")
        render_decision_bars(st, event)

        cols = st.columns(4)
        cols[0].metric("Confidence", f"{float(event.get('confidence') or 0.0):.2f}")
        cols[1].metric("Harmonic", f"{float(event.get('harmonic_score') or 0.0):.1f}")
        cols[2].metric("Best f0", event.get("best_f0_hz") or "none")
        cols[3].metric("Fusion", f"LEVEL {fusion_level}")

        action_cols = st.columns(4)
        action_cols[0].metric("Operator action", operator_action_label(fusion_level, event).upper())
        action_cols[1].metric("Bearing", event.get("estimated_azimuth_deg") if event.get("estimated_azimuth_deg") is not None else "n/a")
        action_cols[2].metric("Beam score", f"{float(event.get('beam_score') or 0.0):.3f}" if event.get("beam_score") is not None else "n/a")
        action_cols[3].metric("Bearing stable", "yes" if event.get("bearing_stable") else "no")

        detail_cols = st.columns(4)
        agreement = event.get("channel_agreement_count")
        channel_count = event.get("channel_count")
        detail_cols[0].metric("Agreement", f"{agreement}/{channel_count}" if channel_count else "n/a")
        detail_cols[1].metric("Strongest ch", event.get("strongest_channel") if event.get("strongest_channel") is not None else "n/a")
        detail_cols[2].metric("RMS", f"{float(event.get('rms') or 0.0):.4f}")
        detail_cols[3].metric("Duration", f"{float(event.get('duration_sec') or 0.0):.1f}s")

        st.markdown(f"[Open Spectrum]({external_spectrum_app_url(station_id, server_url, spectrum_app_url)})")

        if show_inline_mini_spectrum:
            fig = plot_spectrum_figure(metadata, small=True)
            if fig is None:
                st.caption("No spectrum metadata yet.")
            else:
                st.pyplot(fig)
                plt.close(fig)


def _draw_track_card(track: dict):
    level = int(track.get("level", 0))
    title = f"{track.get('track_id', 'track')} - LEVEL {level}"
    with st.container(border=True):
        if level >= 3:
            st.error(title)
        elif level == 2:
            st.warning(title)
        elif level == 1:
            st.info(title)
        else:
            st.success(title)
        st.caption(track.get("interpretation") or "track")
        cols = st.columns(3)
        cols[0].metric("Stations", ", ".join(track.get("station_ids") or []))
        cols[1].metric("Confidence", f"{float(track.get('confidence') or 0.0):.2f}")
        cols[2].metric("Same f0", "yes" if track.get("same_f0") else "no")
        estimated_source = track.get("estimated_source") or {}
        if estimated_source:
            st.caption(
                "Estimated source: "
                f"{estimated_source.get('latitude')}, {estimated_source.get('longitude')} "
                f"({estimated_source.get('source')})"
            )
        st.caption(track.get("reason") or "")


st.set_page_config(page_title="SkyEar Operator Dashboard", layout="wide")
st.title("SkyEar Operator Dashboard")
st.caption("Passive acoustic warning network - tactical overview.")

server_url = st.sidebar.text_input("Server URL", value="http://127.0.0.1:8080")
st.sidebar.button("Refresh")
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_sec = st.sidebar.number_input(
    "Refresh every seconds",
    min_value=0.5,
    max_value=10.0,
    value=1.5,
    step=0.5,
)
show_inline_mini_spectrum = st.sidebar.checkbox("Show inline mini spectrum", value=False)
spectrum_app_url = st.sidebar.text_input("Spectrum app URL", value="http://localhost:8502")
max_stations_per_row = int(
    st.sidebar.number_input("Max stations per row", min_value=1, max_value=4, value=3, step=1)
)

try:
    st.sidebar.success(requests.get(f"{server_url}/health", timeout=1.5).json())
except Exception as e:
    st.sidebar.error(f"Server unavailable: {e}")

col1, col2, col3 = st.columns(3)
fusion_level = 0
try:
    fusion = requests.get(f"{server_url}/fusion", timeout=2).json()
    fusion_level = int(fusion.get("level", 0))
    confidence = float(fusion.get("confidence", 0.0))
    reason = fusion.get("reason", "")
    interpretation = fusion.get("interpretation") or "background"
    tracks = fusion.get("tracks") or []

    if fusion_level >= 3:
        col1.error(f"LEVEL {fusion_level}")
    elif fusion_level == 2:
        col1.warning(f"LEVEL {fusion_level}")
    elif fusion_level == 1:
        col1.info(f"LEVEL {fusion_level}")
    else:
        col1.success("LEVEL 0")

    col2.metric("Fusion confidence", f"{confidence:.2f}")
    col3.metric("Interpretation", interpretation)
    st.metric("Recommended operator action", operator_action_label(fusion_level).upper())
    st.caption(f"{reason} | Active tracks: {len(tracks)}")
    if tracks:
        st.subheader("Active tracks")
        track_cols = st.columns(min(3, len(tracks)))
        for idx, track in enumerate(tracks):
            with track_cols[idx % len(track_cols)]:
                _draw_track_card(track)
except Exception as e:
    st.error(f"Could not load fusion: {e}")

st.subheader("Stations")
try:
    latest_by_station = requests.get(f"{server_url}/stations/latest", timeout=2).json()
    health_by_station = requests.get(f"{server_url}/stations/health", timeout=2).json()
    station_events = sorted(latest_by_station.items())
    heartbeat_only = sorted(
        (station_id, health)
        for station_id, health in health_by_station.items()
        if station_id not in latest_by_station
    )
    if not station_events:
        st.info("No station events yet.")

    for start in range(0, len(station_events), max_stations_per_row):
        row_events = station_events[start : start + max_stations_per_row]
        columns = st.columns(len(row_events))
        for column, (station_id, event) in zip(columns, row_events):
            with column:
                _render_station_card(
                    station_id,
                    event,
                    health_by_station.get(station_id),
                    fusion_level,
                    server_url,
                    spectrum_app_url,
                    show_inline_mini_spectrum,
                )
    if heartbeat_only:
        st.subheader("Online stations without recent events")
        for start in range(0, len(heartbeat_only), max_stations_per_row):
            row_items = heartbeat_only[start : start + max_stations_per_row]
            columns = st.columns(len(row_items))
            for column, (station_id, health) in zip(columns, row_items):
                with column:
                    with st.container(border=True):
                        st.markdown(f"### {station_id}")
                        _draw_health_badge(health)
                        timing = timing_summary(None, health)
                        st.metric("Last heartbeat", timing["heartbeat_age"])
                        st.metric("Latency", timing["latency"])
                        st.caption("Station is online/background; no acoustic event received yet.")
    bearing_rows = bearing_ray_rows([event for _, event in station_events])
    if bearing_rows:
        st.subheader("Map bearing cues")
        st.dataframe(pd.DataFrame(bearing_rows), width="stretch")
except Exception as e:
    st.error(f"Could not load stations: {e}")

st.caption("Direction is only reliable for synchronized mic array profiles.")

st.subheader("Recent alerts")
try:
    alerts = requests.get(f"{server_url}/alerts?limit=50", timeout=2).json()
    if alerts:
        st.dataframe(pd.DataFrame(alerts).iloc[::-1], width="stretch")
    else:
        st.info("No alerts yet.")
except Exception as e:
    st.error(f"Could not load alerts: {e}")

if auto_refresh:
    time.sleep(float(refresh_sec))
    st.rerun()
