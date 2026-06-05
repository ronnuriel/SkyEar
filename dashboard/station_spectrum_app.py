import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from dashboard.station_view import (
    health_badge_label,
    is_event_stale_for_fusion,
    operator_action_label,
    plot_spectrogram_figure,
    plot_spectrum_figure,
    render_decision_bars,
    status_label,
    timing_summary,
)


FUSION_WINDOW_SEC = 8.0


def _query_param(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _show_figure(fig):
    try:
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    except TypeError:
        st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def _select_station(default_station_id: str, station_ids: list[str]) -> str:
    if not station_ids:
        return st.sidebar.text_input("Station ID", value=default_station_id)

    index = station_ids.index(default_station_id) if default_station_id in station_ids else 0
    return st.sidebar.selectbox("Station ID", station_ids, index=index)


st.set_page_config(page_title="SkyEar Station Spectrum", layout="wide")

default_server_url = _query_param("server_url", "http://127.0.0.1:8080")
default_station_id = _query_param("station_id", "")

server_url = st.sidebar.text_input("Server URL", value=default_server_url)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_sec = st.sidebar.number_input(
    "Refresh every seconds",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5,
)

try:
    latest_by_station = requests.get(f"{server_url}/stations/latest", timeout=2).json()
    health_by_station = requests.get(f"{server_url}/stations/health", timeout=2).json()
except Exception as e:
    st.error(f"Could not load station data: {e}")
    latest_by_station = {}
    health_by_station = {}

station_ids = sorted(latest_by_station.keys())
station_id = _select_station(default_station_id, station_ids)
st.title(f"Live Spectrum - {station_id or 'select a station'}")

event = latest_by_station.get(station_id)
if not event:
    st.warning("No event found for this station yet")
    if station_ids:
        st.write("Available station IDs:")
        st.write(station_ids)
    else:
        st.info("Available station IDs: none")
else:
    metadata = event.get("metadata") or {}
    status = event.get("status", "background")
    label, kind = status_label(status)
    getattr(st, kind)(label)
    health = health_by_station.get(station_id)
    timing = timing_summary(event, health)
    st.caption(
        f"{health_badge_label(health)} | "
        f"Generated: {timing['generated']} | Received: {timing['received']} | "
        f"Event age: {timing['event_age']} | Latency: {timing['latency']} | "
        f"Fusion window: {FUSION_WINDOW_SEC:.0f}s"
    )
    if is_event_stale_for_fusion(event, FUSION_WINDOW_SEC):
        st.warning("Latest station event is stale for fusion")
    render_decision_bars(st, event)

    cols = st.columns(6)
    cols[0].metric("Station", event.get("station_id", station_id))
    cols[1].metric("Confidence", f"{float(event.get('confidence') or 0.0):.2f}")
    cols[2].metric("Harmonic", f"{float(event.get('harmonic_score') or 0.0):.1f}")
    cols[3].metric("Best f0", event.get("best_f0_hz") or "none")
    agreement = event.get("channel_agreement_count")
    channel_count = event.get("channel_count")
    cols[4].metric("Agreement", f"{agreement}/{channel_count}" if channel_count else "n/a")
    cols[5].metric("Calibrated", "yes" if event.get("calibrated") else "no")

    detail_cols = st.columns(4)
    detail_cols[0].metric("Strongest ch", event.get("strongest_channel") if event.get("strongest_channel") is not None else "n/a")
    detail_cols[1].metric("RMS", f"{float(event.get('rms') or 0.0):.4f}")
    detail_cols[2].metric("Duration", f"{float(event.get('duration_sec') or 0.0):.1f}s")
    detail_cols[3].metric("Age", timing["event_age"])

    bearing_cols = st.columns(4)
    bearing_cols[0].metric("Operator action", operator_action_label(0, event).upper())
    bearing_used = event.get("bearing_used_for_geo")
    tracked_bearing = event.get("tracked_bearing_deg")
    if bearing_used is False:
        bearing_display = "unreliable"
    elif tracked_bearing is not None:
        bearing_display = f"{float(tracked_bearing):.0f} deg"
    elif event.get("estimated_azimuth_deg") is not None:
        bearing_display = f"{float(event['estimated_azimuth_deg']):.0f} deg"
    else:
        bearing_display = "n/a"
    bearing_cols[1].metric("Bearing", bearing_display)
    bearing_cols[2].metric("Beam score", f"{float(event.get('beam_score') or 0.0):.3f}" if event.get("beam_score") is not None else "n/a")
    bearing_cols[3].metric("Bearing track", event.get("bearing_track_status") or ("stable" if event.get("bearing_stable") else "n/a"))

    if metadata.get("demo_phase"):
        st.info(f"Demo phase: {metadata['demo_phase']}")

    with st.expander("What am I seeing?", expanded=True):
        st.markdown(
            """
- Spectrum = energy by frequency for the latest audio window.
- Dashed vertical lines = detected harmonics f0, 2f0, 3f0...
- Spectrogram = frequency over time.
- Drone rotor signatures usually appear as persistent harmonic bands.
            """.strip()
        )

    st.subheader("Spectrum")
    fig = plot_spectrum_figure(metadata, small=False)
    if fig is None:
        st.warning("No spectrum metadata yet. Start updated station_agent.")
    else:
        _show_figure(fig)

    harmonic_lines = metadata.get("harmonic_lines") or []
    if harmonic_lines:
        st.subheader("Harmonic lines")
        st.dataframe(pd.DataFrame(harmonic_lines), width="stretch")

    st.subheader("Spectrogram")
    fig = plot_spectrogram_figure(metadata)
    if fig is None:
        st.warning("No spectrogram metadata yet. Start updated station_agent.")
    else:
        _show_figure(fig)

    st.subheader("Channel evidence")
    channel_evidence = event.get("channel_evidence") or []
    if channel_evidence:
        st.dataframe(pd.DataFrame(channel_evidence), width="stretch")
    else:
        st.info("No channel evidence yet.")

if auto_refresh:
    time.sleep(float(refresh_sec))
    st.rerun()
