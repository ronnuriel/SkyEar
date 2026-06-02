import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from dashboard.station_view import plot_spectrogram_figure, plot_spectrum_figure, status_label


def _query_param(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


st.set_page_config(page_title="SkyEar Station Spectrum", layout="wide")

default_server_url = _query_param("server_url", "http://127.0.0.1:8080")
default_station_id = _query_param("station_id", "")

server_url = st.sidebar.text_input("Server URL", value=default_server_url)
station_id = st.sidebar.text_input("Station ID", value=default_station_id)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_sec = st.sidebar.number_input(
    "Refresh every seconds",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5,
)

st.title(f"Live Spectrum - {station_id or 'select a station'}")

try:
    latest_by_station = requests.get(f"{server_url}/stations/latest", timeout=2).json()
except Exception as e:
    st.error(f"Could not load station data: {e}")
    latest_by_station = {}

if not station_id and latest_by_station:
    station_id = sorted(latest_by_station)[0]

event = latest_by_station.get(station_id)
if not event:
    st.warning("No event found for this station yet.")
else:
    metadata = event.get("metadata") or {}
    status = event.get("status", "background")
    label, kind = status_label(status)
    getattr(st, kind)(label)

    cols = st.columns(6)
    cols[0].metric("Confidence", f"{float(event.get('confidence') or 0.0):.2f}")
    cols[1].metric("Harmonic", f"{float(event.get('harmonic_score') or 0.0):.1f}")
    cols[2].metric("Best f0", event.get("best_f0_hz") or "none")
    agreement = event.get("channel_agreement_count")
    channel_count = event.get("channel_count")
    cols[3].metric("Agreement", f"{agreement}/{channel_count}" if channel_count else "n/a")
    cols[4].metric("Strongest ch", event.get("strongest_channel") if event.get("strongest_channel") is not None else "n/a")
    cols[5].metric("Calibrated", "yes" if event.get("calibrated") else "no")

    age = time.time() - float(event.get("timestamp_unix") or time.time())
    st.caption(f"Latest event age: {age:.1f}s")
    if metadata.get("demo_phase"):
        st.info(f"Demo phase: {metadata['demo_phase']}")

    with st.expander("What am I seeing?", expanded=True):
        st.markdown(
            """
- Spectrum: magnitude by frequency for the latest audio window.
- Dashed vertical lines: detected harmonic stack f0, 2f0, 3f0...
- Spectrogram: frequency over time; rotor signatures appear as persistent horizontal or diagonal bands.
- Channel evidence: how many channels or microphones agree.
            """.strip()
        )

    st.subheader("Spectrum")
    fig = plot_spectrum_figure(metadata, small=False)
    if fig is None:
        st.warning("No spectrum metadata yet. Start updated station_agent.")
    else:
        st.pyplot(fig)
        plt.close(fig)

    harmonic_lines = metadata.get("harmonic_lines") or []
    if harmonic_lines:
        st.dataframe(pd.DataFrame(harmonic_lines), width="stretch")

    st.subheader("Spectrogram")
    fig = plot_spectrogram_figure(metadata)
    if fig is None:
        st.warning("No spectrogram metadata yet. Start updated station_agent.")
    else:
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Channel evidence")
    channel_evidence = event.get("channel_evidence") or []
    if channel_evidence:
        st.dataframe(pd.DataFrame(channel_evidence), width="stretch")
    else:
        st.info("No channel evidence yet.")

if auto_refresh:
    time.sleep(float(refresh_sec))
    st.rerun()
