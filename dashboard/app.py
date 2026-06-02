import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st


STATUS_STYLE = {
    "background": ("BACKGROUND", "success"),
    "calibrating": ("CALIBRATING", "info"),
    "suspect": ("SUSPECT", "warning"),
    "drone_like": ("DRONE-LIKE", "warning"),
    "alert": ("ALERT", "error"),
}


def _status_label(status: str):
    label, kind = STATUS_STYLE.get(status, (status.upper(), "info"))
    getattr(st, kind)(label)


def _plot_spectrum(metadata: dict):
    freqs = metadata.get("spectrum_freqs_hz") or []
    db = metadata.get("spectrum_db") or []
    harmonic_lines = metadata.get("harmonic_lines") or []
    if not freqs or not db:
        st.warning("No spectrum metadata yet. Start updated station_agent.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(freqs, db, linewidth=1.1)
    for line in harmonic_lines:
        freq = line.get("freq_hz")
        if freq is not None:
            ax.axvline(float(freq), linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative dB")
    ax.set_ylim(-90, 3)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def _plot_spectrogram(metadata: dict):
    freqs = metadata.get("spectrogram_freqs_hz") or []
    times = metadata.get("spectrogram_times_sec") or []
    db = metadata.get("spectrogram_db") or []
    if not freqs or not times or not db:
        st.warning("No spectrogram metadata yet. Start updated station_agent.")
        return

    matrix = np.asarray(db, dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    extent = [min(times), max(times), min(freqs), max(freqs)]
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=-90,
        vmax=0,
    )
    for line in metadata.get("harmonic_lines") or []:
        freq = line.get("freq_hz")
        if freq is not None and min(freqs) <= float(freq) <= max(freqs):
            ax.axhline(float(freq), linestyle="--", linewidth=0.75, color="white", alpha=0.55)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="Relative dB")
    st.pyplot(fig)
    plt.close(fig)


def _status_reason(event: dict) -> list[str]:
    metadata = event.get("metadata") or {}
    score = float(event.get("harmonic_score") or 0.0)
    threshold = metadata.get("suspect_threshold")
    reasons = []
    if threshold is not None and score >= float(threshold):
        reasons.append("harmonic score above threshold")
    agreement = event.get("channel_agreement_count")
    channel_count = event.get("channel_count")
    if agreement is not None and channel_count:
        reasons.append(f"{agreement}/{channel_count} channels agree")
    best_f0 = event.get("best_f0_hz")
    if best_f0 is not None:
        reasons.append(f"best_f0 = {best_f0}Hz")
    return reasons


def _render_station_card(event: dict, show_spectrum: bool, show_spectrogram: bool):
    metadata = event.get("metadata") or {}
    status = event.get("status", "background")
    station_id = event.get("station_id", "unknown")
    station_name = event.get("station_name")

    with st.container(border=True):
        if status in {"suspect", "drone_like", "alert"}:
            st.markdown(f"### STATION {station_id} {status.upper()}")
        else:
            st.markdown(f"### {station_id}")
        if station_name:
            st.caption(station_name)
        _status_label(status)

        cols = st.columns(4)
        cols[0].metric("Confidence", f"{float(event.get('confidence') or 0.0):.2f}")
        cols[1].metric("Harmonic", f"{float(event.get('harmonic_score') or 0.0):.1f}")
        cols[2].metric("Best f0", event.get("best_f0_hz") or "none")
        cols[3].metric("RMS", f"{float(event.get('rms') or 0.0):.4f}")

        detail_cols = st.columns(3)
        agreement = event.get("channel_agreement_count")
        channel_count = event.get("channel_count")
        detail_cols[0].metric("Agreement", f"{agreement}/{channel_count}" if channel_count else "n/a")
        detail_cols[1].metric("Strongest ch", event.get("strongest_channel") if event.get("strongest_channel") is not None else "n/a")
        detail_cols[2].metric("Duration", f"{float(event.get('duration_sec') or 0.0):.1f}s")

        reasons = _status_reason(event)
        if status in {"suspect", "drone_like", "alert"} and reasons:
            st.caption(" | ".join(reasons))

        if show_spectrum:
            _plot_spectrum(metadata)
        if show_spectrogram:
            _plot_spectrogram(metadata)

        harmonic_lines = metadata.get("harmonic_lines") or []
        if harmonic_lines:
            st.dataframe(pd.DataFrame(harmonic_lines), width="stretch")


st.set_page_config(page_title="Drone Acoustic Network Dashboard", layout="wide")
st.title("Drone Acoustic Network Dashboard")
st.caption("Passive acoustic warning network - operator dashboard.")

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
show_spectrum = st.sidebar.checkbox("Show spectrum", value=True)
show_spectrogram = st.sidebar.checkbox("Show spectrogram", value=True)
max_stations_per_row = int(
    st.sidebar.number_input("Max stations per row", min_value=1, max_value=4, value=2, step=1)
)

try:
    st.sidebar.success(requests.get(f"{server_url}/health", timeout=1.5).json())
except Exception as e:
    st.sidebar.error(f"Server unavailable: {e}")

col1, col2, col3 = st.columns(3)
try:
    fusion = requests.get(f"{server_url}/fusion", timeout=2).json()
    level = fusion.get("level", 0)
    confidence = fusion.get("confidence", 0.0)
    reason = fusion.get("reason", "")

    if level >= 3:
        col1.error(f"LEVEL {level}")
    elif level == 2:
        col1.warning(f"LEVEL {level}")
    elif level == 1:
        col1.info(f"LEVEL {level}")
    else:
        col1.success("LEVEL 0")

    col2.metric("Fusion confidence", f"{confidence:.2f}")
    col3.write(reason)
except Exception as e:
    st.error(f"Could not load fusion: {e}")

st.subheader("Stations")
try:
    latest_by_station = requests.get(f"{server_url}/stations/latest", timeout=2).json()
    station_events = sorted(latest_by_station.values(), key=lambda item: item.get("station_id", ""))
    if not station_events:
        st.info("No station events yet.")

    for start in range(0, len(station_events), max_stations_per_row):
        row_events = station_events[start : start + max_stations_per_row]
        columns = st.columns(len(row_events))
        for column, event in zip(columns, row_events):
            with column:
                _render_station_card(event, show_spectrum, show_spectrogram)
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
