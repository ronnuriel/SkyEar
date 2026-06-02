import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from dashboard.map_view import show_station_table

st.set_page_config(page_title="Drone Acoustic Network Dashboard", layout="wide")
st.title("Drone Acoustic Network Dashboard")
st.caption("Passive acoustic warning network — operator dashboard.")

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

st.subheader("Recent events")
try:
    events = requests.get(f"{server_url}/events?limit=200", timeout=2).json()
    show_station_table(events)
    if events:
        latest = events[-1]
        metadata = latest.get("metadata", {})
        spectrum_freqs = metadata.get("spectrum_freqs_hz") or []
        spectrum_db = metadata.get("spectrum_db") or []
        harmonic_lines = metadata.get("harmonic_lines") or []

        st.subheader("Live spectrum")
        if spectrum_freqs and spectrum_db:
            metric_cols = st.columns(5)
            metric_cols[0].metric("Station", latest.get("station_id", "unknown"))
            metric_cols[1].metric("Status", latest.get("status", "unknown"))
            metric_cols[2].metric("Harmonic score", f"{float(latest.get('harmonic_score') or 0.0):.1f}")
            metric_cols[3].metric("Best f0", latest.get("best_f0_hz") or "none")
            metric_cols[4].metric("Confidence", f"{float(latest.get('confidence') or 0.0):.2f}")

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(spectrum_freqs, spectrum_db, linewidth=1.2)
            for line in harmonic_lines:
                freq = line.get("freq_hz")
                if freq is None:
                    continue
                ax.axvline(float(freq), linestyle="--", linewidth=0.9, alpha=0.5)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Relative level (dB)")
            ax.set_ylim(-90, 3)
            ax.grid(True, alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)

            if harmonic_lines:
                st.dataframe(pd.DataFrame(harmonic_lines), width="stretch")
        else:
            st.warning("No live spectrum metadata yet. Start updated station_agent.")

        channel_evidence = latest.get("channel_evidence") or []
        if channel_evidence:
            st.subheader("Latest per-channel evidence")
            st.dataframe(pd.DataFrame(channel_evidence), width="stretch")
    st.caption("Direction is only reliable for synchronized mic array profiles.")
except Exception as e:
    st.error(f"Could not load events: {e}")

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
