import requests
import pandas as pd
import streamlit as st
from dashboard.map_view import show_station_table

st.set_page_config(page_title="Drone Acoustic Network Dashboard", layout="wide")
st.title("Drone Acoustic Network Dashboard")
st.caption("Passive acoustic warning network — operator dashboard.")

server_url = st.sidebar.text_input("Server URL", value="http://127.0.0.1:8080")
st.sidebar.button("Refresh")

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
