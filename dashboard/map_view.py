import pandas as pd
import streamlit as st

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
            "azimuth": e.get("estimated_azimuth_deg"),
            "lat": loc.get("latitude"),
            "lon": loc.get("longitude"),
        })
    st.dataframe(pd.DataFrame(rows).tail(100).iloc[::-1], width="stretch")
