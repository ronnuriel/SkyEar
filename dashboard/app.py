import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from dashboard.map_view import bearing_ray_rows, render_passive_map
from dashboard.station_view import (
    health_badge_label,
    is_event_stale_for_fusion,
    operator_action_label,
    plot_spectrum_figure,
    render_decision_bars,
    status_label,
    timing_summary,
)


FUSION_WINDOW_SEC = 8.0
REFRESH_MODES = {
    "Normal": {"interval_sec": 2.0, "live": False, "state_ttl_sec": 2.0},
    "Live": {"interval_sec": 0.5, "live": True, "state_ttl_sec": 5.0},
    "Debug slow": {"interval_sec": 5.0, "live": False, "state_ttl_sec": 5.0},
    "Manual": {"interval_sec": 0.0, "live": False, "state_ttl_sec": 30.0},
}


def _query_param(name: str, default: str) -> str:
    value = st.query_params.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


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


def _get_json(server_url: str, path: str, *, timeout: float = 2.0) -> dict:
    return requests.get(f"{server_url}{path}", timeout=timeout).json()


def _cached_dashboard_state(server_url: str, *, ttl_sec: float, timeout: float = 2.5) -> dict:
    key = f"dashboard_state::{server_url}"
    now = time.time()
    cached = st.session_state.get(key)
    if cached and now - float(cached.get("fetched_unix", 0.0)) <= float(ttl_sec):
        return cached["payload"]
    payload = _get_json(server_url, "/dashboard/state", timeout=timeout)
    st.session_state[key] = {"fetched_unix": now, "payload": payload}
    return payload


def _cached_recording_state(server_url: str, station_id: str, *, ttl_sec: float = 10.0, force: bool = False) -> dict:
    key = f"recording_state::{server_url}::{station_id}"
    now = time.time()
    cached = st.session_state.get(key)
    if not force and cached and now - float(cached.get("fetched_unix", 0.0)) <= float(ttl_sec):
        return cached["payload"]
    payload = _get_json(server_url, f"/stations/{station_id}/recording/state", timeout=1.5)
    st.session_state[key] = {"fetched_unix": now, "payload": payload}
    return payload


def _is_simulation_station(event: dict | None, health: dict | None = None) -> bool:
    event = event or {}
    health = health or {}
    heartbeat = health.get("heartbeat") or {}
    latest_event = health.get("latest_event") or event
    metadata = {
        **((heartbeat.get("metadata") or {}) if isinstance(heartbeat, dict) else {}),
        **((latest_event.get("metadata") or {}) if isinstance(latest_event, dict) else {}),
        **(event.get("metadata") or {}),
    }
    return (
        str(event.get("station_mode") or latest_event.get("station_mode") or heartbeat.get("station_mode") or "") == "simulation"
        or str(metadata.get("source") or "").startswith("simulate_")
        or bool(metadata.get("scenario_id") in {"fiber_grid", "multi_target", "two_near_one_far"})
    )


def _schedule_autorefresh(enabled: bool, interval_sec: float, *, key: str = "dashboard_autorefresh") -> None:
    if not enabled or float(interval_sec) <= 0.0:
        return
    interval_ms = max(250, int(float(interval_sec) * 1000))
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=interval_ms, key=key)
    except Exception:
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {interval_ms});
            </script>
            """,
            height=0,
        )


def _live_url(server_url: str) -> str:
    url = str(server_url or "http://127.0.0.1:8080").rstrip("/")
    if url.endswith("/events"):
        url = url[: -len("/events")]
    return url + "/live"


def _render_recording_controls(
    station_id: str,
    server_url: str,
    event: dict,
    *,
    simulation_station: bool = False,
    poll_ttl_sec: float = 10.0,
) -> None:
    if simulation_station:
        st.caption("Recording controls hidden for simulation station.")
        return
    labels = [
        "background",
        "drone_off",
        "takeoff",
        "hover",
        "flyby",
        "landing",
        "car",
        "motorcycle",
        "helicopter",
        "fan",
        "unknown_noise",
    ]
    with st.expander("Recording controls"):
        st.warning("Recording may capture voices. Use only where permitted.")
        try:
            rec = _cached_recording_state(server_url, station_id, ttl_sec=poll_ttl_sec)
        except Exception:
            rec = {"state": {}, "pending_commands": []}
        state = rec.get("state") or {}
        cols = st.columns(3)
        cols[0].metric("Recording", "ON" if state.get("recording") else "OFF")
        cols[1].metric("Duration", f"{float(state.get('duration_sec') or 0.0):.0f}s")
        cols[2].metric("Folder", state.get("session_dir") or "n/a")
        if rec.get("pending_commands"):
            st.caption(f"Pending recording command(s): {len(rec['pending_commands'])}")
        if st.button("Refresh recording state", key=f"rec_refresh_{station_id}"):
            try:
                _cached_recording_state(server_url, station_id, ttl_sec=poll_ttl_sec, force=True)
            except Exception:
                pass
            st.rerun()
        session_name = st.text_input("Session name", value="", key=f"rec_session_{station_id}")
        label = st.selectbox("Marker label", labels, key=f"rec_label_{station_id}")
        note = st.text_input("Note", value="", key=f"rec_note_{station_id}")
        distance_m = st.number_input("Distance m", min_value=0.0, value=0.0, step=1.0, key=f"rec_dist_{station_id}")
        drone_model = st.text_input("Drone model", value="", key=f"rec_model_{station_id}")
        action_cols = st.columns(3)
        if action_cols[0].button("Start", key=f"rec_start_{station_id}"):
            requests.post(
                f"{server_url}/stations/{station_id}/recording/start",
                json={"session_name": session_name, "label": label, "note": note},
                timeout=2,
            )
            st.rerun()
        if action_cols[1].button("Stop", key=f"rec_stop_{station_id}"):
            requests.post(f"{server_url}/stations/{station_id}/recording/stop", json={}, timeout=2)
            st.rerun()
        if action_cols[2].button("Mark", key=f"rec_mark_{station_id}"):
            requests.post(
                f"{server_url}/stations/{station_id}/recording/mark",
                json={
                    "label": label,
                    "note": note,
                    "distance_m": None if distance_m <= 0 else distance_m,
                    "bearing_deg": event.get("tracked_bearing_deg") or event.get("estimated_azimuth_deg"),
                    "drone_model": drone_model,
                },
                timeout=2,
            )
            st.rerun()

def _render_station_card(
    station_id: str,
    event: dict,
    health: dict | None,
    fusion_level: int,
    show_inline_mini_spectrum: bool,
    server_url: str,
    *,
    live_mode: bool = False,
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
        side = metadata.get("two_mic_side")
        bearing_used = event.get("bearing_used_for_geo")
        tracked_bearing = event.get("tracked_bearing_deg")
        bearing_value = tracked_bearing if bearing_used is not False and tracked_bearing is not None else event.get("estimated_azimuth_deg")
        if bearing_used is False:
            bearing_display = "unreliable"
        elif bearing_value is not None:
            bearing_display = f"{float(bearing_value):.0f} deg"
        else:
            bearing_display = str(side or "n/a").upper()
        action_cols[1].metric("Bearing / Side", bearing_display)
        action_cols[2].metric("Beam score", f"{float(event.get('beam_score') or 0.0):.3f}" if event.get("beam_score") is not None else "n/a")
        action_cols[3].metric("Bearing track", event.get("bearing_track_status") or ("stable" if event.get("bearing_stable") else "n/a"))
        look_hint = event.get("two_mic_look_hint") or metadata.get("two_mic_look_hint")
        if look_hint:
            hidden_reason = metadata.get("two_mic_suppressed_reason")
            if hidden_reason or "HIDDEN" in str(look_hint):
                st.warning(f"Direction Hint: {look_hint}")
            else:
                st.info(f"Direction Hint: {look_hint}")
        if bearing_used is False:
            st.caption("Bearing unreliable" + (f": {event.get('bearing_reject_reason')}" if event.get("bearing_reject_reason") else ""))

        _render_recording_controls(
            station_id,
            server_url,
            event,
            simulation_station=_is_simulation_station(event, health),
            poll_ttl_sec=10.0 if live_mode else 5.0,
        )

        detail_cols = st.columns(4)
        agreement = event.get("channel_agreement_count")
        channel_count = event.get("channel_count")
        detail_cols[0].metric("Agreement", f"{agreement}/{channel_count}" if channel_count else "n/a")
        detail_cols[1].metric("Strongest ch", event.get("strongest_channel") if event.get("strongest_channel") is not None else "n/a")
        detail_cols[2].metric("RMS", f"{float(event.get('rms') or 0.0):.4f}")
        detail_cols[3].metric("Duration", f"{float(event.get('duration_sec') or 0.0):.1f}s")

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
        observations = track.get("observations") or []
        source_ids = track.get("source_ids") or sorted(
            {
                str(observation.get("source_hint_id") or (observation.get("metadata") or {}).get("simulated_source_id"))
                for observation in observations
                if observation.get("source_hint_id") is not None
                or (observation.get("metadata") or {}).get("simulated_source_id") is not None
            }
        )
        eta_values = [
            float((observation.get("metadata") or {})["target_eta_sec"])
            for observation in observations
            if (observation.get("metadata") or {}).get("target_eta_sec") is not None
        ]
        line_values = [
            str((observation.get("metadata") or {}).get("latest_line_crossed"))
            for observation in observations
            if (observation.get("metadata") or {}).get("latest_line_crossed") is not None
        ]
        eta_sec = track.get("target_eta_sec")
        if eta_sec is None and eta_values:
            eta_sec = min(eta_values)
        latest_line = track.get("latest_line_crossed") or (line_values[-1] if line_values else None)
        if source_ids or eta_sec is not None or latest_line:
            extra = []
            if source_ids:
                extra.append(f"sources={','.join(source_ids)}")
            if latest_line:
                extra.append(f"line={latest_line}")
            if eta_sec is not None:
                extra.append(f"ETA={float(eta_sec):.1f}s")
            st.caption(" | ".join(extra))
        if track.get("ambiguity"):
            st.caption(track.get("ambiguity"))
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

server_url = st.sidebar.text_input("Server URL", value=_query_param("server_url", "http://127.0.0.1:8080"))
live_tactical_url = _live_url(server_url)
st.sidebar.link_button("Open Live Tactical Map", live_tactical_url)
st.sidebar.caption(live_tactical_url)
refresh_mode = st.sidebar.selectbox("Refresh mode", list(REFRESH_MODES), index=list(REFRESH_MODES).index("Manual"))
refresh_cfg = REFRESH_MODES[refresh_mode]
live_mode = bool(refresh_cfg["live"])
if live_mode:
    st.sidebar.info("For smooth tactical map updates use the /live page. Streamlit still reruns the dashboard script.")
manual_refresh = st.sidebar.button("Refresh")
if manual_refresh:
    for cache_key in list(st.session_state):
        if str(cache_key).startswith(("dashboard_state::", "recording_state::")):
            del st.session_state[cache_key]
show_inline_mini_spectrum = st.sidebar.checkbox("Show inline mini spectrum", value=False)
show_bearing_sectors = st.sidebar.checkbox("Show bearing sectors", value=not live_mode)
show_global_uncertainty = st.sidebar.checkbox("Show global uncertainty estimate", value=False)
show_track_estimates = st.sidebar.checkbox("Show track estimates", value=True)
show_station_coverage = st.sidebar.checkbox("Show station coverage", value=False)
show_full_station_cards = st.sidebar.checkbox("Show full station cards", value=not live_mode)
max_stations_per_row = int(
    st.sidebar.number_input("Max stations per row", min_value=1, max_value=4, value=3, step=1)
)

dashboard_state: dict = {}
live_state: dict = {}
data_error: Exception | None = None
try:
    if live_mode:
        live_state = _get_json(server_url, "/dashboard/live", timeout=1.2)
        try:
            dashboard_state = _cached_dashboard_state(
                server_url,
                ttl_sec=float(refresh_cfg["state_ttl_sec"]),
                timeout=2.5,
            )
        except Exception:
            dashboard_state = {}
    else:
        dashboard_state = _cached_dashboard_state(
            server_url,
            ttl_sec=0.0 if manual_refresh else float(refresh_cfg["state_ttl_sec"]),
            timeout=2.5,
        )
    health_payload = dashboard_state.get("health") or {"ok": bool(live_state)}
    st.sidebar.success(health_payload)
except Exception as e:
    data_error = e
    st.sidebar.error(f"Server unavailable: {e}")

fusion = (live_state.get("fusion") if live_mode else dashboard_state.get("fusion")) or {}
map_state = (live_state.get("map_state") if live_mode else dashboard_state.get("map_state")) or {}
latest_by_station = dashboard_state.get("stations_latest") or {}
health_by_station = dashboard_state.get("stations_health") or {}
alerts = dashboard_state.get("alerts") or []
station_health_summary = live_state.get("stations_health_summary") or []

fusion_panel = st.container()
map_panel = st.container()
stations_panel = st.container()
alerts_panel = st.container()

fusion_level = 0
with fusion_panel:
    st.subheader("Fusion")
    col1, col2, col3 = st.columns(3)
    try:
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
        st.subheader("Active tracks")
        if tracks:
            track_cols = st.columns(min(3, len(tracks)))
            for idx, track in enumerate(tracks):
                with track_cols[idx % len(track_cols)]:
                    _draw_track_card(track)
        else:
            st.info("No active tracks.")
    except Exception as e:
        col1.error("LEVEL n/a")
        col2.metric("Fusion confidence", "n/a")
        col3.metric("Interpretation", "server unavailable")
        st.error(f"Could not load fusion: {e}")

with map_panel:
    st.subheader("Map / Passive Acoustic Situation")
    try:
        render_passive_map(
            st,
            map_state,
            show_bearing_sectors=show_bearing_sectors,
            show_global_estimates=show_global_uncertainty,
            show_track_estimates=show_track_estimates,
            show_station_coverage=show_station_coverage,
        )
    except Exception as e:
        st.info(f"Map / Passive Acoustic Situation unavailable: {e}")

with stations_panel:
    st.subheader("Stations")
    try:
        if live_mode and not show_full_station_cards:
            if station_health_summary:
                st.dataframe(pd.DataFrame(station_health_summary), width="stretch")
            else:
                st.info("Station summary will refresh on the slower dashboard state cache.")
            st.caption("Live mode keeps station cards collapsed for speed. Enable full station cards in the sidebar when needed.")
        else:
            if live_mode:
                st.caption("Station cards use a slower cache in Live mode to keep map/fusion responsive.")
        station_events = sorted(latest_by_station.items())
        heartbeat_only = sorted(
            (station_id, health)
            for station_id, health in health_by_station.items()
            if station_id not in latest_by_station
        )
        if not station_events and not (live_mode and not show_full_station_cards):
            st.info("No station events yet.")

        if not (live_mode and not show_full_station_cards):
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
                            show_inline_mini_spectrum,
                            server_url,
                            live_mode=live_mode,
                        )
            st.subheader("Online stations without recent events")
            if heartbeat_only:
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
            else:
                st.info("No heartbeat-only stations.")
            bearing_rows = bearing_ray_rows([event for _, event in station_events])
            st.subheader("Map bearing cues")
            if bearing_rows:
                st.dataframe(pd.DataFrame(bearing_rows), width="stretch")
            else:
                st.info("No bearing cues yet.")
    except Exception as e:
        st.error(f"Could not load stations: {e}")

    st.caption("Direction is only reliable for synchronized mic array profiles.")

with alerts_panel:
    st.subheader("Recent alerts")
    try:
        if alerts:
            st.dataframe(pd.DataFrame(alerts).iloc[::-1], width="stretch")
        else:
            st.info("No alerts yet.")
    except Exception as e:
        st.error(f"Could not load alerts: {e}")

if data_error is None:
    _schedule_autorefresh(
        refresh_mode != "Manual",
        float(refresh_cfg["interval_sec"]),
        key=f"dashboard_autorefresh_{refresh_mode}",
    )
