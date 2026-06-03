from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.station_view import format_pct, render_decision_bars, status_label
from station.local_monitor import is_stale_state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local per-station SkyEar monitor.")
    parser.add_argument("--state", default="runtime/stations/station_001_latest.json")
    parser.add_argument("--history")
    parser.add_argument("--refresh-sec", type=float, default=0.5)
    args, _ = parser.parse_known_args(argv)
    return args


def load_snapshot(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    return parse_snapshot(snapshot)


def parse_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    event = snapshot.get("event") or {}
    return {
        "updated_unix": snapshot.get("updated_unix"),
        "event": event,
        "audio": snapshot.get("audio") or {},
        "spectrum": snapshot.get("spectrum") or {},
        "spectrogram": snapshot.get("spectrogram") or {},
        "harmonic_lines": snapshot.get("harmonic_lines") or [],
        "hf": snapshot.get("hf") or {},
        "beam": snapshot.get("beam") or {},
        "server": snapshot.get("server") or {},
        "station_id": event.get("station_id") or "unknown station",
        "station_name": event.get("station_name"),
    }


def load_history_rows(path: str | Path | None, limit: int = 500) -> list[dict[str, Any]]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def snapshot_is_stale(snapshot: dict[str, Any], stale_after_sec: float = 3.0, now: float | None = None) -> bool:
    return is_stale_state(snapshot, stale_after_sec=stale_after_sec, now=now)


def _metric_pct(value: float | None) -> str:
    return format_pct(None if value is None else float(value))


def _event_value(event: dict[str, Any], name: str):
    metadata = event.get("metadata") or {}
    value = event.get(name)
    return metadata.get(name) if value is None else value


def _show_status(event: dict[str, Any]) -> None:
    status = str(event.get("status") or "background")
    label, kind = status_label(status)
    getattr(st, kind)(label)
    render_decision_bars(st, event)


def _show_waveform(audio: dict[str, Any]) -> None:
    waveform = audio.get("waveform") or []
    if not waveform:
        st.info("No waveform snapshot yet.")
        return
    st.line_chart(pd.DataFrame({"amplitude": waveform}))


def _show_spectrum(spectrum: dict[str, Any], harmonic_lines: list[dict[str, Any]]) -> None:
    freqs = spectrum.get("spectrum_freqs_hz") or []
    db = spectrum.get("spectrum_db") or []
    if not freqs or not db:
        st.info("No spectrum snapshot yet.")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, db, linewidth=1.0)
    for line in harmonic_lines:
        freq = line.get("freq_hz")
        if freq is not None:
            ax.axvline(float(freq), linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative dB")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)
    plt.close(fig)


def _show_spectrogram(spectrogram: dict[str, Any]) -> None:
    freqs = spectrogram.get("spectrogram_freqs_hz") or []
    times = spectrogram.get("spectrogram_times_sec") or []
    matrix = spectrogram.get("spectrogram_db") or []
    if not freqs or not times or not matrix:
        st.info("No spectrogram snapshot yet.")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(
        np.asarray(matrix, dtype=float),
        aspect="auto",
        origin="lower",
        extent=[min(times), max(times), min(freqs), max(freqs)],
        cmap="magma",
        vmin=-90,
        vmax=0,
    )
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Relative dB")
    st.pyplot(fig)
    plt.close(fig)


def _show_history(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No local history rows yet.")
        return
    df = pd.DataFrame(rows)
    available = [
        column
        for column in ("harmonic_evidence_pct_smoothed", "ml_drone_pct", "combined_drone_evidence_pct", "rms")
        if column in df.columns
    ]
    if available:
        st.line_chart(df[available])
    st.dataframe(df.tail(25), width="stretch")


def _show_warnings(snapshot: dict[str, Any], event: dict[str, Any], hf: dict[str, Any], server: dict[str, Any]) -> None:
    if snapshot_is_stale(snapshot):
        st.warning("Local station state is stale")
    if hf.get("error") or event.get("hf_error"):
        st.warning("HF unavailable — harmonic-only mode, alert disabled")
    if server.get("last_send_error"):
        st.warning(f"Server send failed: {server['last_send_error']}")
    if server.get("last_heartbeat_error"):
        st.warning(f"Heartbeat failed: {server['last_heartbeat_error']}")
    peak = float(event.get("peak") or 0.0)
    rms = float(event.get("rms") or 0.0)
    if peak >= 0.95:
        st.warning("Input peak is very high; clipping risk")
    if rms <= 1e-5:
        st.warning("Very low RMS; microphone may be disconnected or muted")


def main() -> None:
    args = parse_args()
    state_path = Path(args.state)
    history_path = Path(args.history) if args.history else state_path.with_name(
        state_path.name.replace("_latest.json", "_history.jsonl")
    )

    st.set_page_config(page_title="SkyEar Local Station Monitor", layout="wide")
    st.sidebar.text_input("State JSON", value=str(state_path))
    st.sidebar.text_input("History JSONL", value=str(history_path))
    auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)

    try:
        if not state_path.exists():
            raise FileNotFoundError(
                f"Local station state not found: {state_path}. Start skyear-station first or pass --state PATH."
            )
        snapshot = load_snapshot(state_path)
    except Exception as exc:
        st.error(f"Could not read local station state: {exc}")
        if auto_refresh:
            time.sleep(float(args.refresh_sec))
            st.rerun()
        return

    event = snapshot["event"]
    hf = snapshot["hf"]
    beam = snapshot["beam"]
    server = snapshot["server"]
    title = snapshot["station_id"]
    if snapshot.get("station_name"):
        title += f" - {snapshot['station_name']}"
    st.title(title)
    st.caption(f"Updated: {time.strftime('%H:%M:%S', time.localtime(float(snapshot.get('updated_unix') or 0.0)))}")
    _show_warnings(snapshot, event, hf, server)
    _show_status(event)

    cols = st.columns(6)
    cols[0].metric("HF drone", _metric_pct(event.get("hf_p_drone") or hf.get("p_drone")))
    cols[1].metric("Harmonic", _metric_pct(_event_value(event, "harmonic_evidence_pct_smoothed")))
    cols[2].metric("Combined", _metric_pct(_event_value(event, "combined_drone_evidence_pct")))
    cols[3].metric("f0", event.get("best_f0_hz") or "none")
    cols[4].metric("RMS", f"{float(event.get('rms') or 0.0):.5f}")
    cols[5].metric("Candidate run", _event_value(event, "candidate_run") or 0)

    beam_cols = st.columns(5)
    beam_cols[0].metric("Bearing", beam.get("estimated_azimuth_deg") if beam.get("estimated_azimuth_deg") is not None else "n/a")
    beam_cols[1].metric("Beam score", f"{float(beam.get('beam_score') or 0.0):.3f}" if beam.get("beam_score") is not None else "n/a")
    beam_cols[2].metric("Beam SNR", f"{float(beam.get('beam_snr_gain_db') or 0.0):.1f} dB" if beam.get("beam_snr_gain_db") is not None else "n/a")
    beam_cols[3].metric("Bearing stable", "yes" if beam.get("bearing_stable") else "no")
    beam_cols[4].metric("Server", "send failed" if server.get("last_send_error") else "ok")

    tab_wave, tab_spec, tab_sgram, tab_history = st.tabs(["Waveform", "Spectrum", "Spectrogram", "Evidence History"])
    with tab_wave:
        _show_waveform(snapshot["audio"])
    with tab_spec:
        _show_spectrum(snapshot["spectrum"], snapshot["harmonic_lines"])
    with tab_sgram:
        _show_spectrogram(snapshot["spectrogram"])
    with tab_history:
        _show_history(load_history_rows(history_path))

    if auto_refresh:
        time.sleep(float(args.refresh_sec))
        st.rerun()


if __name__ == "__main__":
    main()
