from __future__ import annotations

from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np


STATUS_STYLE = {
    "background": ("BACKGROUND", "success"),
    "calibrating": ("CALIBRATING", "info"),
    "suspect": ("SUSPECT", "warning"),
    "drone_like": ("DRONE-LIKE", "warning"),
    "alert": ("ALERT", "error"),
}


def status_label(status: str) -> tuple[str, str]:
    return STATUS_STYLE.get(status, (status.upper(), "info"))


def _score_value(event: dict, name: str) -> float | None:
    metadata = event.get("metadata") or {}
    value = event.get(name)
    if value is None:
        value = metadata.get(name)
    if value is None:
        return None
    return float(np.clip(float(value), 0.0, 1.0))


def _raw_value(event: dict, name: str):
    metadata = event.get("metadata") or {}
    value = event.get(name)
    if value is None:
        value = metadata.get(name)
    return value


def _int_value(event: dict, name: str) -> int:
    value = _raw_value(event, name)
    if value is None:
        return 0
    return int(value)


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(np.clip(value, 0.0, 1.0)) * 100:.0f}%"


def format_age(value: float | None) -> str:
    if value is None:
        return "n/a"
    value = max(0.0, float(value))
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.1f}s"


def format_latency(value: float | None) -> str:
    if value is None:
        return "n/a"
    value = max(0.0, float(value))
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


def format_unix_time(value: float | None) -> str:
    if value is None:
        return "n/a"
    return datetime.fromtimestamp(float(value)).strftime("%H:%M:%S")


def health_badge_label(health: dict | None) -> str:
    if not health:
        return "NO HEARTBEAT"
    state = str(health.get("alive_state") or "offline").upper()
    if not health.get("heartbeat"):
        return "NO HEARTBEAT"
    return state


def timing_summary(event: dict | None, health: dict | None = None, now: float | None = None) -> dict[str, str]:
    now = time.time() if now is None else float(now)
    event = event or {}
    metadata = event.get("metadata") or {}
    server_received = event.get("server_received_unix") or metadata.get("server_received_unix")
    generated = event.get("timestamp_unix")
    latency = metadata.get("station_to_server_latency_sec")
    if latency is None and health:
        latency = health.get("latency_sec")
    event_age = None if server_received is None else now - float(server_received)
    heartbeat_age = None if not health else health.get("heartbeat_age_sec")
    return {
        "generated": format_unix_time(generated),
        "received": format_unix_time(server_received),
        "event_age": format_age(event_age),
        "heartbeat_age": format_age(heartbeat_age),
        "latency": format_latency(latency),
    }


def is_event_stale_for_fusion(event: dict | None, fusion_window_sec: float = 8.0, now: float | None = None) -> bool:
    if not event:
        return False
    now = time.time() if now is None else float(now)
    metadata = event.get("metadata") or {}
    server_received = event.get("server_received_unix") or metadata.get("server_received_unix")
    timestamp = server_received or event.get("timestamp_unix")
    if timestamp is None:
        return False
    return now - float(timestamp) > float(fusion_window_sec)


def _combined_score(ml: float | None, harmonic: float | None) -> float:
    ml_value = float(np.clip(float(ml or 0.0), 0.0, 1.0))
    harmonic_value = float(np.clip(float(harmonic or 0.0), 0.0, 1.0))
    return float(np.clip((2.0 * ml_value * harmonic_value) / (ml_value + harmonic_value + 1e-6), 0.0, 1.0))


def decision_scores(event: dict) -> tuple[float, float | None, float]:
    harmonic = _score_value(event, "harmonic_evidence_pct_smoothed")
    if harmonic is None:
        harmonic = _score_value(event, "harmonic_evidence_pct")
    ml = _score_value(event, "ml_drone_pct_smoothed")
    if ml is None:
        ml = _score_value(event, "ml_drone_pct")
    if ml is None:
        ml = _score_value(event, "hf_p_drone")
    combined = _score_value(event, "combined_drone_evidence_pct")
    if combined is None:
        combined = _combined_score(ml, harmonic)
    return (harmonic or 0.0), ml, combined


def decision_score_values(event: dict) -> dict[str, float | None]:
    raw = _score_value(event, "harmonic_evidence_pct_raw")
    smoothed = _score_value(event, "harmonic_evidence_pct_smoothed")
    if raw is None:
        raw = _score_value(event, "harmonic_evidence_pct")
    if smoothed is None:
        smoothed = _score_value(event, "harmonic_evidence_pct")
    ml = _score_value(event, "ml_drone_pct_smoothed")
    if ml is None:
        ml = _score_value(event, "ml_drone_pct")
    if ml is None:
        ml = _score_value(event, "hf_p_drone")
    combined = _score_value(event, "combined_drone_evidence_pct")
    if combined is None:
        combined = _combined_score(ml, smoothed)
    return {"harmonic_raw": raw, "harmonic_smoothed": smoothed, "ml": ml, "combined": combined}


OPERATOR_LABEL_TEXT = {
    "background": "BACKGROUND",
    "acoustic_harmonic_source": "ACOUSTIC HARMONIC SOURCE",
    "acoustic_drone_watch": "ACOUSTIC DRONE WATCH",
    "weak_local_candidate": "WEAK LOCAL CANDIDATE",
    "non_drone_harmonic": "NON-DRONE HARMONIC",
    "ml_drone_candidate": "ML DRONE CANDIDATE",
    "local_drone_candidate": "LOCAL DRONE CANDIDATE",
    "strong_local_candidate": "STRONG LOCAL CANDIDATE",
    "drone_like": "DRONE-LIKE",
    "alert": "ALERT",
}


def operator_label(event: dict) -> str:
    metadata = event.get("metadata") or {}
    return str(event.get("operator_label") or metadata.get("operator_label") or "")


def format_operator_label(value: str) -> str:
    return OPERATOR_LABEL_TEXT.get(value, value.replace("_", " ").upper() if value else "BACKGROUND")


def operator_action_label(fusion_level: int, event: dict | None = None) -> str:
    event = event or {}
    status = str(event.get("status") or "").lower()
    label = operator_label(event)
    if int(fusion_level or 0) >= 3 or status == "alert" or label == "alert":
        return "take cover"
    if int(fusion_level or 0) >= 1 or status in {"suspect", "drone_like"} or label in {
        "ml_drone_candidate",
        "acoustic_drone_watch",
        "weak_local_candidate",
        "local_drone_candidate",
        "strong_local_candidate",
        "drone_like",
    }:
        return "observe"
    return "all clear"


def decision_display_state(event: dict) -> dict[str, str]:
    harmonic, ml, combined = decision_scores(event)
    ml_value = ml if ml is not None else 0.0
    label = operator_label(event)
    if label == "alert":
        return {"label": format_operator_label(label), "harmonic_color": "#dc2626", "ml_color": "#dc2626", "combined_color": "#dc2626"}
    if label == "acoustic_harmonic_source":
        return {"label": "ACOUSTIC HARMONIC SOURCE", "harmonic_color": "#ca8a04", "ml_color": "#6b7280", "combined_color": "#6b7280"}
    if label == "drone_like":
        return {"label": "DRONE-LIKE", "harmonic_color": "#dc2626", "ml_color": "#ea580c", "combined_color": "#dc2626"}
    if label == "strong_local_candidate":
        return {"label": "STRONG LOCAL CANDIDATE", "harmonic_color": "#ea580c", "ml_color": "#2563eb", "combined_color": "#ea580c"}
    if label == "local_drone_candidate":
        return {"label": "LOCAL DRONE CANDIDATE", "harmonic_color": "#ca8a04", "ml_color": "#2563eb", "combined_color": "#ca8a04"}
    if label == "weak_local_candidate":
        return {"label": "WEAK LOCAL CANDIDATE", "harmonic_color": "#ca8a04", "ml_color": "#2563eb", "combined_color": "#ca8a04"}
    if label == "acoustic_drone_watch":
        return {"label": "ACOUSTIC DRONE WATCH", "harmonic_color": "#ca8a04", "ml_color": "#2563eb", "combined_color": "#a3a3a3"}
    if combined >= 0.60:
        return {"label": "STRONG ML DRONE CANDIDATE", "harmonic_color": "#ea580c", "ml_color": "#2563eb", "combined_color": "#dc2626"}
    if harmonic >= 0.60 and ml_value >= 0.60:
        return {"label": "ML + harmonic agree", "harmonic_color": "#dc2626", "ml_color": "#ea580c", "combined_color": "#dc2626"}
    if label == "non_drone_harmonic" or (harmonic >= 0.60 and ml_value <= 0.30):
        return {"label": "NON-DRONE HARMONIC", "harmonic_color": "#ca8a04", "ml_color": "#6b7280", "combined_color": "#6b7280"}
    if label == "ml_drone_candidate" or (ml_value >= 0.60 and harmonic <= 0.30):
        return {"label": "ML DRONE CANDIDATE", "harmonic_color": "#a3a3a3", "ml_color": "#2563eb", "combined_color": "#ca8a04"}
    return {"label": "background", "harmonic_color": "#16a34a", "ml_color": "#22c55e", "combined_color": "#16a34a"}


def render_decision_bars(st_module, event: dict) -> None:
    scores = decision_score_values(event)
    harmonic_raw = scores["harmonic_raw"] or 0.0
    harmonic_smoothed = scores["harmonic_smoothed"] or 0.0
    ml = scores["ml"]
    combined = scores["combined"] or 0.0
    display = decision_display_state(event)
    ml_width = 0.0 if ml is None else ml
    reason = event.get("decision_reason") or (event.get("metadata") or {}).get("decision_reason")
    reason_html = f"<div class='sky-score-reason'>{reason}</div>" if reason else ""
    candidate_run = _int_value(event, "candidate_run")
    ml_positive_run = _int_value(event, "ml_positive_run")
    strong_run = _int_value(event, "strong_run")
    delay = _raw_value(event, "estimated_detection_delay_sec")
    delay_text = "n/a" if delay is None else f"{float(delay):.1f}s"
    hf_error = bool(_raw_value(event, "hf_error"))
    if hf_error:
        st_module.warning("HF unavailable — harmonic-only mode, alert disabled")
    st_module.markdown(
        f"""
<div class="sky-score-box">
  <div class="sky-score-label">{display['label']}</div>
  <div class="sky-score-row"><span>Harmonic evidence raw/smoothed</span><b>{format_pct(harmonic_raw)} / {format_pct(harmonic_smoothed)}</b></div>
  <div class="sky-score-track"><div style="width:{harmonic_smoothed * 100:.0f}%;background:{display['harmonic_color']};"></div></div>
  <div class="sky-score-row"><span>ML drone probability</span><b>{format_pct(ml)}</b></div>
  <div class="sky-score-track"><div style="width:{ml_width * 100:.0f}%;background:{display['ml_color']};"></div></div>
  <div class="sky-score-row"><span>Combined drone evidence</span><b>{format_pct(combined)}</b></div>
  <div class="sky-score-track"><div style="width:{combined * 100:.0f}%;background:{display['combined_color']};"></div></div>
  <div class="sky-score-row"><span>Persistence</span><b>candidate={candidate_run} ML={ml_positive_run} strong={strong_run} delay={delay_text}</b></div>
  {reason_html}
</div>
<style>
.sky-score-box {{ margin: .35rem 0 .65rem 0; }}
.sky-score-label {{ font-size: .8rem; font-weight: 700; margin-bottom: .2rem; }}
.sky-score-row {{ display:flex; justify-content:space-between; gap:.5rem; font-size:.75rem; }}
.sky-score-track {{ height: .45rem; background:#e5e7eb; border-radius: 999px; overflow:hidden; margin:.1rem 0 .3rem 0; }}
.sky-score-track div {{ height: 100%; border-radius: 999px; }}
.sky-score-reason {{ color:#6b7280; font-size:.72rem; line-height:1.2; margin-top:.2rem; }}
</style>
        """.strip(),
        unsafe_allow_html=True,
    )


def plot_spectrum_figure(metadata: dict, small: bool = False):
    freqs = metadata.get("spectrum_freqs_hz") or []
    db = metadata.get("spectrum_db") or []
    harmonic_lines = metadata.get("harmonic_lines") or []
    if not freqs or not db:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 1.8) if small else (12.0, 4.0))
    ax.plot(freqs, db, linewidth=1.0)
    for line in harmonic_lines:
        freq = line.get("freq_hz")
        if freq is None:
            continue
        freq = float(freq)
        ax.axvline(freq, linestyle="--", linewidth=0.9, alpha=0.55)
        if not small:
            ax.text(freq, 1.5, f"{line.get('k')}x", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative dB")
    ax.set_ylim(-90, 3)
    ax.grid(True, alpha=0.25)
    return fig


def plot_spectrogram_figure(metadata: dict):
    freqs = metadata.get("spectrogram_freqs_hz") or []
    times = metadata.get("spectrogram_times_sec") or []
    db = metadata.get("spectrogram_db") or []
    if not freqs or not times or not db:
        return None

    matrix = np.asarray(db, dtype=float)
    fig, ax = plt.subplots(figsize=(12.0, 5.0))
    extent = [min(times), max(times), min(freqs), max(freqs)]
    image = ax.imshow(
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
    fig.colorbar(image, ax=ax, label="Relative dB")
    return fig
