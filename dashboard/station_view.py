from __future__ import annotations

from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np


STATUS_STYLE = {
    "background": ("BACKGROUND", "success"),
    "calibrating": ("CALIBRATING", "info"),
    "suspect": ("SUSPECT", "warning"),
    "drone_like": ("DRONE-LIKE", "warning"),
    "alert": ("ALERT", "error"),
}


def spectrum_page_url(station_id: str, server_url: str) -> str:
    return (
        "01_station_spectrum"
        f"?station_id={quote(station_id, safe='')}"
        f"&server_url={quote(server_url, safe='')}"
    )


def external_spectrum_app_url(
    station_id: str,
    server_url: str,
    spectrum_app_url: str = "http://localhost:8502",
) -> str:
    return (
        spectrum_app_url.rstrip("/")
        + "/"
        + f"?station_id={quote(station_id, safe='')}"
        + f"&server_url={quote(server_url, safe='')}"
    )


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


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(np.clip(value, 0.0, 1.0)) * 100:.0f}%"


def decision_scores(event: dict) -> tuple[float, float | None]:
    harmonic = _score_value(event, "harmonic_evidence_pct")
    ml = _score_value(event, "ml_drone_pct")
    if ml is None:
        ml = _score_value(event, "hf_p_drone")
    return (harmonic or 0.0), ml


def decision_display_state(event: dict) -> dict[str, str]:
    harmonic, ml = decision_scores(event)
    ml_value = ml if ml is not None else 0.0
    if harmonic >= 0.60 and ml_value >= 0.60:
        return {"label": "ML + harmonic agree", "harmonic_color": "#dc2626", "ml_color": "#ea580c"}
    if harmonic >= 0.60 and ml_value <= 0.30:
        return {"label": "non-drone harmonic", "harmonic_color": "#ca8a04", "ml_color": "#6b7280"}
    if ml_value >= 0.60 and harmonic <= 0.30:
        return {"label": "ML-only suspect", "harmonic_color": "#a3a3a3", "ml_color": "#2563eb"}
    return {"label": "background", "harmonic_color": "#16a34a", "ml_color": "#22c55e"}


def render_decision_bars(st_module, event: dict) -> None:
    harmonic, ml = decision_scores(event)
    display = decision_display_state(event)
    ml_width = 0.0 if ml is None else ml
    reason = event.get("decision_reason") or (event.get("metadata") or {}).get("decision_reason")
    reason_html = f"<div class='sky-score-reason'>{reason}</div>" if reason else ""
    st_module.markdown(
        f"""
<div class="sky-score-box">
  <div class="sky-score-label">{display['label']}</div>
  <div class="sky-score-row"><span>Harmonic evidence</span><b>{format_pct(harmonic)}</b></div>
  <div class="sky-score-track"><div style="width:{harmonic * 100:.0f}%;background:{display['harmonic_color']};"></div></div>
  <div class="sky-score-row"><span>ML drone probability</span><b>{format_pct(ml)}</b></div>
  <div class="sky-score-track"><div style="width:{ml_width * 100:.0f}%;background:{display['ml_color']};"></div></div>
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
