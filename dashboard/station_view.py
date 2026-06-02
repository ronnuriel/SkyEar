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


def status_label(status: str) -> tuple[str, str]:
    return STATUS_STYLE.get(status, (status.upper(), "info"))


def plot_spectrum_figure(metadata: dict, small: bool = False):
    freqs = metadata.get("spectrum_freqs_hz") or []
    db = metadata.get("spectrum_db") or []
    harmonic_lines = metadata.get("harmonic_lines") or []
    if not freqs or not db:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 1.8) if small else (9.0, 3.2))
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
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
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
