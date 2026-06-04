from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from station.array_profiles import array_profile
from station.direction import SPEED_OF_SOUND


def angle_delta_deg(start_deg: float, end_deg: float) -> float:
    return (float(end_deg) - float(start_deg) + 180.0) % 360.0 - 180.0


def moving_bearings(
    *,
    start_deg: float,
    end_deg: float,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.asarray([], dtype=np.float64)
    progress = np.linspace(0.0, 1.0, int(sample_count), endpoint=False, dtype=np.float64)
    return (float(start_deg) + angle_delta_deg(start_deg, end_deg) * progress) % 360.0


def harmonic_source(
    *,
    sample_rate: int,
    duration_sec: float,
    source_type: str,
    f0_hz: float,
) -> np.ndarray:
    sample_count = int(round(float(sample_rate) * float(duration_sec)))
    t = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    source_type = str(source_type)
    if source_type == "tone":
        source = np.sin(2.0 * np.pi * float(f0_hz) * t)
    elif source_type == "harmonic_drone":
        source = np.zeros_like(t)
        for harmonic, gain in ((1, 1.0), (2, 0.55), (3, 0.30), (4, 0.18)):
            source += gain * np.sin(2.0 * np.pi * float(f0_hz) * harmonic * t)
        source *= 0.65 + 0.08 * np.sin(2.0 * np.pi * 5.0 * t)
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    peak = float(np.max(np.abs(source))) if source.size else 1.0
    return (0.35 * source / max(peak, 1e-9)).astype(np.float32)


def noise_source(*, sample_rate: int, duration_sec: float, seed: int = 123) -> np.ndarray:
    sample_count = int(round(float(sample_rate) * float(duration_sec)))
    rng = np.random.default_rng(int(seed))
    source = rng.normal(0.0, 1.0, sample_count)
    peak = float(np.max(np.abs(source))) if source.size else 1.0
    return (0.35 * source / max(peak, 1e-9)).astype(np.float32)


def _pink_noise(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=shape)
    freqs = np.fft.rfftfreq(shape[0])
    weights = np.ones_like(freqs)
    weights[1:] = 1.0 / np.sqrt(freqs[1:])
    spectrum = np.fft.rfft(white, axis=0) * weights[:, None]
    pink = np.fft.irfft(spectrum, n=shape[0], axis=0)
    std = float(np.std(pink))
    return pink / max(std, 1e-9)


def add_noise(
    audio: np.ndarray,
    *,
    snr_db: float,
    noise_type: str,
    seed: int = 123,
) -> np.ndarray:
    if noise_type == "none":
        return audio.astype(np.float32)
    rng = np.random.default_rng(int(seed))
    if noise_type == "pink":
        noise = _pink_noise(audio.shape, rng)
    elif noise_type == "white":
        noise = rng.normal(0.0, 1.0, size=audio.shape)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    signal_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    return (audio + noise * (target_noise_rms / max(noise_rms, 1e-9))).astype(np.float32)


def parse_float_list(value: str | None) -> list[float]:
    if value is None or str(value).strip() == "":
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _render_plane_wave(
    *,
    source: np.ndarray,
    mic_positions_m: np.ndarray,
    sample_rate: int,
    bearings_deg: np.ndarray,
) -> np.ndarray:
    sample_index = np.arange(source.size, dtype=np.float64)
    radians = np.deg2rad(bearings_deg)
    direction_xy = np.stack([np.cos(radians), np.sin(radians)], axis=1)
    positions = np.asarray(mic_positions_m, dtype=np.float64)
    audio = np.zeros((source.size, positions.shape[0]), dtype=np.float32)
    for channel_idx, position in enumerate(positions):
        delays_samples = (direction_xy @ position[:2]) / SPEED_OF_SOUND * float(sample_rate)
        audio[:, channel_idx] = np.interp(
            sample_index - delays_samples,
            sample_index,
            source,
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    return audio


def _delay_mono(source: np.ndarray, delay_samples: float) -> np.ndarray:
    sample_index = np.arange(source.size, dtype=np.float64)
    return np.interp(sample_index - float(delay_samples), sample_index, source, left=0.0, right=0.0).astype(np.float32)


def _apply_channel_delay_jitter(audio: np.ndarray, delays_us: np.ndarray, sample_rate: int) -> np.ndarray:
    sample_index = np.arange(audio.shape[0], dtype=np.float64)
    out = np.zeros_like(audio, dtype=np.float32)
    for channel_idx in range(audio.shape[1]):
        delay_samples = float(delays_us[channel_idx]) * 1e-6 * float(sample_rate)
        out[:, channel_idx] = np.interp(
            sample_index - delay_samples,
            sample_index,
            audio[:, channel_idx],
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    return out


def _apply_highpass(audio: np.ndarray, sample_rate: int, highpass_hz: float | None) -> np.ndarray:
    if highpass_hz is None or float(highpass_hz) <= 0.0:
        return audio.astype(np.float32)
    try:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(4, float(highpass_hz), btype="highpass", fs=int(sample_rate), output="sos")
        return sosfiltfilt(sos, audio, axis=0).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


def synthesize_array_audio(
    *,
    mic_positions_m: np.ndarray,
    sample_rate: int,
    duration_sec: float,
    bearing_start_deg: float,
    bearing_end_deg: float,
    source_type: str,
    f0_hz: float,
    snr_db: float,
    noise_type: str = "white",
    mic_gain_jitter_db: float = 0.0,
    mic_position_jitter_cm: float = 0.0,
    channel_delay_jitter_us: float = 0.0,
    drop_channels: list[int] | None = None,
    drop_random_channels: int = 0,
    permute_channels: str = "none",
    reflection_count: int = 0,
    reflection_delay_ms: list[float] | None = None,
    reflection_gain_db: list[float] | None = None,
    reflection_bearing_offset_deg: float = 40.0,
    interferer_type: str | None = None,
    interferer_bearing_deg: float = 180.0,
    interferer_f0_hz: float = 1000.0,
    interferer_snr_db: float = 0.0,
    wind_noise_level: float = 0.0,
    highpass_hz: float | None = None,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    nominal_positions = np.asarray(mic_positions_m, dtype=np.float64)
    positions = nominal_positions.copy()
    if float(mic_position_jitter_cm) > 0.0:
        jitter_m = rng.normal(0.0, float(mic_position_jitter_cm) / 100.0, size=positions[:, :2].shape)
        positions[:, :2] += jitter_m

    source = harmonic_source(
        sample_rate=int(sample_rate),
        duration_sec=float(duration_sec),
        source_type=source_type,
        f0_hz=float(f0_hz),
    )
    bearings = moving_bearings(
        start_deg=float(bearing_start_deg),
        end_deg=float(bearing_end_deg),
        sample_count=source.size,
    )
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError("mic_positions_m must be an Nx2 or Nx3 array")
    audio = _render_plane_wave(
        source=source,
        mic_positions_m=positions,
        sample_rate=int(sample_rate),
        bearings_deg=bearings,
    )
    direct_rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))

    reflection_delay_ms = reflection_delay_ms or []
    reflection_gain_db = reflection_gain_db or []
    for reflection_idx in range(max(0, int(reflection_count))):
        delay_ms = reflection_delay_ms[reflection_idx % len(reflection_delay_ms)] if reflection_delay_ms else 12.0
        gain_db = reflection_gain_db[reflection_idx % len(reflection_gain_db)] if reflection_gain_db else -10.0
        delayed = _delay_mono(source, float(delay_ms) * 1e-3 * float(sample_rate))
        reflected_bearings = (bearings + float(reflection_bearing_offset_deg) * float(reflection_idx + 1)) % 360.0
        audio += float(10.0 ** (float(gain_db) / 20.0)) * _render_plane_wave(
            source=delayed,
            mic_positions_m=positions,
            sample_rate=int(sample_rate),
            bearings_deg=reflected_bearings,
        )

    if interferer_type:
        normalized_interferer = str(interferer_type).strip().lower()
        if normalized_interferer == "tone":
            interferer = harmonic_source(
                sample_rate=int(sample_rate),
                duration_sec=float(duration_sec),
                source_type="tone",
                f0_hz=float(interferer_f0_hz),
            )
        elif normalized_interferer == "harmonic":
            interferer = harmonic_source(
                sample_rate=int(sample_rate),
                duration_sec=float(duration_sec),
                source_type="harmonic_drone",
                f0_hz=float(interferer_f0_hz),
            )
        elif normalized_interferer == "noise":
            interferer = noise_source(sample_rate=int(sample_rate), duration_sec=float(duration_sec), seed=int(seed) + 77)
        else:
            raise ValueError(f"Unsupported interferer type: {interferer_type}")
        interferer_audio = _render_plane_wave(
            source=interferer,
            mic_positions_m=positions,
            sample_rate=int(sample_rate),
            bearings_deg=np.full_like(bearings, float(interferer_bearing_deg), dtype=np.float64),
        )
        interferer_rms = float(np.sqrt(np.mean(interferer_audio.astype(np.float64) ** 2)))
        target_rms = direct_rms / (10.0 ** (float(interferer_snr_db) / 20.0))
        audio += interferer_audio * (target_rms / max(interferer_rms, 1e-9))

    gain_db = rng.normal(0.0, float(mic_gain_jitter_db), size=audio.shape[1]) if float(mic_gain_jitter_db) else np.zeros(audio.shape[1])
    audio *= (10.0 ** (gain_db / 20.0))[None, :]

    delay_us = (
        rng.normal(0.0, float(channel_delay_jitter_us), size=audio.shape[1])
        if float(channel_delay_jitter_us)
        else np.zeros(audio.shape[1])
    )
    if float(channel_delay_jitter_us):
        audio = _apply_channel_delay_jitter(audio, delay_us, int(sample_rate))

    if float(wind_noise_level) > 0.0:
        wind = _pink_noise(audio.shape, rng)
        wind_rms = float(np.sqrt(np.mean(wind**2)))
        audio += wind * (direct_rms * float(wind_noise_level) / max(wind_rms, 1e-9))

    dropped = sorted({int(ch) for ch in (drop_channels or []) if 0 <= int(ch) < audio.shape[1]})
    if int(drop_random_channels) > 0:
        candidates = [idx for idx in range(audio.shape[1]) if idx not in dropped]
        random_drop = rng.choice(candidates, size=min(len(candidates), int(drop_random_channels)), replace=False)
        dropped = sorted(set(dropped).union(int(ch) for ch in random_drop))
    for channel_idx in dropped:
        audio[:, channel_idx] = 0.0

    permutation = list(range(audio.shape[1]))
    if str(permute_channels).lower() == "random":
        permutation = [int(idx) for idx in rng.permutation(audio.shape[1])]
        audio = audio[:, permutation]
    elif str(permute_channels).lower() not in {"none", ""}:
        raise ValueError(f"Unsupported permutation mode: {permute_channels}")

    audio = add_noise(audio, snr_db=float(snr_db), noise_type=noise_type, seed=int(seed) + 5)
    audio = _apply_highpass(audio, int(sample_rate), highpass_hz)
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 0.98:
        audio = audio * (0.98 / peak)

    metadata = {
        "nominal_mic_positions_m": nominal_positions.tolist(),
        "actual_mic_positions_m": positions.tolist(),
        "mic_gain_db": [float(value) for value in gain_db],
        "channel_delay_us": [float(value) for value in delay_us],
        "dropped_channels_before_permutation": dropped,
        "channel_permutation_output_to_original": permutation,
        "reflection_count": int(reflection_count),
        "reflection_delay_ms": [float(value) for value in reflection_delay_ms],
        "reflection_gain_db": [float(value) for value in reflection_gain_db],
        "reflection_bearing_offset_deg": float(reflection_bearing_offset_deg),
        "interferer_type": interferer_type,
        "interferer_bearing_deg": float(interferer_bearing_deg),
        "interferer_f0_hz": float(interferer_f0_hz),
        "interferer_snr_db": float(interferer_snr_db),
        "wind_noise_level": float(wind_noise_level),
        "highpass_hz": None if highpass_hz is None else float(highpass_hz),
        "seed": int(seed),
    }
    return audio.astype(np.float32), bearings, metadata


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), audio, int(sample_rate))
    except Exception:
        from scipy.io import wavfile

        clipped = np.clip(audio, -1.0, 1.0)
        wavfile.write(str(path), int(sample_rate), (clipped * 32767.0).astype(np.int16))


def write_truth_csv(
    path: str | Path,
    *,
    bearings: np.ndarray,
    sample_rate: int,
    duration_sec: float,
    source_type: str,
    snr_db: float,
    step_sec: float = 1.0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp_sec", "bearing_deg", "source_type", "snr_db"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        steps = max(1, int(np.ceil(float(duration_sec) / float(step_sec)))) + 1
        for idx in range(steps):
            timestamp_sec = min(float(duration_sec), idx * float(step_sec))
            sample_idx = min(len(bearings) - 1, int(round(timestamp_sec * int(sample_rate))))
            writer.writerow(
                {
                    "timestamp_sec": f"{timestamp_sec:.6f}",
                    "bearing_deg": f"{float(bearings[sample_idx]):.6f}",
                    "source_type": source_type,
                    "snr_db": f"{float(snr_db):.3f}",
                }
            )


def truth_metadata_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix(".metadata.json")


def write_truth_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    metadata_path = truth_metadata_path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic moving-source 8-channel array audio.")
    parser.add_argument("--profile", default="field_8ch_r0_35m")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--bearing-start-deg", type=float, default=300.0)
    parser.add_argument("--bearing-end-deg", type=float, default=60.0)
    parser.add_argument("--source-type", choices=["tone", "harmonic_drone"], default="harmonic_drone")
    parser.add_argument("--f0", type=float, default=1200.0)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--noise", choices=["white", "pink", "none"], default="white")
    parser.add_argument("--mic-gain-jitter-db", type=float, default=0.0)
    parser.add_argument("--mic-position-jitter-cm", type=float, default=0.0)
    parser.add_argument("--channel-delay-jitter-us", type=float, default=0.0)
    parser.add_argument("--drop-channel", type=int, action="append", default=[])
    parser.add_argument("--drop-random-channels", type=int, default=0)
    parser.add_argument("--permute-channels", choices=["none", "random"], default="none")
    parser.add_argument("--reflection-count", type=int, default=0)
    parser.add_argument("--reflection-delay-ms", default="")
    parser.add_argument("--reflection-gain-db", default="")
    parser.add_argument("--reflection-bearing-offset-deg", type=float, default=40.0)
    parser.add_argument("--interferer-type", choices=["tone", "harmonic", "noise"], default=None)
    parser.add_argument("--interferer-bearing-deg", type=float, default=180.0)
    parser.add_argument("--interferer-f0", type=float, default=1000.0)
    parser.add_argument("--interferer-snr-db", type=float, default=0.0)
    parser.add_argument("--wind-noise-level", type=float, default=0.0)
    parser.add_argument("--highpass-hz", type=float)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="reports/sim/moving_source_8ch.wav")
    parser.add_argument("--truth", default="reports/sim/moving_source_truth.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = array_profile(args.profile)
    if profile is None or not profile.get("mic_positions_m"):
        raise SystemExit(f"Unknown or non-array profile: {args.profile}")
    positions = np.asarray(profile["mic_positions_m"], dtype=np.float64)
    audio, bearings, metadata = synthesize_array_audio(
        mic_positions_m=positions,
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
        bearing_start_deg=float(args.bearing_start_deg),
        bearing_end_deg=float(args.bearing_end_deg),
        source_type=str(args.source_type),
        f0_hz=float(args.f0),
        snr_db=float(args.snr_db),
        noise_type=str(args.noise),
        mic_gain_jitter_db=float(args.mic_gain_jitter_db),
        mic_position_jitter_cm=float(args.mic_position_jitter_cm),
        channel_delay_jitter_us=float(args.channel_delay_jitter_us),
        drop_channels=list(args.drop_channel or []),
        drop_random_channels=int(args.drop_random_channels),
        permute_channels=str(args.permute_channels),
        reflection_count=int(args.reflection_count),
        reflection_delay_ms=parse_float_list(args.reflection_delay_ms),
        reflection_gain_db=parse_float_list(args.reflection_gain_db),
        reflection_bearing_offset_deg=float(args.reflection_bearing_offset_deg),
        interferer_type=args.interferer_type,
        interferer_bearing_deg=float(args.interferer_bearing_deg),
        interferer_f0_hz=float(args.interferer_f0),
        interferer_snr_db=float(args.interferer_snr_db),
        wind_noise_level=float(args.wind_noise_level),
        highpass_hz=args.highpass_hz,
        seed=int(args.seed),
    )
    write_wav(args.output, audio, int(args.sample_rate))
    write_truth_csv(
        args.truth,
        bearings=bearings,
        sample_rate=int(args.sample_rate),
        duration_sec=float(args.duration_sec),
        source_type=str(args.source_type),
        snr_db=float(args.snr_db),
    )
    metadata_path = write_truth_metadata(
        args.truth,
        {
            **metadata,
            "profile": args.profile,
            "sample_rate": int(args.sample_rate),
            "duration_sec": float(args.duration_sec),
            "bearing_start_deg": float(args.bearing_start_deg),
            "bearing_end_deg": float(args.bearing_end_deg),
            "source_type": str(args.source_type),
            "f0_hz": float(args.f0),
            "snr_db": float(args.snr_db),
            "noise": str(args.noise),
        },
    )
    print(
        f"wrote {args.output} shape={audio.shape} sample_rate={args.sample_rate} "
        f"truth={args.truth} metadata={metadata_path} bearing={args.bearing_start_deg}->{args.bearing_end_deg}"
    )


if __name__ == "__main__":
    main()
