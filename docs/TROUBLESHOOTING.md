# Troubleshooting

## Audio Overflow

Run:

```bash
skyear check audio --diagnostic-sec 20
```

If overflow increases, use:

```yaml
audio:
  latency: high
  capture_block_sec: 0.25
```

If the wrong device is selected, run:

```bash
skyear setup audio
skyear check audio --dry-run
```

Direction hints are suppressed when `overflow_recent=true`.

## Direction Jumps Left/Right

Check `bearing_quality`, `bearing_reject_reason`, `bearing_flip_suppressed`, `bearing_track_status`, `raw_bearing_deg`, and `tracked_bearing_deg`.

Common causes:

- Channel order does not match mic geometry.
- Array calibration is placeholder/invalid or has silent channels.
- `heading_offset_deg` is wrong.
- Multipath creates a second lobe.
- Unsynchronized two-mic devices cannot provide precise bearing.

For Volt 2, use the local monitor's Direction Hint panel as a search sector, not a map pin.

## Harmonic Bands Visible But Background

Check `mono_mix_mode`, `selected_mono_channel`, `per_channel_harmonic_score`, `harmonic_soft_present`, `harmonic_soft_run`, `calibration_p95`, `threshold_source`, and `adaptive_threshold_reason`.

For unsynchronized dual-mic devices, prefer:

```yaml
audio:
  mono_mix_mode: strongest_harmonic
```

## HF Unavailable

Run:

```bash
skyear check hf
```

HF errors do not upload raw audio. Harmonic-only mode remains available, but alerts stay conservative.
