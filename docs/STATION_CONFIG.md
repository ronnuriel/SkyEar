# Station Config

Default config resolution:

1. `--config PATH`
2. `SKYEAR_CONFIG`
3. `configs/config_station.yaml`

Example:

```bash
skyear --config configs/config_station.yaml station
SKYEAR_CONFIG=configs/config_station.yaml skyear monitor
```

Core sections:

- `station`: station identity and location.
- `audio`: device, sample rate, channel count, capture block size, high-pass filter, mono mix mode.
- `detector`: harmonic detector thresholds and f0 range.
- `detection`: candidate/alert persistence policy.
- `hf`: HF model settings and cadence.
- `harmonic`: ridge tracking and lock-on settings.
- `mic_array`: microphone profile and sync mode.
- `two_mic_direction`: Volt 2 style left/right/center hints.
- `direction`/`beamforming`: synchronized array bearing settings.
- `recording`: local recording controls.
- `server`: event posting target.
- `local_monitor`: local JSON snapshot paths.

List audio devices:

```bash
skyear station --list-devices
```

Run capture diagnostics:

```bash
skyear check audio --diagnostic-sec 20
```

For USB interfaces, prefer:

```yaml
audio:
  latency: high
  capture_block_sec: 0.25
```
