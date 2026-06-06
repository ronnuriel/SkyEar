# Two-Mic Direction

Volt 2 and similar two-channel setups can provide practical operator hints:

- `LOOK LEFT`
- `LOOK RIGHT`
- `LOOK CENTER`
- `UNKNOWN / UNSTABLE`

Two microphones do not provide true 360 degree bearing. They cannot distinguish front-left from back-left, so SkyEar marks two-mic hints as front/back ambiguous.

Run a live check:

```bash
skyear setup volt2
skyear check two-mic --tracked
```

Expected checks:

- Clap near left mic -> `LOOK LEFT`
- Clap near right mic -> `LOOK RIGHT`
- Clap centered in front -> `LOOK CENTER`

Important config:

```yaml
two_mic_direction:
  enabled: true
  spacing_m: 2.00
  left_channel: 0
  right_channel: 1
  center_deadzone_deg: 12
  look_sector_width_deg: 60
  smoothing_windows: 7
  min_stable_windows: 4
```

Map sectors are not drawn from two-mic hints unless `front_heading_deg` is configured and the ambiguity is explicitly represented as possible front/back azimuths.
