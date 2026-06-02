# SkyEar Operator Guide

SkyEar is a passive acoustic warning system. It helps an operator inspect rotor-like acoustic evidence; it does not perform any active countermeasure.

## Spectrum

The spectrum shows magnitude by frequency for the latest station audio window. A drone-like rotor source often produces a fundamental frequency, `f0`, plus repeated harmonic peaks at `2f0`, `3f0`, `4f0`, and so on.

Dashed vertical lines mark the detector's current harmonic stack estimate. Strong repeated peaks near those markers are more useful than one isolated peak.

## Spectrogram

The spectrogram shows frequency over time. Rotor signatures often appear as persistent horizontal or slightly diagonal bands. Short chaotic bursts, speech, and music can create harmonics too, but they often shift quickly or have broader structure.

## f0 And Harmonics

`best_f0_hz` is the detector's best estimate for the fundamental rotor-like frequency. Stable f0 over several windows is important for escalating beyond `SUSPECT`.

## Channel Agreement

`channel_agreement_count / channel_count` shows how many channels detected harmonic evidence. For unsynchronized multi-mic setups, this is voting only, not reliable direction finding.

Direction is only reliable for synchronized microphone arrays with known geometry.

## HF Support

Optional Hugging Face model output is advisory. It can support confidence when harmonic evidence exists, but it cannot trigger ALERT by itself.

## Fusion Level

- LEVEL 0: no current acoustic evidence.
- LEVEL 1: one station reports suspect evidence.
- LEVEL 2: one alert station or multiple suspect stations.
- LEVEL 3: multiple independent alert-level stations. Treat this as a public-warning candidate only after validation.

## Demo Phases

- `background`: all stations should remain background.
- `motorcycle_false_positive_test`: motorcycle-like audio should not produce ALERT.
- `two_station_drone`: two stations should move toward suspect/drone_like/alert and fusion should rise.
- `single_station_drone`: one station remains affected; fusion should be lower than the two-station phase.
- `all_clear`: stations should clear back toward background after clean windows.
