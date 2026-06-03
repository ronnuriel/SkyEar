# SkyEar Field Test Protocol

SkyEar field sessions are engineering dry-runs only, not operational deployments. The project is passive warning and visual confirmation only. Do not use SkyEar for targeting, lasers, weapon integration, interdiction, or public warning decisions.

The goal is to collect clean acoustic, station-health, and ground-truth notes for later evaluation and model training.

## Equipment Checklist

- Station computer with SkyEar installed and the station virtual environment tested.
- Microphone or synchronized microphone array, with known channel order.
- Windscreen, tripod/mast, cables, strain relief, and power bank or mains power.
- Central server laptop, or a known same-LAN/Tailscale endpoint.
- Local monitor browser access on each station.
- Stopwatch or synchronized clock source.
- Drone, batteries, pilot/controller, spotter, and site permission.
- Safety perimeter markers and a written abort plan.

## Station Setup Checklist

- Confirm station ID and config file before recording.
- Confirm sample rate, channel count, mic profile, and sync mode.
- Confirm station location and, for arrays, mic positions and array orientation.
- Start the central server if the session will post live events.
- Start each station and verify the local monitor updates even if the server is down.
- Verify raw candidate recording is enabled when the session needs WAV evidence.
- Run a short no-drone dry run and save a debug capture.

## Gain, RMS, And Clipping Checklist

- Watch local monitor RMS before any drone run.
- Very low RMS can mean the wrong input device or disconnected mic.
- Peaks near full scale indicate clipping; reduce gain before collecting labels.
- Record at least one baseline environment segment with no drone.
- Note wind gusts, vehicles, voices, aircraft, generators, fans, or other tonal sources.

## Server And Local Monitor Checklist

- Server `/health` is OK.
- Station connectivity check warns only if expected.
- Local monitor state file is fresh, not stale.
- HF status is known: available, unavailable, or intentionally disabled.
- Station history JSONL is growing.
- Server dashboard shows station heartbeats if a central server is used.

## Startup Checklist

1. Create a session folder with `skyear-start-field-session`.
2. Start the server or verify the planned offline/local-monitor mode.
3. Start each station with the correct config.
4. Open each local monitor and confirm fresh state, waveform, spectrum, HF status, and server status.
5. Run `skyear-check-server` if the station will post to a central server.
6. Mark a baseline no-drone event before drone activity starts.

## Baseline No-Drone Recording

Before drone activity, record at least 2 minutes of the site background:

- quiet baseline
- wind baseline if wind is present
- known local noise sources if unavoidable

Mark these as `background`, `wind`, or the most specific label available.

## Distance Schedule

Recommended distances:

- 20 m
- 50 m
- 100 m
- 150 m
- 200 m

For each distance, mark the event before the maneuver starts and note the estimated bearing if available.

## Maneuver Cases

Collect several cases at each distance when safe:

- hover
- fly-by
- approach
- departure

Keep each case long enough to produce several 1 second windows. A 20-30 second case is a useful minimum for persistence metrics.

## Wind, Noise, And Environment Notes

For each run, write notes for:

- wind estimate and gusts
- surface type and reflections
- nearby roads, vehicles, people, birds, aircraft, fans, engines, or generators
- obstructions between station and drone
- station height and microphone orientation

## Operator Safety Notes

- Follow local law, site permissions, and pilot instructions.
- Keep a spotter watching the aircraft and the test area.
- Do not use SkyEar output for targeting, laser cueing, weapon integration, or public warning decisions.
- Treat PTZ/gimbal cues as visual observation aids only.
- Abort the run if the pilot, spotter, or site owner calls stop.

## Required Files To Save After Each Test

- `session.yaml`
- `notes.csv`
- station latest JSON snapshots from `runtime/stations/*_latest.json`
- station history JSONL files from `runtime/stations/*_history.jsonl`
- raw candidate WAV files and JSON sidecars when raw recording is enabled
- local monitor debug captures when an operator manually captures a case
- server reports or exported station reports used for evaluation
- any manual weather, distance, bearing, or safety notes not already in `notes.csv`

## Data Labeling Protocol

- Start a field session with `skyear-start-field-session`.
- Mark each ground-truth event with `skyear-mark-field-event`.
- Use labels: `drone`, `background`, `helicopter`, `wind`, `vehicle`, or `unknown`.
- Include distance, drone model, maneuver, bearing, and notes when known.
- Keep labels conservative. If unsure, use `unknown`.
- Do not relabel after seeing detector output unless you also preserve the original note.
- Keep raw candidate WAVs and metadata sidecars together with the session folder.

## Minimal Session Flow

```bash
skyear-start-field-session --location "test field north" --station-id station_001 --drone-model DJI_Neo
skyear-mark-field-event --session field_sessions/<session_id> --label background --note "2 min no-drone baseline"
skyear-mark-field-event --session field_sessions/<session_id> --label drone --distance-m 50 --drone-model DJI_Neo --note "hover 30 sec north"
skyear-save-debug-capture --seconds 30 --label unknown --note "manual capture"
skyear-eval-field-session --session field_sessions/<session_id>
```
