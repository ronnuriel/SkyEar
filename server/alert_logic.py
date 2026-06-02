from __future__ import annotations
import math
import time
from shared.event_schema import AcousticEvent, FusedAlert


BACKGROUND_STATUSES = {"background", "calibrating"}


def _event_status(event: AcousticEvent) -> str:
    return str(event.status.value if hasattr(event.status, "value") else event.status)


def _thresholds(event: AcousticEvent) -> tuple[float, float]:
    metadata = event.metadata or {}
    suspect = float(metadata.get("suspect_threshold") or 16.0)
    alert = float(metadata.get("alert_threshold") or 22.0)
    if alert <= suspect:
        alert = suspect + 1.0
    return suspect, alert


def _harmonic_norm(event: AcousticEvent, suspect_threshold: float, alert_threshold: float) -> float:
    metadata = event.metadata or {}
    value_from_event = event.harmonic_evidence_pct_smoothed
    if value_from_event is None:
        value_from_event = event.harmonic_evidence_pct
    if value_from_event is None:
        value_from_event = metadata.get("harmonic_evidence_pct_smoothed")
    if value_from_event is None:
        value_from_event = metadata.get("harmonic_evidence_pct")
    if value_from_event is not None:
        return float(max(0.0, min(1.0, float(value_from_event))))
    value = (float(event.harmonic_score or 0.0) - suspect_threshold) / (alert_threshold - suspect_threshold)
    return float(max(0.0, min(1.0, value)))


def _ml_pct(event: AcousticEvent) -> float:
    metadata = event.metadata or {}
    value = event.ml_drone_pct_smoothed
    if value is None:
        value = event.ml_drone_pct
    if value is None:
        value = metadata.get("ml_drone_pct_smoothed")
    if value is None:
        value = metadata.get("ml_drone_pct")
    if value is None:
        value = event.hf_p_drone
    return float(max(0.0, min(1.0, float(value or 0.0))))


def _combined_pct(event: AcousticEvent) -> float:
    metadata = event.metadata or {}
    value = event.combined_drone_evidence_pct
    if value is None:
        value = metadata.get("combined_drone_evidence_pct")
    if value is not None:
        return float(max(0.0, min(1.0, float(value))))
    ml = _ml_pct(event)
    harmonic = _harmonic_norm(event, *_thresholds(event))
    return float(max(0.0, min(1.0, (2.0 * ml * harmonic) / (ml + harmonic + 1e-6))))


def _int_metric(event: AcousticEvent, key: str) -> int:
    metadata = event.metadata or {}
    value = getattr(event, key, None)
    if value is None:
        value = metadata.get(key)
    if value is None:
        return 0
    return int(value)


def _is_near_threshold_background(event: AcousticEvent, suspect_threshold: float) -> bool:
    hf = _ml_pct(event)
    harmonic = float(event.harmonic_score or 0.0)
    harmonic_pct = _harmonic_norm(event, suspect_threshold, suspect_threshold + 1.0)
    return hf >= 0.90 and (harmonic >= suspect_threshold * 0.85 or harmonic_pct > 0.0)


def _hf_negative(event: AcousticEvent) -> bool:
    metadata = event.metadata or {}
    if event.hf_negative is not None:
        return bool(event.hf_negative)
    if "hf_negative" in metadata:
        return bool(metadata.get("hf_negative"))
    return event.hf_p_drone is not None and float(event.hf_p_drone) < 0.20


def _strong_multichannel_evidence(event: AcousticEvent) -> bool:
    metadata = event.metadata or {}
    return int(event.channel_agreement_count or 0) >= 2 and bool(metadata.get("f0_stable"))


def station_evidence_score(event: AcousticEvent) -> float:
    status = _event_status(event)
    suspect_threshold, alert_threshold = _thresholds(event)
    candidate_run = _int_metric(event, "candidate_run")
    strong_run = _int_metric(event, "strong_run")
    if status in BACKGROUND_STATUSES and candidate_run <= 0 and not _is_near_threshold_background(event, suspect_threshold):
        return 0.0

    harmonic_norm = _harmonic_norm(event, suspect_threshold, alert_threshold)
    hf = _ml_pct(event)
    combined = _combined_pct(event)
    local_conf = float(event.confidence or 0.0)
    score = 0.30 * local_conf + 0.20 * hf + 0.20 * harmonic_norm + 0.30 * combined
    if combined >= 0.60:
        score = max(score, 0.65)
    elif hf >= 0.90 and harmonic_norm >= 0.15:
        score = max(score, 0.55)
    elif hf >= 0.90 and harmonic_norm <= 0.01:
        score = min(score, 0.59)
    if candidate_run >= 3 or strong_run >= 3:
        score = max(score, 0.75)
    elif candidate_run >= 2:
        score = max(score, 0.60)
    elif candidate_run == 1 and hf >= 0.90:
        score = min(score, 0.55)

    if status == "suspect":
        score += 0.15
    elif status == "drone_like":
        score += 0.35
    elif status == "alert":
        score += 0.55

    metadata = event.metadata or {}
    if bool(metadata.get("f0_stable")):
        score += 0.10
    if int(event.channel_agreement_count or 0) >= 2:
        score += 0.10
    if _hf_negative(event) and not _strong_multichannel_evidence(event):
        score *= 0.40
    if candidate_run == 1 and hf >= 0.90 and status != "alert":
        score = min(score, 0.59)

    return float(max(0.0, min(1.0, score)))


def _latest_by_station(events: list[AcousticEvent]) -> list[AcousticEvent]:
    latest: dict[str, AcousticEvent] = {}
    for event in events:
        existing = latest.get(event.station_id)
        if existing is None or event.timestamp_unix >= existing.timestamp_unix:
            latest[event.station_id] = event
    return list(latest.values())


def _same_f0_detected(events: list[AcousticEvent]) -> bool:
    f0s = [float(event.best_f0_hz) for event in events if event.best_f0_hz]
    for idx, left in enumerate(f0s):
        for right in f0s[idx + 1 :]:
            if abs(left - right) <= 120.0:
                return True
            if abs(left * 2.0 - right) <= 120.0:
                return True
            if abs(left - right * 2.0) <= 120.0:
                return True
    return False


def _metadata_value(event: AcousticEvent, key: str):
    return (event.metadata or {}).get(key)


def _scenario_key(event: AcousticEvent) -> tuple[str, str] | None:
    scenario_id = _metadata_value(event, "scenario_id")
    source_id = _metadata_value(event, "simulated_source_id")
    if scenario_id and source_id:
        return str(scenario_id), str(source_id)
    return None


def _shared_scenario_source(events: list[AcousticEvent]) -> bool:
    keys = [_scenario_key(event) for event in events]
    keys = [key for key in keys if key is not None]
    return len(keys) >= 2 and len(set(keys)) == 1


def _coverage_radius_m(event: AcousticEvent) -> float | None:
    value = _metadata_value(event, "coverage_radius_m")
    if value is None:
        return None
    return max(0.0, float(value))


def _station_distance_m(left: AcousticEvent, right: AcousticEvent) -> float | None:
    if not left.station_location or not right.station_location:
        return None
    lat1 = math.radians(float(left.station_location.latitude))
    lon1 = math.radians(float(left.station_location.longitude))
    lat2 = math.radians(float(right.station_location.latitude))
    lon2 = math.radians(float(right.station_location.longitude))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 6371000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _has_spatial_overlap(events: list[AcousticEvent]) -> bool | None:
    if len(events) < 2:
        return None
    saw_geometry = False
    for idx, left in enumerate(events):
        for right in events[idx + 1 :]:
            left_radius = _coverage_radius_m(left)
            right_radius = _coverage_radius_m(right)
            distance = _station_distance_m(left, right)
            if left_radius is None or right_radius is None or distance is None:
                continue
            saw_geometry = True
            if distance <= left_radius + right_radius:
                return True
    return False if saw_geometry else None


def alert_level_from_recent_events(events: list[AcousticEvent], window_sec: float = 8.0) -> FusedAlert:
    now = time.time()
    recent = [
        e
        for e in events
        if now - float(e.server_received_unix or (e.metadata or {}).get("server_received_unix") or e.timestamp_unix)
        <= window_sec
    ]
    latest_events = _latest_by_station(recent)
    scored = [(event, station_evidence_score(event)) for event in latest_events]
    active = [(event, score) for event, score in scored if score > 0.0]

    active_events = [event for event, _ in active]
    active_station_count = len(active_events)
    total_score = sum(score for _, score in active)
    same_f0_raw = _same_f0_detected(active_events)
    shared_scenario_source = _shared_scenario_source(active_events)
    spatial_overlap = _has_spatial_overlap(active_events)
    spatially_separate = spatial_overlap is False and not shared_scenario_source
    same_source_f0 = same_f0_raw and not spatially_separate
    if same_source_f0 and active_station_count >= 2:
        total_score += 0.15

    confirming_active = [
        (event, score)
        for event, score in active
        if not _hf_negative(event) or _strong_multichannel_evidence(event) or same_source_f0
    ]
    confirming_events = [event for event, _ in confirming_active]

    local_alert_count = sum(1 for event in confirming_events if _event_status(event) == "alert")
    drone_like_or_alert_count = sum(1 for event in confirming_events if _event_status(event) in {"drone_like", "alert"})
    suspect_count = sum(1 for event in confirming_events if _event_status(event) == "suspect")
    weak_station_count = sum(1 for _, score in confirming_active if score >= 0.45)
    combined_mid_count = sum(1 for event in confirming_events if _combined_pct(event) >= 0.45)
    combined_strong_count = sum(1 for event in confirming_events if _combined_pct(event) >= 0.60)
    combined_high_count = sum(1 for event in confirming_events if _combined_pct(event) >= 0.75)
    local_candidate_count = sum(1 for event in confirming_events if _int_metric(event, "candidate_run") >= 2)
    strong_candidate_count = sum(
        1
        for event in confirming_events
        if _int_metric(event, "candidate_run") >= 3 or _int_metric(event, "strong_run") >= 3
    )
    single_candidate_count = sum(1 for event in confirming_events if _int_metric(event, "candidate_run") == 1)
    ml_partial_count = sum(
        1
        for event in confirming_events
        if _ml_pct(event) >= 0.90 and _harmonic_norm(event, *_thresholds(event)) >= 0.15
    )
    strong_harmonic_count = sum(1 for event in confirming_events if _harmonic_norm(event, *_thresholds(event)) >= 0.45)
    hf_negative_count = sum(1 for event in active_events if _hf_negative(event))

    if (
        (same_source_f0 and combined_high_count >= 2)
        or (drone_like_or_alert_count >= 2 and not spatially_separate)
        or (suspect_count >= 3 and same_source_f0 and combined_mid_count >= 3)
    ):
        level = 3
    elif (
        total_score >= 1.2
        or weak_station_count >= 2
        or combined_mid_count >= 2
        or strong_candidate_count >= 2
        or ml_partial_count >= 2
        or local_alert_count >= 1
    ):
        level = 2
    elif total_score >= 0.6 or combined_strong_count >= 1 or local_candidate_count >= 1 or strong_candidate_count >= 1:
        level = 1
    else:
        level = 0

    max_hf = max([_ml_pct(event) for event in active_events], default=0.0)
    mean_harmonic = (
        sum(float(event.harmonic_score or 0.0) for event in active_events) / active_station_count
        if active_station_count
        else 0.0
    )
    if active_station_count == 0:
        interpretation = "background"
    elif active_station_count == 1:
        interpretation = "single-station candidate"
    elif spatially_separate:
        interpretation = "multiple local candidates"
    else:
        interpretation = "network confirmation candidate"

    reason_prefix = interpretation
    if spatially_separate:
        reason_prefix += "; multiple local acoustic candidates; stations are not spatially overlapping"

    reason = (
        f"{reason_prefix}; "
        f"active_stations={active_station_count}, max_hf_p_drone={max_hf:.2f}, "
        f"mean_harmonic={mean_harmonic:.1f}, same_f0={'yes' if same_f0_raw else 'no'}, "
        f"same_source_f0={'yes' if same_source_f0 else 'no'}, "
        f"spatial_overlap={'unknown' if spatial_overlap is None else ('yes' if spatial_overlap else 'no')}, "
        f"shared_simulated_source={'yes' if shared_scenario_source else 'no'}, "
        f"local_candidates={local_candidate_count}, strong_candidates={strong_candidate_count}, "
        f"single_window_candidates={single_candidate_count}, "
        f"combined_high={combined_high_count}, local_alerts={local_alert_count}, "
        f"hf_negative_count={hf_negative_count}, total_score={total_score:.2f}"
    )

    confidence = float(max(0.0, min(1.0, total_score / 2.0)))

    return FusedAlert(
        timestamp_unix=now,
        level=level,
        status=f"LEVEL_{level}",
        confidence=confidence,
        reason=reason,
        interpretation=interpretation,
        events_used=active_events[-10:],
    )
