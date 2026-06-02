from __future__ import annotations
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
    value = (float(event.harmonic_score or 0.0) - suspect_threshold) / (alert_threshold - suspect_threshold)
    return float(max(0.0, min(1.0, value)))


def _is_near_threshold_background(event: AcousticEvent, suspect_threshold: float) -> bool:
    hf = float(event.hf_p_drone or 0.0)
    harmonic = float(event.harmonic_score or 0.0)
    return hf >= 0.90 and harmonic >= suspect_threshold * 0.85


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
    if status in BACKGROUND_STATUSES and not _is_near_threshold_background(event, suspect_threshold):
        return 0.0

    harmonic_norm = _harmonic_norm(event, suspect_threshold, alert_threshold)
    hf = float(event.hf_p_drone or 0.0)
    local_conf = float(event.confidence or 0.0)
    score = 0.45 * local_conf + 0.30 * hf + 0.25 * harmonic_norm

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


def alert_level_from_recent_events(events: list[AcousticEvent], window_sec: float = 8.0) -> FusedAlert:
    now = time.time()
    recent = [e for e in events if now - e.timestamp_unix <= window_sec]
    latest_events = _latest_by_station(recent)
    scored = [(event, station_evidence_score(event)) for event in latest_events]
    active = [(event, score) for event, score in scored if score > 0.0]

    active_events = [event for event, _ in active]
    active_station_count = len(active_events)
    total_score = sum(score for _, score in active)
    same_f0 = _same_f0_detected(active_events)
    if same_f0 and active_station_count >= 2:
        total_score += 0.15

    confirming_active = [
        (event, score)
        for event, score in active
        if not _hf_negative(event) or _strong_multichannel_evidence(event) or same_f0
    ]
    confirming_events = [event for event, _ in confirming_active]

    local_alert_count = sum(1 for event in confirming_events if _event_status(event) == "alert")
    drone_like_or_alert_count = sum(1 for event in confirming_events if _event_status(event) in {"drone_like", "alert"})
    suspect_count = sum(1 for event in confirming_events if _event_status(event) == "suspect")
    weak_station_count = sum(1 for _, score in confirming_active if score >= 0.45)
    hf_negative_count = sum(1 for event in active_events if _hf_negative(event))

    if (total_score >= 2.0 and active_station_count >= 2) or drone_like_or_alert_count >= 2 or (
        suspect_count >= 3 and same_f0
    ):
        level = 3
    elif total_score >= 1.2 or weak_station_count >= 2 or local_alert_count >= 1:
        level = 2
    elif total_score >= 0.6:
        level = 1
    else:
        level = 0

    max_hf = max([float(event.hf_p_drone or 0.0) for event in active_events], default=0.0)
    mean_harmonic = (
        sum(float(event.harmonic_score or 0.0) for event in active_events) / active_station_count
        if active_station_count
        else 0.0
    )
    reason = (
        "network acoustic confirmation candidate; "
        f"active_stations={active_station_count}, max_hf_p_drone={max_hf:.2f}, "
        f"mean_harmonic={mean_harmonic:.1f}, same_f0={'yes' if same_f0 else 'no'}, "
        f"local_alerts={local_alert_count}, hf_negative_count={hf_negative_count}, total_score={total_score:.2f}"
    )

    confidence = float(max(0.0, min(1.0, total_score / 2.0)))

    return FusedAlert(
        timestamp_unix=now,
        level=level,
        status=f"LEVEL_{level}",
        confidence=confidence,
        reason=reason,
        events_used=active_events[-10:],
    )
