from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class EventStatus(str, Enum):
    CALIBRATING = "calibrating"
    BACKGROUND = "background"
    SUSPECT = "suspect"
    DRONE_LIKE = "drone_like"
    ALERT = "alert"

class GeoPoint(BaseModel):
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None

class ChannelEvidence(BaseModel):
    channel_index: int
    rms: Optional[float] = None
    harmonic_score: float
    best_f0_hz: Optional[int] = None
    passed: bool

class AcousticEvent(BaseModel):
    station_id: str
    station_name: Optional[str] = None
    timestamp_unix: float
    server_received_unix: Optional[float] = None
    station_location: Optional[GeoPoint] = None
    station_latitude: Optional[float] = None
    station_longitude: Optional[float] = None
    station_altitude_m: Optional[float] = None
    station_heading_offset_deg: Optional[float] = None
    station_location_label: Optional[str] = None
    status: EventStatus
    confidence: float = Field(ge=0.0, le=1.0)
    harmonic_score: float
    harmonic_score_smoothed: Optional[float] = None
    harmonic_evidence_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    harmonic_evidence_pct_smoothed: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    best_f0_hz: Optional[int] = None
    raw_best_f0_hz: Optional[int] = None
    canonical_best_f0_hz: Optional[int] = None
    f0_family_stable: Optional[bool] = None
    ml_drone_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ml_drone_pct_smoothed: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    combined_drone_evidence_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hf_p_drone: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cnn_p_drone: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hf_error: Optional[bool] = None
    hf_negative: Optional[bool] = None
    hf_positive: Optional[bool] = None
    harmonic_activity_duration_sec: Optional[float] = Field(default=None, ge=0.0)
    decision_reason: Optional[str] = None
    operator_label: Optional[str] = None
    candidate_run: Optional[int] = Field(default=None, ge=0)
    hf_candidate_run: Optional[int] = Field(default=None, ge=0)
    acoustic_candidate_run: Optional[int] = Field(default=None, ge=0)
    fused_candidate_run: Optional[int] = Field(default=None, ge=0)
    ml_positive_run: Optional[int] = Field(default=None, ge=0)
    strong_run: Optional[int] = Field(default=None, ge=0)
    hf_age_sec: Optional[float] = Field(default=None, ge=0.0)
    harmonic_age_sec: Optional[float] = Field(default=None, ge=0.0)
    max_hf_age_sec: Optional[float] = Field(default=None, ge=0.0)
    max_acoustic_age_sec: Optional[float] = Field(default=None, ge=0.0)
    estimated_detection_delay_sec: Optional[float] = Field(default=None, ge=0.0)
    decision_stage: Optional[str] = None
    blocked_by: Optional[str] = None
    hf_watch_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hf_candidate_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hf_strong_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    hf_candidate_pass: Optional[bool] = None
    hf_strong_pass: Optional[bool] = None
    harmonic_pass: Optional[bool] = None
    single_channel_mode: Optional[bool] = None
    candidate_block_reason: Optional[str] = None
    alert_block_reason: Optional[str] = None
    alert_blocked_reason: Optional[str] = None
    why_candidate_run_reset: Optional[str] = None
    harmonic_track_active: Optional[bool] = None
    tracked_f0_hz: Optional[int] = None
    tracked_ridges: list[Dict[str, Any]] = Field(default_factory=list)
    harmonic_track_age_sec: Optional[float] = Field(default=None, ge=0.0)
    f0_raw_hz: Optional[int] = None
    f0_track_hz: Optional[int] = None
    f0_jump_reason: Optional[str] = None
    stable_harmonic_ridge_count: Optional[int] = Field(default=None, ge=0)
    longest_ridge_duration_sec: Optional[float] = Field(default=None, ge=0.0)
    estimated_azimuth_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    raw_bearing_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    tracked_bearing_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    bearing_velocity_deg_per_sec: Optional[float] = None
    bearing_track_age_sec: Optional[float] = Field(default=None, ge=0.0)
    bearing_track_stable: Optional[bool] = None
    bearing_track_status: Optional[str] = None
    bearing_flip_suppressed: Optional[bool] = None
    bearing_used_for_geo: Optional[bool] = None
    direction_confidence: Optional[float] = None
    beamforming_method: Optional[str] = None
    beam_score: Optional[float] = None
    beam_snr_gain_db: Optional[float] = None
    beam_confidence_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    beam_peak_to_median: Optional[float] = None
    beam_peak_to_second_peak: Optional[float] = None
    second_peak_bearing_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    second_peak_ratio: Optional[float] = None
    peak_ratio: Optional[float] = None
    bearing_ambiguity_deg: Optional[float] = None
    bearing_reliable: Optional[bool] = None
    bearing_reject_reason: Optional[str] = None
    bearing_quality: Optional[str] = None
    bearing_stable: Optional[bool] = None
    bearing_uncertainty_deg: Optional[float] = None
    two_mic_side: Optional[str] = None
    two_mic_delay_us: Optional[float] = None
    two_mic_angle_from_center_deg: Optional[float] = None
    two_mic_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    two_mic_peak_ratio: Optional[float] = None
    two_mic_peak_to_second_peak: Optional[float] = None
    two_mic_lag_ambiguity_us: Optional[float] = None
    two_mic_reason: Optional[str] = None
    two_mic_look_label: Optional[str] = None
    two_mic_look_hint: Optional[str] = None
    two_mic_sector_width_deg: Optional[float] = None
    two_mic_front_back_ambiguous: Optional[bool] = None
    two_mic_direction_stable: Optional[bool] = None
    possible_front_azimuth_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    possible_back_azimuth_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    overflow_recent: Optional[bool] = None
    overflow_timestamps: list[float] = Field(default_factory=list)
    recording_continuity_ok: Optional[bool] = None
    track_id: Optional[str] = None
    rms: Optional[float] = None
    peak: Optional[float] = None
    duration_sec: Optional[float] = None
    calibrated: bool = False
    strongest_channel: Optional[int] = None
    channel_agreement_count: Optional[int] = None
    channel_count: Optional[int] = None
    channel_evidence: list[ChannelEvidence] = Field(default_factory=list)
    detector_version: Optional[str] = None
    station_mode: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StationHeartbeat(BaseModel):
    station_id: str
    station_name: Optional[str] = None
    timestamp_unix: float
    server_received_unix: Optional[float] = None
    status: str = "online"
    station_location: Optional[GeoPoint] = None
    audio_device: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    calibrated: Optional[bool] = None
    detector_version: Optional[str] = None
    station_mode: Optional[str] = None
    last_event_status: Optional[str] = None
    last_harmonic_score: Optional[float] = None
    last_hf_p_drone: Optional[float] = None
    last_error: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class TrackSummary(BaseModel):
    track_id: str
    station_ids: list[str] = Field(default_factory=list)
    events: list[AcousticEvent] = Field(default_factory=list)
    level: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    estimated_source: Optional[Dict[str, Any]] = None
    interpretation: str
    same_f0: bool = False

class FusedAlert(BaseModel):
    timestamp_unix: float
    level: int = Field(ge=0, le=3)
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    interpretation: Optional[str] = None
    global_level: Optional[int] = None
    tracks: list[TrackSummary] = Field(default_factory=list)
    events_used: list[AcousticEvent] = Field(default_factory=list)
    estimated_location: Optional[GeoPoint] = None
