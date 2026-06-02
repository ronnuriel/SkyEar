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
    hf_negative: Optional[bool] = None
    hf_positive: Optional[bool] = None
    decision_reason: Optional[str] = None
    operator_label: Optional[str] = None
    estimated_azimuth_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    direction_confidence: Optional[float] = None
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
