from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class EventStatus(str, Enum):
    BACKGROUND = "background"
    SUSPECT = "suspect"
    DRONE_LIKE = "drone_like"
    ALERT = "alert"

class GeoPoint(BaseModel):
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None

class AcousticEvent(BaseModel):
    station_id: str
    station_name: Optional[str] = None
    timestamp_unix: float
    station_location: Optional[GeoPoint] = None
    status: EventStatus
    confidence: float = Field(ge=0.0, le=1.0)
    harmonic_score: float
    best_f0_hz: Optional[int] = None
    hf_p_drone: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cnn_p_drone: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    estimated_azimuth_deg: Optional[float] = Field(default=None, ge=0.0, le=360.0)
    direction_confidence: Optional[float] = None
    rms: Optional[float] = None
    peak: Optional[float] = None
    duration_sec: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class FusedAlert(BaseModel):
    timestamp_unix: float
    level: int = Field(ge=0, le=3)
    status: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    events_used: list[AcousticEvent] = Field(default_factory=list)
    estimated_location: Optional[GeoPoint] = None
