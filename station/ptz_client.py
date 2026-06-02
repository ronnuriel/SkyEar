from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class PTZCommand:
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    zoom: Optional[float] = None

class PTZClient:
    """
    Safe visual confirmation only.
    Do not connect to laser, jammer, weapon, or active countermeasure.
    """
    def __init__(self, endpoint: str | None = None):
        self.endpoint = endpoint

    def point_camera(self, command: PTZCommand) -> bool:
        print(f"[PTZ] camera-only command: {command}")
        return True
