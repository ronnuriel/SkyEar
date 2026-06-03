from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass
class PTZCommand:
    azimuth_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    zoom: Optional[float] = None
    track_id: Optional[str] = None


class PTZAdapter(Protocol):
    """Camera-only visual confirmation adapter."""

    def slew_to_cue(self, command: PTZCommand) -> bool:
        ...


class MockPTZAdapter:
    def slew_to_cue(self, command: PTZCommand) -> bool:
        print(f"[PTZ:mock] camera-only cue: {command}")
        return True


class ONVIFPTZAdapter:
    """Placeholder ONVIF adapter for visual confirmation cameras only."""

    def __init__(self, endpoint: str, username: str | None = None, password: str | None = None):
        self.endpoint = endpoint
        self.username = username
        self.password = password

    def slew_to_cue(self, command: PTZCommand) -> bool:
        print(f"[PTZ:onvif] visual-confirmation cue endpoint={self.endpoint} command={command}")
        return True


class SerialPTZAdapter:
    """Placeholder serial adapter for visual confirmation cameras only."""

    def __init__(self, port: str, baudrate: int = 9600):
        self.port = port
        self.baudrate = int(baudrate)

    def slew_to_cue(self, command: PTZCommand) -> bool:
        print(f"[PTZ:serial] visual-confirmation cue port={self.port} command={command}")
        return True

class PTZClient:
    """
    Safe visual confirmation only.
    Do not connect to laser, jammer, weapon, or active countermeasure.
    """
    def __init__(self, endpoint: str | None = None, adapter: PTZAdapter | None = None):
        self.endpoint = endpoint
        self.adapter = adapter or MockPTZAdapter()

    def point_camera(self, command: PTZCommand) -> bool:
        return self.slew_to_cue(
            bearing_deg=command.azimuth_deg,
            elevation_deg=command.elevation_deg,
            zoom=command.zoom,
            track_id=command.track_id,
        )

    def slew_to_cue(
        self,
        bearing_deg: float | None,
        *,
        track_id: str | None = None,
        elevation_deg: float | None = None,
        zoom: float | None = None,
    ) -> bool:
        command = PTZCommand(
            azimuth_deg=bearing_deg,
            elevation_deg=elevation_deg,
            zoom=zoom,
            track_id=track_id,
        )
        return self.adapter.slew_to_cue(command)
