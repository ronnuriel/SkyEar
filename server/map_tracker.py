from typing import Optional
from shared.event_schema import AcousticEvent, GeoPoint

def estimate_location_placeholder(events: list[AcousticEvent]) -> Optional[GeoPoint]:
    # TODO: implement bearing intersection when multiple synchronized-array stations exist.
    return None
