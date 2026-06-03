from __future__ import annotations

from station.ptz_client import PTZClient, PTZCommand


class RecordingAdapter:
    def __init__(self):
        self.commands = []

    def slew_to_cue(self, command: PTZCommand) -> bool:
        self.commands.append(command)
        return True


def test_ptz_client_dispatches_visual_confirmation_cue_only():
    adapter = RecordingAdapter()
    client = PTZClient(adapter=adapter)

    assert client.slew_to_cue(123.0, track_id="track_A", zoom=1.5) is True

    command = adapter.commands[0]
    assert command.azimuth_deg == 123.0
    assert command.track_id == "track_A"
    assert command.zoom == 1.5
