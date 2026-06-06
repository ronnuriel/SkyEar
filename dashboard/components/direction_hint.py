from __future__ import annotations

from typing import Any


def direction_hint_state(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata") or {}
    hint = event.get("two_mic_look_hint") or metadata.get("two_mic_look_hint")
    hidden_reason = metadata.get("two_mic_suppressed_reason")
    return {
        "hint": hint,
        "hidden": bool(hidden_reason or (hint and "HIDDEN" in str(hint))),
        "hidden_reason": hidden_reason,
        "front_back_ambiguous": event.get("two_mic_front_back_ambiguous")
        if event.get("two_mic_front_back_ambiguous") is not None
        else metadata.get("two_mic_front_back_ambiguous"),
    }
