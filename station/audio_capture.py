from __future__ import annotations
from typing import Optional, Iterator
import numpy as np
import sounddevice as sd

def list_input_devices():
    out = []
    for idx, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({
                "id": idx,
                "name": d["name"],
                "max_input_channels": int(d["max_input_channels"]),
                "default_samplerate": int(d.get("default_samplerate", 44100)),
            })
    return out

def audio_blocks(device_id: Optional[int], sample_rate: int, channels: int, window_sec: float) -> Iterator[np.ndarray]:
    block = int(sample_rate * window_sec)
    with sd.InputStream(device=device_id, channels=channels, samplerate=sample_rate, blocksize=block, dtype="float32") as stream:
        while True:
            audio, _ = stream.read(block)
            yield np.asarray(audio, dtype=np.float32)

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)
