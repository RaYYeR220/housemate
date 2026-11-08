"""PCM helpers shared by the audio pipeline (16 kHz mono PCM16 LE, 20 ms frames).

Only numpy and the standard library; numpy is imported lazily so that importing the
package stays cheap.
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from typing import Iterator

__all__ = [
    "CHUNK_SAMPLES",
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_S",
    "FRAME_SAMPLES",
    "SAMPLE_RATE",
    "duration_s",
    "iter_frames",
    "mean_square",
    "pcm_to_wav",
    "read_wav",
    "rms",
    "rms_dbfs",
]

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_S = FRAME_MS / 1000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
FRAME_BYTES = FRAME_SAMPLES * 2  # 640
CHUNK_SAMPLES = 1280  # 80 ms: one wake-word score


def _samples(pcm: bytes):
    import numpy as np

    return np.frombuffer(pcm, dtype="<i2")


def mean_square(frame: bytes) -> float:
    """Mean of squared samples, normalised to full scale (0.0 .. 1.0)."""
    if len(frame) < 2:
        return 0.0
    x = _samples(frame[: len(frame) // 2 * 2]).astype("float32") / 32768.0
    return float((x * x).mean())


def rms(frame: bytes) -> float:
    """Root mean square level, normalised to full scale (0.0 .. 1.0)."""
    return math.sqrt(mean_square(frame))


def rms_dbfs(frame: bytes) -> float:
    return 10 * math.log10(mean_square(frame) + 1e-10)


def duration_s(pcm: bytes, rate: int = SAMPLE_RATE) -> float:
    return len(pcm) / (2 * rate)


def iter_frames(pcm: bytes, frame_bytes: int = FRAME_BYTES) -> Iterator[bytes]:
    """Split PCM into fixed-size frames; the last partial frame is zero-padded."""
    pcm = bytes(pcm)
    for i in range(0, len(pcm), frame_bytes):
        part = pcm[i : i + frame_bytes]
        if len(part) < frame_bytes:
            part += b"\x00" * (frame_bytes - len(part))
        yield part


def pcm_to_wav(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap mono PCM16 LE in a WAV container (44-byte header)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm))
    return buf.getvalue()


def read_wav(path: Path | str) -> bytes:
    """PCM bytes of a 16 kHz mono PCM16 WAV file."""
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SAMPLE_RATE):
            raise ValueError(f"{path}: need 16 kHz mono PCM16")
        return w.readframes(w.getnframes())
