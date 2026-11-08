"""Synthesise web/chime.wav: a short, soft two-tone chime (G5 then C6).

Pure standard library and fully deterministic, so the committed WAV can always be
re-made from this recipe: ``python scripts/make_chime.py``.
"""

from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

RATE = 22_050
DURATION_S = 1.2
PEAK = 0.6  # of full scale: clear on a phone speaker, never clipped
FADE_S = 0.08  # cosine fade at the very end, so the file never clicks

# (start s, frequency Hz, decay time constant s, gain)
NOTES = (
    (0.00, 783.99, 0.30, 1.0),  # G5
    (0.22, 1046.50, 0.42, 0.9),  # C6
)
# Partials as (frequency multiple, relative gain): a little brightness, no harshness.
PARTIALS = ((1.0, 1.0), (2.0, 0.22), (3.0, 0.06))
ATTACK_S = 0.006

OUT = Path(__file__).resolve().parents[1] / "web" / "chime.wav"


def _samples() -> list[float]:
    n = int(RATE * DURATION_S)
    out = [0.0] * n
    for start, freq, tau, gain in NOTES:
        first = int(start * RATE)
        for i in range(first, n):
            t = (i - first) / RATE
            env = gain * min(1.0, t / ATTACK_S) * math.exp(-t / tau)
            out[i] += env * sum(g * math.sin(2 * math.pi * freq * k * t) for k, g in PARTIALS)
    fade = int(FADE_S * RATE)
    for j in range(fade):
        out[n - fade + j] *= 0.5 * (1 + math.cos(math.pi * (j + 1) / fade))
    return out


def render() -> bytes:
    """The WAV file bytes: PCM16 LE mono at RATE."""
    samples = _samples()
    scale = PEAK * 32767 / max(abs(s) for s in samples)
    pcm = struct.pack(f"<{len(samples)}h", *(round(s * scale) for s in samples))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def main() -> None:
    data = render()
    OUT.write_bytes(data)
    print(f"wrote {OUT} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
