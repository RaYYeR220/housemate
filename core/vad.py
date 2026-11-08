"""Voice activity detection: Silero VAD (ONNX) with a dependency-free energy fallback.

Both take 20 ms PCM16 LE frames at 16 kHz and return a speech probability in [0, 1].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, runtime_checkable

from .audio import FRAME_SAMPLES, SAMPLE_RATE, rms_dbfs

__all__ = ["EnergyVad", "SileroVad", "Vad"]


@runtime_checkable
class Vad(Protocol):
    def speech_prob(self, frame: bytes) -> float: ...


class SileroVad:
    """Silero VAD v5/v6 ONNX, the way its own OnnxWrapper runs it.

    20 ms frames are buffered into 512-sample windows (64-sample context carried over,
    recurrent state ``[2, 1, 128]``). ``speech_prob`` returns the newest completed window's
    probability, so it is at most ~31 ms stale. The ONNX session is shared across devices.
    """

    WINDOW = 512
    CONTEXT = 64

    def __init__(self, path: Path | str, *, threads: int = 1) -> None:
        import numpy as np

        from .models import onnx_session

        self._np = np
        self._s = onnx_session(path, threads=threads)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(self.CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self.prob = 0.0

    def speech_prob(self, frame: bytes) -> float:
        np = self._np
        x = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, x])
        while self._pending.shape[0] >= self.WINDOW:
            win, self._pending = self._pending[: self.WINDOW], self._pending[self.WINDOW :]
            inp = np.concatenate([self._ctx, win])[None, :]
            out, self._state = self._s.run(None, {"input": inp, "state": self._state, "sr": self._sr})
            self._ctx = inp[0, -self.CONTEXT :]
            self.prob = float(out[0][0])
        return self.prob


class EnergyVad:
    """RMS in dBFS against an adaptive noise floor, squashed to [0, 1] (no model needed)."""

    def __init__(
        self,
        margin_db: float = 12.0,
        min_dbfs: float = -50.0,
        knee_db: float = 3.0,
        rise_db_per_s: float = 1.0,
    ) -> None:
        self.margin_db = margin_db
        self.min_dbfs = min_dbfs
        self.knee_db = knee_db
        self._rise = rise_db_per_s * FRAME_SAMPLES / SAMPLE_RATE
        self.reset()

    def reset(self) -> None:
        self.floor_db = -70.0
        self.prob = 0.0

    def speech_prob(self, frame: bytes) -> float:
        db = rms_dbfs(frame)
        # fast fall, slow rise noise-floor tracker
        self.floor_db = db if db < self.floor_db else self.floor_db + self._rise
        thr = max(self.floor_db + self.margin_db, self.min_dbfs)
        z = (db - thr) / self.knee_db
        self.prob = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
        return self.prob
