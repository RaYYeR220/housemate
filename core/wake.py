"""Streaming wake-word detection with openWakeWord-compatible ONNX models.

``OnnxWakeDetector`` reproduces openWakeWord 0.6.0's streaming pipeline (ONNX path) with
only onnxruntime + numpy; its scores match the package to float precision:

  int16 audio -> melspectrogram.onnx on the last 1280 + 480 samples -> spec / 10 + 2
  -> keep the last 76 mel frames -> embedding_model.onnx -> one 96-d embedding
  -> keep the last N embeddings (16 for the stock heads) -> wake head -> score in [0, 1]

One score per 80 ms (1280-sample) chunk. Initial state matches the package: mel buffer
of ones, feature buffer seeded with the embeddings of 4 s of uniform noise (seeded, so
deterministic), and the first 5 scores after a reset forced to 0.

The three ONNX sessions are shared by every detector in the process (see
``models.onnx_session``); each detector only holds its own small streaming buffers. Any
openWakeWord-format head works, e.g. a custom "hey hermes" model.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

from .audio import CHUNK_SAMPLES, SAMPLE_RATE

__all__ = ["OnnxWakeDetector", "WakeDetector"]

MEL_CONTEXT = 480  # 160 * 3 extra samples fed to the mel model, as openWakeWord does
EMB_WINDOW = 76  # mel frames per embedding
MEL_BINS = 32
EMB_DIM = 96
WARMUP_CHUNKS = 5  # openWakeWord zeroes the first 5 predictions


@runtime_checkable
class WakeDetector(Protocol):
    def feed(self, frame: bytes) -> float | None:
        """Add PCM16 LE mono 16 kHz audio; returns a score when an 80 ms chunk completes."""
        ...


_noise_lock = threading.Lock()
_noise_cache: dict[tuple[str, str, int], object] = {}


class OnnxWakeDetector:
    def __init__(
        self,
        mel_path: Path | str,
        emb_path: Path | str,
        head_path: Path | str,
        *,
        seed: int = 0,
        threads: int = 1,
    ) -> None:
        import numpy as np

        from .models import onnx_session

        self._np = np
        self._mel = onnx_session(mel_path, threads=threads)
        self._emb = onnx_session(emb_path, threads=threads)
        self._head = onnx_session(head_path, threads=threads)
        self._mel_in = self._mel.get_inputs()[0].name
        self._emb_in = self._emb.get_inputs()[0].name
        hin = self._head.get_inputs()[0]
        self._head_in = hin.name
        shape = list(hin.shape)
        if len(shape) != 3 or (isinstance(shape[2], int) and shape[2] != EMB_DIM):
            raise ValueError(f"{head_path}: not an openWakeWord head (input shape {shape}, want [1, N, {EMB_DIM}])")
        self.n_features = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 16
        self.head_path = Path(head_path)

        key = (str(Path(mel_path).resolve()), str(Path(emb_path).resolve()), int(seed))
        with _noise_lock:
            noise = _noise_cache.get(key)
        if noise is None:
            clip = np.random.RandomState(seed).randint(-1000, 1000, SAMPLE_RATE * 4).astype(np.int16)
            noise = self._embed_clip(clip)
            with _noise_lock:
                noise = _noise_cache.setdefault(key, noise)
        self._noise_feat = noise
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._pending = bytearray()
        self._raw = np.zeros(0, dtype=np.int16)
        self._melbuf = np.ones((EMB_WINDOW, MEL_BINS), dtype=np.float32)
        self._feat = np.array(self._noise_feat[-self.n_features :], dtype=np.float32, copy=True)
        self.chunks = 0
        self.last_score = 0.0

    def feed(self, frame: bytes) -> float | None:
        """Add audio (any length, normally one 640-byte frame).

        Returns the score when at least one 80 ms chunk completed (the max if several did,
        as openWakeWord does), else None.
        """
        self._pending += frame
        nbytes = CHUNK_SAMPLES * 2
        out: float | None = None
        while len(self._pending) >= nbytes:
            chunk = self._np.frombuffer(bytes(self._pending[:nbytes]), dtype="<i2")
            del self._pending[:nbytes]
            s = self._process_chunk(chunk)
            out = s if out is None else max(out, s)
        return out

    # --- internals -----------------------------------------------------------------------

    def _melspec(self, x):
        np = self._np
        spec = self._mel.run(None, {self._mel_in: x.astype(np.float32)[None, :]})[0]
        return np.squeeze(spec) / 10 + 2  # (frames, 32)

    def _embed_clip(self, x):
        """openWakeWord ``AudioFeatures._get_embeddings`` (initial feature buffer only)."""
        np = self._np
        spec = self._melspec(x)
        wins = [spec[i : i + EMB_WINDOW] for i in range(0, spec.shape[0], 8)]
        wins = [w for w in wins if w.shape[0] == EMB_WINDOW]
        batch = np.expand_dims(np.array(wins), axis=-1).astype(np.float32)
        return self._emb.run(None, {self._emb_in: batch})[0].squeeze()

    def _process_chunk(self, chunk) -> float:
        np = self._np
        self._raw = np.concatenate([self._raw, chunk])[-(CHUNK_SAMPLES + MEL_CONTEXT) :]
        self._melbuf = np.vstack([self._melbuf, self._melspec(self._raw)])[-EMB_WINDOW:]
        win = self._melbuf[None, :, :, None].astype(np.float32)
        emb = self._emb.run(None, {self._emb_in: win})[0].squeeze()
        self._feat = np.vstack([self._feat, emb[None, :]])[-self.n_features :]
        out = self._head.run(None, {self._head_in: self._feat[None].astype(np.float32)})[0]
        score = float(np.asarray(out).reshape(-1)[0])
        self.chunks += 1
        if self.chunks <= WARMUP_CHUNKS:
            score = 0.0
        self.last_score = score
        return score
