"""VAD-based endpointer: cuts a continuous 20 ms frame stream into utterances.

A segment starts on the first speech frame (with ``preroll_ms`` of audio before it) and
ends after ``end_silence_ms`` of non-speech. Segments with less than ``min_speech_ms`` of
speech are dropped; a segment that reaches ``max_utterance_s`` is emitted truncated, and
the rest of that talk is ignored until the next silence gap.

The endpointer is fed every frame (so the VAD state and the preroll stay warm); the
listener decides which utterances it wants and calls ``restart()`` when a capture begins.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .audio import FRAME_BYTES, FRAME_MS, FRAME_S, SAMPLE_RATE
from .vad import Vad

__all__ = ["Endpointer", "Utterance"]


@dataclass(frozen=True)
class Utterance:
    pcm: bytes  # PCM16 LE mono 16 kHz
    started_at: float  # wall-clock time of the first sample in ``pcm``
    ended_at: float  # wall-clock time of the last sample in ``pcm``
    truncated: bool  # hit max_utterance_s

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / (2 * SAMPLE_RATE)


class Endpointer:
    def __init__(
        self,
        vad: Vad,
        end_silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_s: float = 15,
        preroll_ms: int = 300,
        *,
        speech_threshold: float = 0.5,
        tail_ms: int = 300,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.vad = vad
        self.speech_threshold = speech_threshold
        self._end_frames = max(1, math.ceil(end_silence_ms / FRAME_MS))
        self._min_speech = max(1, math.ceil(min_speech_ms / FRAME_MS))
        self._max_frames = max(1, int(round(max_utterance_s * 1000 / FRAME_MS)))
        self._tail = min(self._end_frames, max(0, math.ceil(tail_ms / FRAME_MS)))
        self._pre: deque[bytes] = deque(maxlen=max(0, preroll_ms // FRAME_MS))
        self._clock = clock
        self.last_prob = 0.0
        self._clear_segment()
        self._cooldown = False

    # --- state -------------------------------------------------------------------------

    @property
    def max_frames(self) -> int:
        """``max_utterance_s`` in 20 ms frames."""
        return self._max_frames

    @property
    def min_speech_frames(self) -> int:
        """``min_speech_ms`` in 20 ms frames."""
        return self._min_speech

    @property
    def active(self) -> bool:
        """True while a speech segment is being collected."""
        return self._seg is not None

    def _clear_segment(self) -> None:
        self._seg: list[bytes] | None = None
        self._speech = 0  # speech frames in the segment
        self._silence = 0  # consecutive non-speech frames
        self._last_speech = 0  # segment length right after the newest speech frame

    def restart(self) -> None:
        """Drop the current segment (and any post-truncation wait); keep preroll + VAD state.

        The next speech frame starts a fresh segment, with the most recent audio as preroll.
        """
        if self._seg is not None and self._pre.maxlen:
            self._pre.extend(self._seg[-self._pre.maxlen :])
        self._clear_segment()
        self._cooldown = False

    def reset(self) -> None:
        """Forget everything, including the VAD's recurrent state and the preroll."""
        self._pre.clear()
        self._clear_segment()
        self._cooldown = False
        reset = getattr(self.vad, "reset", None)
        if callable(reset):
            reset()

    # --- streaming -----------------------------------------------------------------------

    def push(self, frame: bytes) -> Utterance | None:
        """Feed one 20 ms frame; returns an utterance when one is complete."""
        prob = float(self.vad.speech_prob(frame))
        self.last_prob = prob
        speech = prob >= self.speech_threshold

        if self._cooldown:  # after truncation: wait for the talk to pause before re-arming
            self._silence = 0 if speech else self._silence + 1
            if self._silence >= self._end_frames:
                self._cooldown = False
                self._silence = 0
            self._pre.append(frame)
            return None

        if self._seg is None:
            if not speech:
                self._pre.append(frame)
                return None
            self._seg = list(self._pre)
            self._pre.clear()

        seg = self._seg
        seg.append(frame)
        if speech:
            self._speech += 1
            self._silence = 0
            self._last_speech = len(seg)
        else:
            self._silence += 1

        if len(seg) >= self._max_frames:
            utt = self._emit(seg, len(seg), truncated=True)
            self._end_segment(seg)
            self._cooldown = True
            return utt
        if self._silence >= self._end_frames:
            utt = self._emit(seg, min(len(seg), self._last_speech + self._tail), truncated=False)
            self._end_segment(seg)
            return utt
        return None

    def _end_segment(self, seg: list[bytes]) -> None:
        if self._pre.maxlen:
            self._pre.extend(seg[-self._pre.maxlen :])
        self._clear_segment()

    def _emit(self, seg: list[bytes], upto: int, *, truncated: bool) -> Utterance | None:
        if self._speech < self._min_speech:
            return None
        now = self._clock()
        started = now - len(seg) * FRAME_S
        pcm = b"".join(seg[:upto])
        return Utterance(
            pcm=pcm,
            started_at=started,
            ended_at=started + len(pcm) / FRAME_BYTES * FRAME_S,
            truncated=truncated,
        )
