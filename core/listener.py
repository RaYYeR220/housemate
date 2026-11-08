"""Per-device listener: wake word -> arbitration -> endpointed utterance.

Threading model: ``feed`` and the control methods are called on the hub's event loop
and only append to a queue (cheap). All inference (wake model, VAD, endpointer) runs in
a worker from ``executor``, strictly in arrival order and never two batches of the same
device at once. Utterance and wake callbacks are marshalled back to the event loop with
``call_soon_threadsafe`` (or the injected ``dispatch``). Tests inject ``InlineExecutor``
and ``dispatch=lambda fn: fn()`` to run everything synchronously.

Triggers:
- ``wake``: the wake score crosses the threshold (``threshold + echo_margin`` while any
  device in the earshot group plays), the device offers itself to the ``WakeArbiter`` and
  starts capturing; if it loses the round the capture is dropped silently.
- ``followup``: within ``open_followup(seconds)`` speech is captured without a wake word,
  except while any device in the earshot group is playing (echo guard).
- ``ptt``: tap-to-talk captures everything between ``set_ptt(True)`` and ``set_ptt(False)``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Any, Callable, Literal

from .arbiter import WakeArbiter
from .audio import FRAME_BYTES, FRAME_MS, FRAME_S, mean_square
from .endpointer import Endpointer, Utterance
from .wake import WakeDetector

__all__ = ["DeviceListener", "InlineExecutor", "TriggerKind", "shared_executor"]

log = logging.getLogger(__name__)

TriggerKind = Literal["wake", "followup", "ptt"]

_IDLE, _PENDING, _CAPTURE, _FOLLOWUP, _PTT, _PTT_FULL = "idle", "pending", "capture", "followup", "ptt", "ptt_full"


class InlineExecutor(Executor):
    """Runs submitted work immediately on the calling thread (tests, offline tools)."""

    def submit(self, fn, /, *args, **kwargs) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 - surfaced through the future
            fut.set_exception(exc)
        return fut


_shared: ThreadPoolExecutor | None = None
_shared_lock = threading.Lock()


def shared_executor() -> ThreadPoolExecutor:
    """A small process-wide pool for audio inference (onnxruntime releases the GIL)."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = ThreadPoolExecutor(
                max_workers=max(1, min(4, (os.cpu_count() or 2) // 2)),
                thread_name_prefix="housemate-audio",
            )
        return _shared


class DeviceListener:
    def __init__(
        self,
        device_id: str,
        wake: WakeDetector,
        endpointer: Endpointer,
        arbiter: WakeArbiter,
        threshold: float = 0.5,
        *,
        room: str | None = None,
        group: str | None = None,
        executor: Executor | None = None,
        dispatch: Callable[[Callable[[], None]], Any] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        echo_margin: float = 0.15,
        wake_listen_s: float = 5.0,
        refractory_s: float = 1.0,
        max_queue_s: float = 5.0,
    ) -> None:
        self.device_id = device_id
        self.wake = wake
        self.endpointer = endpointer
        self.arbiter = arbiter
        self.threshold = float(threshold)
        self.echo_margin = echo_margin
        self.room = room
        self.group = group
        self._executor = executor
        self._dispatch_fn = dispatch
        self._loop = loop
        self._clock = clock
        self._wall = wall_clock
        self._wake_listen = max(1, round(wake_listen_s / FRAME_S))
        self._refractory = max(0, round(refractory_s / FRAME_S))
        self._max_queue = max(10, round(max_queue_s / FRAME_S))
        self._max_ptt = endpointer.max_frames
        self._min_ptt = endpointer.min_speech_frames
        self._playing_self = False

        self._utt_cbs: list[Callable[[str, Utterance, TriggerKind], None]] = []
        self._wake_cbs: list[Callable[[str], None]] = []

        # queue (filled on the loop, drained in a worker)
        self._q: deque[tuple[Any, float]] = deque()
        self._qframes = 0
        self._qlock = threading.Lock()
        self._scheduled = False
        self._closed = False

        # worker-side state
        self._state = _IDLE
        self._muted = False
        self._n = 0  # frames processed: the listener's audio clock
        self._since = 0  # frame count when the current capture/window began
        self._window_end = 0
        self._refractory_until = 0
        self._armed = True  # wake re-arms once the score falls below the threshold
        self._pending_utt: Utterance | None = None
        self._ptt: list[bytes] = []
        self._energy: deque[float] = deque(maxlen=round(1.0 / FRAME_S))
        self.last_score = 0.0
        self.stats = {"frames": 0, "dropped": 0, "offers": 0, "won": 0, "lost": 0, "utterances": 0, "errors": 0}

    # --- public API (event-loop side) ----------------------------------------------------

    def on_utterance(self, cb: Callable[[str, Utterance, TriggerKind], None]) -> None:
        """``cb(device_id, utterance, kind)``, called on the event loop."""
        self._utt_cbs.append(cb)

    def on_wake(self, cb: Callable[[str], None]) -> None:
        """``cb(device_id)`` when this device won a wake round (show "listening", barge in)."""
        self._wake_cbs.append(cb)

    def feed(self, frame: bytes) -> None:
        """One 20 ms mic frame (640 bytes). Cheap: queues it for the worker."""
        if len(frame) != FRAME_BYTES:
            return
        self._enqueue(bytes(frame))

    def open_followup(self, seconds: float = 6.0) -> None:
        self._enqueue(("followup", float(seconds)))

    def set_playing(self, on: bool) -> None:
        """This device started/stopped playing audio (echo guard for its whole earshot group)."""
        self._playing_self = bool(on)
        room, group = self._earshot()
        self.arbiter.set_playing(self.device_id, room, group, self._playing_self)

    def set_ptt(self, on: bool) -> None:
        self._enqueue(("ptt", bool(on)))

    def set_muted(self, on: bool) -> None:
        self._enqueue(("mute", bool(on)))

    def set_room(self, room: str | None, group: str | None = None) -> None:
        """The device moved (``room.set``) or its room's earshot group changed."""
        self._enqueue(("room", (room, group)))

    def close(self) -> None:
        """The device disconnected: stop processing and release its arbiter state."""
        with self._qlock:
            self._closed = True
            self._q.clear()
            self._qframes = 0
        self.arbiter.forget(self.device_id)

    @property
    def state(self) -> str:
        return "muted" if self._muted else self._state

    # --- queue -----------------------------------------------------------------------------

    def _enqueue(self, item: Any) -> None:
        if self._loop is None and self._dispatch_fn is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        t = self._clock()
        with self._qlock:
            if self._closed:
                return
            if isinstance(item, bytes):
                if self._qframes >= self._max_queue:  # worker far behind: drop the oldest audio
                    for i, (old, _) in enumerate(self._q):
                        if isinstance(old, bytes):
                            del self._q[i]
                            self._qframes -= 1
                            self.stats["dropped"] += 1
                            break
                self._qframes += 1
            self._q.append((item, t))
            if self._scheduled:
                return
            self._scheduled = True
        executor = self._executor or shared_executor()
        try:
            executor.submit(self._drain)
        except RuntimeError:  # executor shut down (gateway stopping)
            with self._qlock:
                self._scheduled = False

    def _drain(self) -> None:
        while True:
            with self._qlock:
                if not self._q or self._closed:
                    self._q.clear()
                    self._qframes = 0
                    self._scheduled = False
                    return
                batch = list(self._q)
                self._q.clear()
                self._qframes = 0
            for item, t in batch:
                try:
                    if isinstance(item, bytes):
                        self._process(item, t)
                    else:
                        self._command(item[0], item[1])
                except Exception:
                    if self.stats["errors"] % 500 == 0:
                        log.exception("housemate: listener %s failed on a frame", self.device_id)
                    self.stats["errors"] += 1

    # --- worker side -----------------------------------------------------------------------

    def _earshot(self) -> tuple[str, str | None]:
        if self.room:
            return self.room, self.group
        return f"~{self.device_id}", None  # roomless device: its own earshot

    def _group_playing(self) -> bool:
        room, group = self._earshot()
        return self.arbiter.group_playing(room, group)

    def _command(self, name: str, arg: Any) -> None:
        if name == "room":
            self.room, self.group = arg
            if self._playing_self:  # move the playback flag to the new earshot group
                room, group = self._earshot()
                self.arbiter.set_playing(self.device_id, room, group, True)
            return
        if name == "mute":
            if arg and not self._muted:
                self._muted = True
                self._to_idle()
                self.endpointer.reset()
                reset = getattr(self.wake, "reset", None)
                if callable(reset):
                    reset()
            elif not arg and self._muted:
                self._muted = False
                self._armed = False
            return
        if self._muted:
            return
        if name == "followup":
            if self._state in (_IDLE, _FOLLOWUP):
                self._state = _FOLLOWUP
                self._since = self._n
                self._window_end = self._n + max(1, round(arg / FRAME_S))
                self.endpointer.restart()
        elif name == "ptt":
            if arg:
                self._to_idle()
                self._state = _PTT
                self._ptt = []
            elif self._state in (_PTT, _PTT_FULL):
                frames, full = self._ptt, self._state == _PTT_FULL
                self._to_idle()
                self.endpointer.restart()
                if not full and len(frames) >= self._min_ptt:
                    self._emit_ptt(frames, truncated=False)

    def _to_idle(self) -> None:
        self._state = _IDLE
        self._pending_utt = None
        self._ptt = []

    def _process(self, frame: bytes, t: float) -> None:
        if self._muted:
            return
        self._n += 1
        self.stats["frames"] += 1
        self._energy.append(mean_square(frame))
        utt = self.endpointer.push(frame)
        st = self._state

        if st == _PTT:
            self._ptt.append(frame)
            if len(self._ptt) >= self._max_ptt:
                self._emit_ptt(self._ptt, truncated=True)
                self._ptt = []
                self._state = _PTT_FULL
        elif st == _PENDING:
            if utt is not None:
                self._pending_utt = utt
        elif st == _CAPTURE:
            if utt is not None:
                self._to_idle()
                self._emit(utt, "wake")
            elif not self.endpointer.active and self._n - self._since > self._wake_listen:
                self._to_idle()  # woke, but nothing was said
        elif st == _FOLLOWUP:
            if self._group_playing():
                self.endpointer.restart()  # never capture the group's own playback
                if self._n >= self._window_end:
                    self._to_idle()
            elif utt is not None:
                self._to_idle()
                self._emit(utt, "followup")
            elif not self.endpointer.active and self._n >= self._window_end:
                self._to_idle()

        score = self.wake.feed(frame)
        if score is not None:
            self._on_score(float(score), t)

        if self._state == _PENDING:
            self._check_arbitration()

    def _on_score(self, score: float, t: float) -> None:
        self.last_score = score
        if score < self.threshold:
            self._armed = True
        if self._state not in (_IDLE, _FOLLOWUP):
            return
        if not self._armed or self._n < self._refractory_until:
            return
        need = self.threshold + (self.echo_margin if self._group_playing() else 0.0)
        if score < need:
            return
        self._armed = False
        self._refractory_until = self._n + self._refractory
        energy = math.sqrt(sum(self._energy) / len(self._energy)) if self._energy else 0.0
        room, group = self._earshot()
        self.arbiter.offer(self.device_id, room, group, score, energy, at=t)
        self.stats["offers"] += 1
        self.endpointer.restart()
        self._state = _PENDING
        self._since = self._n
        self._pending_utt = None

    def _check_arbitration(self) -> None:
        outcome = self.arbiter.outcome(self.device_id)
        if outcome == "won":
            self.stats["won"] += 1
            held = self._pending_utt
            self._state = _CAPTURE
            self._pending_utt = None
            self._call(self._wake_cbs, self.device_id)
            if held is not None:
                self._to_idle()
                self._emit(held, "wake")
        elif outcome == "lost":
            self.stats["lost"] += 1
            self._to_idle()
            self._refractory_until = self._n + self._refractory
        elif self._n - self._since > self._wake_listen:
            self._to_idle()  # the arbiter forgot us (device re-registered): give up quietly

    # --- output ------------------------------------------------------------------------------

    def _emit_ptt(self, frames: list[bytes], *, truncated: bool) -> None:
        pcm = b"".join(frames)
        end = self._wall()
        start = end - len(frames) * FRAME_MS / 1000
        self._emit(Utterance(pcm=pcm, started_at=start, ended_at=end, truncated=truncated), "ptt")

    def _emit(self, utt: Utterance, kind: TriggerKind) -> None:
        self.stats["utterances"] += 1
        self._call(self._utt_cbs, self.device_id, utt, kind)

    def _call(self, cbs: list[Callable[..., None]], *args: Any) -> None:
        for cb in list(cbs):
            self._dispatch(_safe(cb, args))

    def _dispatch(self, fn: Callable[[], None]) -> None:
        if self._dispatch_fn is not None:
            self._dispatch_fn(fn)
            return
        loop = self._loop
        if loop is None:
            fn()
            return
        try:
            loop.call_soon_threadsafe(fn)
        except RuntimeError:  # loop closed: the gateway is shutting down
            log.debug("housemate: dropped a listener callback for %s (loop closed)", self.device_id)


def _safe(cb: Callable[..., None], args: tuple) -> Callable[[], None]:
    def run() -> None:
        try:
            cb(*args)
        except Exception:
            log.exception("housemate: listener callback failed")

    return run
