"""Paced PCM playout to one satellite.

Hermes's streaming TTS hands the adapter PCM in irregular chunks (52 bytes .. 5 KB, odd sizes
included) and much faster than real time (18 s of speech in 7 s). Sending it all at once would
make a phone buffer the whole answer, so a barge-in ("hey jarvis" while it talks) or a dropped
socket could not cut it short, and the satellite's playback-done signal would drift.

``PcmRealigner`` keeps every chunk on whole 16-bit samples (an odd trailing byte waits for the
next chunk). ``Playout`` buffers the stream and sends it to the device at most ``lead_s`` ahead of
the device's play position, in frames of at most ``max_frame_s``; it holds the first frame until
``prebuffer_s`` of audio is in (or ``prebuffer_wait_s`` passed) so playback does not start with a
gap. A failed send (device gone) ends the playout quietly with ``"failed"``.

No Hermes imports; time and sleeping are injectable for tests.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

__all__ = ["PcmRealigner", "Playout"]


class PcmRealigner:
    """Cuts a byte stream into whole PCM16 samples, carrying an odd byte to the next chunk."""

    def __init__(self) -> None:
        self._carry = b""

    @property
    def pending(self) -> int:
        return len(self._carry)

    def feed(self, chunk: bytes) -> bytes:
        data = self._carry + bytes(chunk)
        cut = len(data) - (len(data) % 2)
        self._carry = data[cut:]
        return data[:cut]

    def flush(self) -> bytes:
        """End of stream: a dangling half sample is dropped (sending it would shift every later sample)."""
        self._carry = b""
        return b""


class Playout:
    def __init__(
        self,
        send: Callable[[bytes], Awaitable[bool]],
        *,
        sample_rate: int,
        lead_s: float = 0.5,
        max_frame_s: float = 0.2,
        min_frame_s: float = 0.04,
        low_water_s: float = 0.15,
        prebuffer_s: float = 0.12,
        prebuffer_wait_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._send = send
        self.sample_rate = int(sample_rate)
        self._bps = self.sample_rate * 2
        self.lead_s = float(lead_s)
        self.max_frame_s = float(max_frame_s)
        self.min_frame_s = float(min_frame_s)
        self.low_water_s = float(low_water_s)
        self.prebuffer_s = float(prebuffer_s)
        self.prebuffer_wait_s = float(prebuffer_wait_s)
        self._clock = clock
        self._sleep = sleep
        self._buf = bytearray()
        self._finished = False
        self._aborted = False
        self._closed = False
        self._wake = asyncio.Event()
        self.sent_bytes = 0
        self.written_bytes = 0
        self.first_sent_at: float | None = None
        self.result: str | None = None

    # --- producer side -------------------------------------------------------------------------

    def write(self, pcm: bytes) -> None:
        """Queue whole samples (use ``PcmRealigner`` first). Ignored once finished or ended."""
        if self._closed or self._aborted or self._finished or not pcm:
            return
        self._buf += pcm
        self.written_bytes += len(pcm)
        self._wake.set()

    def finish(self) -> None:
        """No more audio: play out what is buffered, then end with ``"done"``."""
        self._finished = True
        self._wake.set()

    def abort(self) -> None:
        """Stop now; what is buffered is dropped."""
        self._aborted = True
        self._wake.set()

    @property
    def ended(self) -> bool:
        return self._closed

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def sent_s(self) -> float:
        return self.sent_bytes / self._bps

    @property
    def buffered_s(self) -> float:
        return len(self._buf) / self._bps

    # --- the pump --------------------------------------------------------------------------------

    def _bytes(self, seconds: float) -> int:
        n = int(max(0.0, seconds) * self._bps)
        return n - (n % 2)

    async def _wait(self, timeout: float | None) -> None:
        self._wake.clear()
        if timeout is None:
            await self._wake.wait()
            return
        try:
            await asyncio.wait_for(self._wake.wait(), max(0.001, timeout))
        except asyncio.TimeoutError:
            pass

    def _end(self, result: str) -> str:
        self._closed = True
        self.result = result
        self._buf.clear()
        return result

    async def run(self) -> str:
        """Send everything at a real-time pace. Returns ``"done"``, ``"aborted"`` or ``"failed"``."""
        loop = asyncio.get_running_loop()
        started = False
        pre_deadline: float | None = None
        t0 = 0.0  # clock time at which the device's play position was 0
        prebuffer = self._bytes(self.prebuffer_s)
        min_frame = max(2, self._bytes(self.min_frame_s))
        try:
            while True:
                if self._aborted:
                    return self._end("aborted")
                if not self._buf:
                    if self._finished:
                        return self._end("done")
                    await self._wait(None)
                    continue
                if not started:
                    if len(self._buf) < prebuffer and not self._finished:
                        if pre_deadline is None:
                            pre_deadline = loop.time() + self.prebuffer_wait_s
                        remaining = pre_deadline - loop.time()
                        if remaining > 0:
                            await self._wait(remaining)
                            continue
                    started = True
                    t0 = self._clock()
                now = self._clock()
                ahead = self.sent_s - (now - t0)
                if ahead < 0:  # the device ran dry: its play position restarts now
                    t0 = now - self.sent_s
                    ahead = 0.0
                room = self.lead_s - ahead
                if room < self.min_frame_s - 1e-6:
                    await self._sleep(max(0.005, self.min_frame_s - room))
                    continue
                if len(self._buf) < min_frame and not self._finished and ahead > self.low_water_s:
                    await self._wait(ahead - self.low_water_s)  # more is coming: send bigger frames
                    continue
                n = min(len(self._buf), max(2, self._bytes(min(room, self.max_frame_s))))
                n -= n % 2
                chunk = bytes(self._buf[:n])
                del self._buf[:n]
                if not await self._send(chunk):
                    return self._end("failed")
                if self.first_sent_at is None:
                    self.first_sent_at = now
                self.sent_bytes += n
        except asyncio.CancelledError:
            self._end("aborted")
            raise
