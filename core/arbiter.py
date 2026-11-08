"""Cross-device wake arbitration and earshot-group playback state.

Several satellites in earshot of each other hear the same wake word. Each one offers its
score and speech energy; offers in the same earshot group within ``window_ms`` of the
first one form a round, and when the window closes exactly one device wins: the highest
wake score, with near-equal scores (within ``score_tie``) going to the louder device, the
one the speaker is most likely facing. Everyone else in the round loses and must drop the
capture: no reply, no follow-up window.

An earshot group is the room's ``earshot_group`` when set, else the room itself.

The arbiter also tracks which devices are playing audio, per earshot group, so listeners
can apply the echo guard to the whole group. All methods are thread-safe (listeners call
them from worker threads).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

__all__ = ["WakeArbiter", "WakeWinner", "earshot_key"]

Outcome = Literal["won", "lost"]


def earshot_key(room: str | None, group: str | None) -> str:
    """The arbitration bucket of a device: its earshot group, else its room."""
    if group:
        return f"group:{group}"
    return f"room:{room or ''}"


@dataclass(frozen=True)
class WakeWinner:
    device_id: str
    room: str
    group: str | None
    score: float
    energy: float
    at: float  # when the winning device heard the wake (arbiter clock)
    losers: tuple[str, ...] = ()


@dataclass
class _Offer:
    device_id: str
    room: str
    group: str | None
    score: float
    energy: float
    at: float


@dataclass
class _Round:
    start: float
    offers: dict[str, _Offer] = field(default_factory=dict)


class WakeArbiter:
    def __init__(
        self,
        window_ms: int = 400,
        clock: Callable[[], float] = time.monotonic,
        *,
        score_tie: float = 0.05,
        max_pending: int = 64,
    ) -> None:
        self.window_s = window_ms / 1000
        self.score_tie = score_tie
        self._clock = clock
        self._lock = threading.RLock()
        self._open: dict[str, _Round] = {}
        self._decided: dict[str, tuple[float, str]] = {}  # key -> (round start, winner id)
        self._outcomes: dict[str, Outcome] = {}
        self._winners: deque[WakeWinner] = deque(maxlen=max_pending)
        self._last_losers: list[str] = []
        self._playing: dict[str, str] = {}  # device id -> earshot key

    # --- arbitration ---------------------------------------------------------------------

    def offer(
        self,
        device_id: str,
        room: str,
        group: str | None,
        score: float,
        energy: float,
        *,
        at: float | None = None,
    ) -> None:
        """A device heard the wake word. ``at`` = when its audio arrived (defaults to now)."""
        now = self._clock()
        at = now if at is None else at
        key = earshot_key(room, group)
        offer = _Offer(device_id, room, group, float(score), float(energy), at)
        with self._lock:
            self._close_expired(now)
            rnd = self._open.get(key)
            if rnd is not None and at >= rnd.start + self.window_s:
                self._decide(key, rnd)  # stale round (clock never polled past it)
                rnd = None
            if rnd is not None:
                self._outcomes.pop(device_id, None)
                prev = rnd.offers.get(device_id)
                if prev is None or (offer.score, offer.energy) > (prev.score, prev.energy):
                    rnd.offers[device_id] = offer
                return
            decided = self._decided.get(key)
            if decided is not None and decided[0] <= at < decided[0] + self.window_s:
                # Same wake, heard late: the round is over and somebody else answered.
                self._outcomes[device_id] = "won" if decided[1] == device_id else "lost"
                return
            self._outcomes.pop(device_id, None)
            self._open[key] = _Round(start=at, offers={device_id: offer})

    def poll(self) -> WakeWinner | None:
        """The next winner whose window has closed (one per round), else None."""
        with self._lock:
            self._close_expired(self._clock())
            if not self._winners:
                return None
            win = self._winners.popleft()
            self._last_losers = list(win.losers)
            return win

    def losers(self) -> list[str]:
        """Devices that lost the round of the winner most recently returned by ``poll``."""
        with self._lock:
            return list(self._last_losers)

    def outcome(self, device_id: str) -> Outcome | None:
        """``"won"``/``"lost"`` once the device's latest offer is decided, else None."""
        with self._lock:
            self._close_expired(self._clock())
            return self._outcomes.get(device_id)

    def _close_expired(self, now: float) -> None:
        for key, rnd in list(self._open.items()):
            if now >= rnd.start + self.window_s:
                self._decide(key, rnd)
        for key, (start, _) in list(self._decided.items()):
            if now >= start + 2 * self.window_s:
                del self._decided[key]

    def _decide(self, key: str, rnd: _Round) -> None:
        self._open.pop(key, None)
        offers = list(rnd.offers.values())
        best = max(o.score for o in offers)
        close = [o for o in offers if o.score >= best - self.score_tie]
        win = max(close, key=lambda o: (o.energy, o.score))
        losers = tuple(sorted(o.device_id for o in offers if o is not win))
        self._outcomes[win.device_id] = "won"
        for dev in losers:
            self._outcomes[dev] = "lost"
        self._decided[key] = (rnd.start, win.device_id)
        self._winners.append(WakeWinner(win.device_id, win.room, win.group, win.score, win.energy, win.at, losers))

    # --- earshot playback state (echo guard) ---------------------------------------------

    def set_playing(self, device_id: str, room: str | None, group: str | None, on: bool) -> None:
        with self._lock:
            if on:
                self._playing[device_id] = earshot_key(room, group)
            else:
                self._playing.pop(device_id, None)

    def group_playing(self, room: str | None, group: str | None) -> bool:
        """Is any device in this earshot group (the asking device included) playing?"""
        key = earshot_key(room, group)
        with self._lock:
            return key in self._playing.values()

    def forget(self, device_id: str) -> None:
        """The device went away: drop its playback flag and any pending outcome."""
        with self._lock:
            self._playing.pop(device_id, None)
            self._outcomes.pop(device_id, None)
