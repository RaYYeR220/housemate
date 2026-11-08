"""Deterministic timers and alarms: no LLM anywhere in the ring path.

Timers live in the store's ``timers`` table (``due_at`` is UTC epoch seconds) and move through
``pending -> ringing -> done | stopped | cancelled`` (or straight ``pending -> cancelled``).
The ``pending -> ringing`` step is a compare-and-set in SQLite that happens *before* anyone is
told, so a timer rings at most once, also across restarts and a crash mid-ring:
``restore()`` rings overdue pending timers once (``missed=True``) and retires rings left over
from a previous process without ringing them again.

While a timer rings, ``RingController`` plays the chime and speaks the label in its room every
20 s until someone stops it or 3 minutes pass. Time comes only from the injected ``clock``
(epoch seconds) and ``sleep``, so tests run on a fake clock. No Hermes imports: the adapter
wires the callbacks (``ring_cb``, ``play_chime``, ``speak``, ``broadcast_timers``).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, tzinfo
from datetime import time as dtime
from typing import Any, Awaitable, Callable, Iterable

from .store import HousemateStore, TimerRow, slugify

__all__ = [
    "ACTIVE_STATES",
    "RING_EVERY_S",
    "RING_MAX_S",
    "RingController",
    "Timer",
    "TimerService",
    "next_occurrence",
    "parse_hhmm",
    "ring_text",
]

log = logging.getLogger(__name__)

ACTIVE_STATES = ("pending", "ringing")
KINDS = ("timer", "alarm")
RING_EVERY_S = 20.0  # chime + spoken label this often while ringing
RING_MAX_S = 180.0  # then it stops by itself
MISSED_AFTER_S = 60.0  # a tick this late (PC asleep, loop blocked) counts as missed
MAX_TIMER_S = 7 * 24 * 3600
MAX_LABEL = 64

Clock = Callable[[], float]


@dataclass(frozen=True)
class Timer:
    id: str
    room: str
    kind: str  # "timer" | "alarm"
    label: str
    due_at: float  # UTC epoch seconds
    created_at: float
    duration_s: float | None  # None for alarms set by time of day
    state: str  # pending | ringing | done | stopped | cancelled
    fired_at: float | None = None
    remaining_s: int = 0  # whole seconds left (rounded up), 0 once due
    missed: bool = False  # rang late because the hub was offline / asleep

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- spoken text and time of day ---------------------------------------------------------------


def _human_duration(seconds: float) -> str:
    s = int(round(seconds))
    hours, rest = divmod(s, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour")
    if minutes:
        parts.append(f"{minutes} minute")
    if secs or not parts:
        parts.append(f"{secs} second")
    return " ".join(parts)


def _spoken_clock(epoch: float, tz: tzinfo | None) -> str:
    dt = datetime.fromtimestamp(epoch, tz) if tz is not None else datetime.fromtimestamp(epoch)
    return f"{dt.hour}:{dt.minute:02d}"


def ring_text(timer: Timer, tz: tzinfo | None = None) -> str:
    """What gets said while it rings: "Your pasta timer is done." """
    label = " ".join(timer.label.split())
    if timer.kind == "alarm":
        when = _spoken_clock(timer.due_at, tz)
        if not label:
            name = "alarm"
        else:
            name = label if label.lower().endswith("alarm") else f"{label} alarm"
        if timer.missed:
            return f"Your {when} {name} went off while I was offline."
        return f"It's {when}. Your {name}."
    if label:
        name = label if label.lower().endswith("timer") else f"{label} timer"
    elif timer.duration_s:
        name = f"{_human_duration(timer.duration_s)} timer"
    else:
        name = "timer"
    if timer.missed:
        return f"Your {name} finished while I was offline."
    return f"Your {name} is done."


_HHMM = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*(?:([ap])\.?\s*m\.?)?\s*$", re.IGNORECASE)


def parse_hhmm(text: str) -> tuple[int, int]:
    """ "07:30", "7:30", "19:05", "7:05 pm" -> (hour, minute). Raises ValueError."""
    m = _HHMM.match(str(text or ""))
    if not m:
        raise ValueError(f"not a time of day: {text!r} (use HH:MM)")
    hour, minute, ampm = int(m[1]), int(m[2]), m[3]
    if minute > 59:
        raise ValueError(f"not a time of day: {text!r}")
    if ampm:
        if not 1 <= hour <= 12:
            raise ValueError(f"not a time of day: {text!r}")
        hour = hour % 12 + (12 if ampm.lower() == "p" else 0)
    elif hour > 23:
        raise ValueError(f"not a time of day: {text!r}")
    return hour, minute


def next_occurrence(hhmm: str, now: float, tz: tzinfo | None = None) -> datetime:
    """The next local ``hhmm`` strictly after ``now`` (epoch): today, else tomorrow.

    ``tz=None`` uses the host's local time zone (DST handled by the OS). The result is aware.
    """
    hour, minute = parse_hhmm(hhmm)
    today = (datetime.fromtimestamp(now, tz) if tz is not None else datetime.fromtimestamp(now)).date()

    def at(day) -> datetime:
        local = datetime.combine(day, dtime(hour, minute), tzinfo=tz)
        return local if tz is not None else local.astimezone()

    candidate = at(today)
    if candidate.timestamp() <= now:
        candidate = at(today + timedelta(days=1))
    return candidate


def _to_epoch(at: datetime, tz: tzinfo | None) -> float:
    if at.tzinfo is None and tz is not None:
        at = at.replace(tzinfo=tz)
    return at.timestamp()  # a naive datetime is read as host local time


def _norm_label(text: str) -> str:
    s = " ".join(str(text).lower().split())
    s = re.sub(r"^(the|my|our)\s+", "", s)
    return re.sub(r"\s+(timer|alarm)$", "", s)


# --- background callbacks ------------------------------------------------------------------------


async def _call_logged(fn: Callable[..., Any], args: tuple, what: str) -> None:
    try:
        result = fn(*args)
        if inspect.isawaitable(result):
            await result
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("housemate: %s failed", what)


class _Background:
    """Fire-and-forget callbacks that never block the tick and never raise into it."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._deferred: list[tuple[Callable[..., Any], tuple, str]] = []

    def spawn(self, fn: Callable[..., Any] | None, *args: Any, what: str, defer: bool = True) -> asyncio.Task | None:
        if fn is None:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # called outside the loop (e.g. restore() before run())
            if defer:
                self._deferred.append((fn, args, what))
            return None
        task = loop.create_task(_call_logged(fn, args, what))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def flush(self) -> None:
        deferred, self._deferred = self._deferred, []
        for fn, args, what in deferred:
            self.spawn(fn, *args, what=what)

    async def drain(self) -> None:
        while True:
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    def cancel(self) -> None:
        self._deferred.clear()
        for task in list(self._tasks):
            task.cancel()


# --- ringing -----------------------------------------------------------------------------------------


@dataclass(eq=False)
class _Ring:
    timers: dict[str, tuple[Timer, float]]  # id -> (timer, ring started at)
    next_cycle_at: float
    task: asyncio.Task | None = None


class RingController:
    """Chime + spoken label in a ringing room every ``every_s``, auto-stop after ``max_s``.

    Driven by ``tick()`` (the TimerService calls it every second). One ring per room: two
    timers ringing in the same room are spoken together. ``play_chime(room)`` and
    ``speak(room, text)`` run as background tasks; an explicit ``stop`` cancels the one in
    flight, so ``speak`` should abort its playback on ``CancelledError``.
    """

    def __init__(
        self,
        clock: Clock,
        *,
        play_chime: Callable[[str], Awaitable[Any]] | None = None,
        speak: Callable[[str, str], Awaitable[Any]] | None = None,
        text_for: Callable[[Timer], str] = ring_text,
        every_s: float = RING_EVERY_S,
        max_s: float = RING_MAX_S,
    ) -> None:
        self._clock = clock
        self._play_chime = play_chime
        self._speak = speak
        self._text_for = text_for
        self.every_s = every_s
        self.max_s = max_s
        self._rings: dict[str, _Ring] = {}
        self._bg = _Background()

    def start(self, *timers: Timer, now: float | None = None) -> None:
        """Start ringing; timers started together in one room are spoken in one cycle."""
        now = self._clock() if now is None else now
        rooms: dict[str, _Ring] = {}
        for timer in timers:
            ring = self._rings.get(timer.room)
            if ring is None:
                ring = self._rings[timer.room] = _Ring(timers={}, next_cycle_at=now)
            ring.timers[timer.id] = (timer, now)
            ring.next_cycle_at = now  # say the newcomer now, or right after the cycle in flight
            rooms[timer.room] = ring
        for room, ring in rooms.items():
            self._cycle_if_due(room, ring, now)

    def tick(self, now: float | None = None) -> list[Timer]:
        """Advance every ring; returns the timers that just auto-stopped."""
        now = self._clock() if now is None else now
        expired: list[Timer] = []
        for room, ring in list(self._rings.items()):
            for timer_id, (timer, started) in list(ring.timers.items()):
                if now - started >= self.max_s:
                    del ring.timers[timer_id]
                    expired.append(timer)
            if not ring.timers:
                del self._rings[room]  # a cycle still playing finishes by itself
                continue
            self._cycle_if_due(room, ring, now)
        return expired

    def stop(self, room: str) -> list[Timer]:
        """Silence a room now; returns the timers that were ringing there."""
        ring = self._rings.pop(room, None)
        if ring is None:
            return []
        if ring.task is not None and not ring.task.done():
            ring.task.cancel()
        return [timer for timer, _ in ring.timers.values()]

    def stop_timer(self, timer_id: str) -> Timer | None:
        for room, ring in list(self._rings.items()):
            entry = ring.timers.pop(timer_id, None)
            if entry is not None:
                if not ring.timers:
                    self.stop(room)
                return entry[0]
        return None

    def find(self, timer_id: str) -> Timer | None:
        for ring in self._rings.values():
            entry = ring.timers.get(timer_id)
            if entry is not None:
                return entry[0]
        return None

    def ringing(self, room: str | None = None) -> list[Timer]:
        if room is None:
            rings = list(self._rings.values())
        else:
            rings = [self._rings[room]] if room in self._rings else []
        return [timer for ring in rings for timer, _ in ring.timers.values()]

    def is_ringing(self, room: str) -> bool:
        return room in self._rings

    async def drain(self) -> None:
        await self._bg.drain()

    def close(self) -> None:
        self._rings.clear()
        self._bg.cancel()

    def _cycle_if_due(self, room: str, ring: _Ring, now: float) -> None:
        if now < ring.next_cycle_at:
            return
        if ring.task is not None and not ring.task.done():
            return  # still speaking the previous cycle; go again when it is over
        text = " ".join(self._text_for(timer) for timer, _ in ring.timers.values())
        task = self._bg.spawn(self._cycle, room, text, what="timer ring", defer=False)
        if task is None:
            return  # no event loop yet: the first tick plays it
        ring.task = task
        ring.next_cycle_at = now + self.every_s

    async def _cycle(self, room: str, text: str) -> None:
        if self._play_chime is not None:
            await _call_logged(self._play_chime, (room,), "timer chime")
        if self._speak is not None:
            await _call_logged(self._speak, (room, text), "timer speech")


# --- the service ------------------------------------------------------------------------------------


class TimerService:
    """Timers and alarms for every room. Loop-affine: call it from the hub's event loop
    (code in another thread hops over with ``asyncio.run_coroutine_threadsafe``).

    ``ring_cb(timer)`` runs exactly once per timer when it starts ringing. ``play_chime(room)``
    and ``speak(room, text)`` drive the ring (see ``RingController``). ``broadcast_timers(room,
    items)`` gets the room's pending + ringing timers (``Timer.to_dict()``) after every change.
    """

    def __init__(
        self,
        store: HousemateStore,
        clock: Clock = time.time,
        ring_cb: Callable[[Timer], Awaitable[None]] | None = None,
        *,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        tz: tzinfo | None = None,
        play_chime: Callable[[str], Awaitable[Any]] | None = None,
        speak: Callable[[str, str], Awaitable[Any]] | None = None,
        broadcast_timers: Callable[[str, list[dict]], Awaitable[Any]] | None = None,
        ring_every_s: float = RING_EVERY_S,
        ring_max_s: float = RING_MAX_S,
        tick_s: float = 1.0,
    ) -> None:
        self._store = store
        self._clock = clock
        self._ring_cb = ring_cb
        self._sleep = sleep
        self.tz = tz
        self._broadcast_cb = broadcast_timers
        self.tick_s = tick_s
        self._bg = _Background()
        self._restored = False
        self.ringer = RingController(
            clock,
            play_chime=play_chime,
            speak=speak,
            text_for=lambda t: ring_text(t, tz),
            every_s=ring_every_s,
            max_s=ring_max_s,
        )

    # --- commands ----------------------------------------------------------------------------

    def add(
        self,
        room: str,
        seconds: float | None = None,
        at: datetime | None = None,
        label: str = "",
        kind: str = "timer",
    ) -> Timer:
        return self._add(room, seconds, at, label, kind, self._clock())

    def add_alarm(self, room: str, hhmm: str, label: str = "") -> Timer:
        """An alarm at the next local ``HH:MM`` (today if still ahead, else tomorrow)."""
        now = self._clock()
        return self._add(room, None, next_occurrence(hhmm, now, self.tz), label, "alarm", now)

    def get(self, timer_id: str) -> Timer | None:
        row = self._store.get_timer(timer_id)
        return self._timer(row, self._clock()) if row else None

    def list(self, room: str | None = None) -> list[Timer]:
        """Pending and ringing timers, soonest first, with ``remaining_s`` filled in."""
        now = self._clock()
        slug = slugify(room) if room is not None else None
        return [self._timer(r, now) for r in self._store.list_timers(states=ACTIVE_STATES, room=slug)]

    def cancel(
        self, timer_id: str | None = None, room: str | None = None, label: str | None = None
    ) -> list[Timer]:
        """Cancel by id, by label (case-insensitive; within ``room`` if given) or everything
        in ``room``. Ringing timers it matches stop ringing. Returns what was cancelled."""
        if not timer_id and room is None and not label:
            raise ValueError("say which timer to cancel: an id, a label or a room")
        slug = slugify(room) if room is not None else None
        want = _norm_label(label) if label else None
        now = self._clock()
        cancelled: list[Timer] = []
        for row in self._store.list_timers(states=ACTIVE_STATES, room=slug):
            if timer_id and row.id != timer_id:
                continue
            if want is not None and _norm_label(row.label) != want:
                continue
            if self._store.transition_timer(row.id, ACTIVE_STATES, "cancelled"):
                if row.state == "ringing":
                    self.ringer.stop_timer(row.id)
                cancelled.append(self._timer(replace(row, state="cancelled"), now))
        self._broadcast({t.room for t in cancelled})
        return cancelled

    def stop_ringing(self, room: str) -> bool:
        """Silence whatever rings in ``room`` (tap, spoken "stop"). False if nothing rang."""
        try:
            slug = slugify(room)
        except ValueError:
            return False
        stopped = self.ringer.stop(slug)
        for timer in stopped:
            self._store.transition_timer(timer.id, ("ringing",), "stopped")
        if stopped:
            self._broadcast({slug})
        return bool(stopped)

    def is_ringing(self, room: str) -> bool:
        try:
            return self.ringer.is_ringing(slugify(room))
        except ValueError:
            return False

    # --- lifecycle ---------------------------------------------------------------------------

    def restore(self, now: float) -> list[Timer]:
        """After a (re)start: ring overdue pending timers once with ``missed=True``, and retire
        rings a previous process left behind (they already rang; never ring them twice)."""
        changed: set[str] = set()
        for row in self._store.list_timers(states=("ringing",)):
            if self.ringer.find(row.id) is None and self._store.transition_timer(row.id, ("ringing",), "done"):
                changed.add(row.room)
        rung = self._fire_due(now, missed_after=-math.inf)
        changed.update(t.room for t in rung)
        self._restored = True
        self._broadcast(changed)
        return rung

    async def tick(self) -> list[Timer]:
        """One scheduler step: ring what is due, advance rings. Returns timers that started."""
        self._bg.flush()
        now = self._clock()
        fired = self._fire_due(now, missed_after=MISSED_AFTER_S)
        changed = {t.room for t in fired}
        for timer in self.ringer.tick(now):
            if self._store.transition_timer(timer.id, ("ringing",), "done"):
                changed.add(timer.room)
        self._broadcast(changed)
        return fired

    async def run(self) -> None:
        """Restore once, then tick on every whole ``tick_s`` until cancelled."""
        if not self._restored:
            try:
                self.restore(self._clock())
            except Exception:
                log.exception("housemate: restoring timers failed")
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("housemate: timer tick failed")
            now = self._clock()
            await self._sleep(self.tick_s - (now % self.tick_s))

    async def drain(self) -> None:
        """Wait for callbacks in flight (tests, orderly shutdown)."""
        await self._bg.drain()
        await self.ringer.drain()

    def close(self) -> None:
        self._bg.cancel()
        self.ringer.close()

    # --- internals ---------------------------------------------------------------------------

    def _add(self, room: str, seconds: float | None, at: datetime | None, label: str, kind: str, now: float) -> Timer:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        slug = slugify(room)
        if (seconds is None) == (at is None):
            raise ValueError("give either seconds or at")
        if seconds is not None:
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
                raise ValueError(f"seconds must be a number, not {seconds!r}")
            if not 0 < seconds <= MAX_TIMER_S:
                raise ValueError(f"a timer runs for 1 second up to {MAX_TIMER_S // 86400} days")
            due_at, duration = now + float(seconds), float(seconds)
        else:
            due_at, duration = _to_epoch(at, self.tz), None
            if due_at <= now:
                raise ValueError("that time has already passed")
            if due_at - now > MAX_TIMER_S:
                raise ValueError(f"that is more than {MAX_TIMER_S // 86400} days away")
        row = TimerRow(
            id=uuid.uuid4().hex[:12],
            room=slug,
            kind=kind,
            label=" ".join(str(label or "").split())[:MAX_LABEL],
            due_at=due_at,
            created_at=now,
            duration_s=duration,
            state="pending",
        )
        self._store.add_timer(row)
        self._broadcast({slug})
        return self._timer(row, now)

    def _fire_due(self, now: float, *, missed_after: float) -> list[Timer]:
        """Ring every pending timer due by ``now``, exactly once each."""
        fired: list[Timer] = []
        for row in self._store.list_timers(states=("pending",), due_before=now):
            # Persist first: if the process dies after this line the timer never rings again.
            if not self._store.transition_timer(row.id, ("pending",), "ringing", fired_at=now):
                continue
            missed = (now - row.due_at) > missed_after
            fired.append(self._timer(replace(row, state="ringing", fired_at=now), now, missed=missed))
        for timer in fired:
            self._bg.spawn(self._ring_cb, timer, what="timer ring callback")
        if fired:
            self.ringer.start(*fired, now=now)
        return fired

    def _timer(self, row: TimerRow, now: float, *, missed: bool | None = None) -> Timer:
        if missed is None:
            ringing = self.ringer.find(row.id) if row.state == "ringing" else None
            missed = bool(ringing and ringing.missed)
        remaining = max(0, math.ceil(row.due_at - now - 1e-6)) if row.state == "pending" else 0
        return Timer(
            id=row.id,
            room=row.room,
            kind=row.kind,
            label=row.label,
            due_at=row.due_at,
            created_at=row.created_at,
            duration_s=row.duration_s,
            state=row.state,
            fired_at=row.fired_at,
            remaining_s=remaining,
            missed=missed,
        )

    def _broadcast(self, rooms: Iterable[str]) -> None:
        if self._broadcast_cb is None:
            return
        for room in sorted(set(rooms)):
            items = [t.to_dict() for t in self.list(room)]
            self._bg.spawn(self._broadcast_cb, room, items, what="timers broadcast")
