"""What the Desktop Rooms page shows: one snapshot of the home, the hub's QR code, the change feed.

``desk_state(services)`` runs on the hub's event loop (it is the ``desk_state`` control op) and
returns everything the page draws: rooms with their earshot group, every paired device with its
live state, timers with their due time, devices waiting for pairing approval, the last exchange
per room, recent wake-arbitration rounds and the conversation mode.

``DeskLog`` is what the adapter records for the page as things happen (captions, arbitration).
``DeskFeed`` coalesces changes and pushes the new snapshot to Desktop clients as the plugin event
``plugin.housemate.rooms.changed`` (Hermes's public ``broadcast_plugin_event``); a page that
receives nothing polls instead.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .arbiter import WakeWinner
    from .control import HubServices

__all__ = ["CONVERSATION_MODES", "DeskFeed", "DeskLog", "EVENT", "broadcast", "desk_state", "persist_mode", "qr_svg"]

log = logging.getLogger(__name__)

PLUGIN_ID = "housemate"
EVENT = "rooms.changed"  # wire name: plugin.housemate.rooms.changed
CONVERSATION_MODES = ("household", "per_room")
_ROOM_PRIORITY = {"speaking": 5, "ringing": 4, "thinking": 3, "listening": 2, "idle": 1, "offline": 0}
EXCHANGE_CHARS = 280
WAKE_KEEP = 8


# --- the QR code ---------------------------------------------------------------------------------

_qr_cache: dict[str, dict | None] = {}


def qr_svg(text: str) -> dict | None:
    """``{"size": n, "path": "M2 2h7v1h-7z…"}``: the QR code of ``text`` as one SVG path in module
    units (quiet zone included), runs of dark modules merged. None without the qrcode package."""
    if text in _qr_cache:
        return _qr_cache[text]
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M
    except ImportError:
        return None
    qr = qrcode.QRCode(border=2, error_correction=ERROR_CORRECT_M)
    qr.add_data(text)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    parts: list[str] = []
    for y, row in enumerate(matrix):
        x = 0
        while x < len(row):
            if not row[x]:
                x += 1
                continue
            start = x
            while x < len(row) and row[x]:
                x += 1
            parts.append(f"M{start} {y}h{x - start}v1h-{x - start}z")
    result = {"size": len(matrix), "path": "".join(parts), "text": text}
    if len(_qr_cache) > 16:
        _qr_cache.clear()
    _qr_cache[text] = result
    return result


# --- what the adapter records --------------------------------------------------------------------


class DeskLog:
    """The last exchange per room and the recent arbitration rounds. Thread-safe."""

    def __init__(self, clock: Callable[[], float] = time.time, on_change: Callable[[], None] | None = None) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._exchanges: dict[str, dict[str, Any]] = {}
        self._wakes: deque[dict[str, Any]] = deque(maxlen=WAKE_KEEP)
        self._seq = 0
        self.on_change = on_change
        self.info: dict[str, Any] = {}  # hub facts the page lists (wake word)

    def caption(self, room: str | None, role: str, text: str) -> None:
        """A caption went to ``room``: user / assistant / announcement make the room's last exchange."""
        if not room or not text or role not in ("user", "assistant", "announcement"):
            return
        text = " ".join(str(text).split())[:EXCHANGE_CHARS]
        now = self._clock()
        with self._lock:
            current = self._exchanges.get(room)
            if role == "user":
                self._exchanges[room] = {"you": text, "say": None, "from": None, "at": now}
            elif role == "assistant":
                if current is not None and current.get("say") is None and current.get("you") and now - current["at"] < 300:
                    current["say"], current["at"] = text, now
                else:
                    self._exchanges[room] = {"you": None, "say": text, "from": None, "at": now}
            else:
                self._exchanges[room] = {"you": None, "say": text, "from": "announcement", "at": now}
        self._changed()

    def wake(self, winner: WakeWinner) -> None:
        """One decided arbitration round (called on a listener thread)."""
        with self._lock:
            self._seq += 1
            self._wakes.append({
                "id": self._seq,
                "at": self._clock(),
                "room": winner.room,
                "group": winner.group,
                "winner": winner.device_id,
                "deferred": list(winner.losers),
                "score": round(float(winner.score), 3),
            })
        self._changed()

    def exchanges(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {room: dict(x) for room, x in self._exchanges.items()}

    def wakes(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(w) for w in self._wakes]

    def _changed(self) -> None:
        cb = self.on_change
        if cb is not None:
            try:
                cb()
            except Exception:
                log.debug("housemate: desk change callback failed", exc_info=True)


# --- the snapshot --------------------------------------------------------------------------------


def _timer_item(timer: Any) -> dict[str, Any]:
    return {
        "id": timer.id,
        "room": timer.room,
        "kind": timer.kind,
        "label": timer.label,
        "state": timer.state,
        "due_at": timer.due_at,
        "remaining_s": timer.remaining_s,
        "missed": timer.missed,
    }


def desk_state(s: HubServices, *, include_qr: bool = False) -> dict[str, Any]:
    """Everything the Rooms page draws. Runs on the hub's loop (store and registry reads only)."""
    store, registry, hub = s.store, s.registry, s.hub
    online = set(hub.online_devices()) if hub is not None else set()
    devices = []
    for d in store.list_devices():
        if not d.paired:
            continue
        conn = registry.get(d.device_id)
        room = conn.room if conn is not None and conn.room else d.room
        if d.device_id not in online:
            state = "offline"
        else:
            state = conn.state if conn is not None else "idle"
        devices.append({"device_id": d.device_id, "name": d.name, "room": room, "online": d.device_id in online,
                        "state": state})
    devices.sort(key=lambda d: (d["room"] or "~", d["name"].lower(), d["device_id"]))

    rooms = []
    known = {r.slug: r for r in store.list_rooms()}
    for d in devices:  # a room only the registry knows (should not happen; never hide a device)
        if d["room"] and d["room"] not in known:
            known[d["room"]] = None
    for slug in sorted(known):
        room = known[slug]
        members = [d for d in devices if d["room"] == slug]
        live = [d["state"] for d in members if d["online"]]
        rooms.append({
            "room": slug,
            "name": room.name if room is not None else slug.replace("-", " ").title(),
            "earshot_group": room.earshot_group if room is not None else None,
            "online": bool(live),
            "state": max(live, key=lambda st: _ROOM_PRIORITY.get(st, 0)) if live else "offline",
            "devices": [d["device_id"] for d in members],
        })

    timers = [_timer_item(t) for t in s.timers.list()] if s.timers is not None else []
    desk = s.desk
    mode = s.mode() if callable(s.mode) else None
    hub_url = s.urls[0] if s.urls else (hub.url if hub is not None else None)
    state: dict[str, Any] = {
        "ok": True,
        "now": time.time(),
        "rooms": rooms,
        "devices": devices,
        "timers": timers,
        "pending": hub.pending_pairings() if hub is not None and hasattr(hub, "pending_pairings") else [],
        "exchanges": desk.exchanges() if desk is not None else {},
        "wakes": desk.wakes() if desk is not None else [],
        "mode": mode,
        "hub": {"url": hub_url, "urls": list(s.urls), "fingerprint": s.fingerprint,
                **(dict(desk.info) if desk is not None else {})},
    }
    if include_qr and hub_url:
        state["qr"] = qr_svg(hub_url)
    return state


# --- the conversation mode -----------------------------------------------------------------------


def persist_mode(mode: str) -> str | None:
    """Save ``platforms.housemate.conversation`` in the profile's config.yaml (Hermes's public
    ``write_platform_config_field``). None when saved, else why not."""
    try:
        from hermes_cli.config import write_platform_config_field
    except ImportError as exc:
        return f"Hermes config is not writable from here ({exc})"
    try:
        try:
            write_platform_config_field(PLUGIN_ID, "conversation", mode, raw=True)
        except TypeError:  # a Hermes without the raw switch
            write_platform_config_field(PLUGIN_ID, "conversation", mode)
    except Exception as exc:
        log.warning("housemate: saving the conversation mode failed", exc_info=True)
        return str(exc) or type(exc).__name__
    return None


# --- the change feed ------------------------------------------------------------------------------


def broadcast(payload: dict[str, Any]) -> None:
    """Push ``plugin.housemate.rooms.changed`` to the Desktop clients of this process."""
    from hermes_cli.plugin_events import broadcast_plugin_event

    broadcast_plugin_event(PLUGIN_ID, EVENT, payload)


class DeskFeed:
    """Coalesces change pokes (from any thread) into one pushed snapshot per ``delay_s``."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        build: Callable[[], dict[str, Any]],
        emit: Callable[[dict[str, Any]], None] = broadcast,
        *,
        delay_s: float = 0.12,
    ) -> None:
        self._loop = loop
        self._build = build
        self._emit = emit
        self.delay_s = delay_s
        self.rev = 0
        self._pending = False
        self._handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._failures = 0

    def poke(self) -> None:
        """Something the page shows changed. Safe from any thread; never raises."""
        if self._closed:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._schedule()
            return
        try:
            self._loop.call_soon_threadsafe(self._schedule)
        except RuntimeError:  # loop closed: the gateway is shutting down
            pass

    def _schedule(self) -> None:
        if self._pending or self._closed:
            return
        self._pending = True
        self._handle = self._loop.call_later(self.delay_s, self._flush)

    def _flush(self) -> None:
        self._pending = False
        self._handle = None
        if self._closed:
            return
        self.rev += 1
        try:
            state = self._build()
            self._emit({"rev": self.rev, "state": state})
            self._failures = 0
        except Exception:
            self._failures += 1
            if self._failures in (1, 10, 100):
                log.warning("housemate: pushing the rooms state to Desktop failed", exc_info=True)

    def close(self) -> None:
        self._closed = True
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
