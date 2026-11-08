"""Control API: how agent tools, slash commands and the CLI reach the hub's services.

The hub lives in the gateway process (the adapter starts it). Tools and commands can run in any
Hermes process, so there are two ways in, both ending in ``run_op``:

- In the gateway process the adapter calls ``register_local_services(services, loop)`` and
  ``HousemateClient`` schedules ``run_op`` straight onto the hub's event loop.
- Everywhere else (``hermes housemate ...``, a CLI chat) ``HousemateClient`` posts to
  ``ControlServer``: ``POST http://127.0.0.1:<port>/v1/<op>`` with a bearer token read from a
  0600 file in the plugin data dir. The server binds the loopback interface only, refuses
  browser requests (any ``Origin`` header) and rotates the token every start.

Every op returns a JSON-able dict with ``ok`` and never raises: bad input becomes
``{"ok": false, "error": "..."}`` the model can read and act on.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import socket
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .store import slugify
from .timers import ACTIVE_STATES, parse_hhmm

if TYPE_CHECKING:
    from .announce import Announcer
    from .hub import Hub
    from .rooms import RoomRegistry
    from .store import HousemateStore
    from .timers import Timer, TimerService

__all__ = [
    "DEFAULT_PORT",
    "OPS",
    "PORT_FILE",
    "ControlPortInUse",
    "ControlServer",
    "bind_loopback",
    "HubServices",
    "local_services",
    "register_local_services",
    "run_op",
    "unregister_local_services",
]

log = logging.getLogger(__name__)

LOOPBACK = "127.0.0.1"
DEFAULT_PORT = 8743
PORT_FILE = "control.json"
MAX_BODY = 64 * 1024
MAX_TITLE = 200
MAX_TEXT = 16_000


@dataclass
class HubServices:
    """What the control API needs from a running hub. The adapter fills it in ``connect()``."""

    store: HousemateStore
    registry: RoomRegistry
    hub: Hub
    timers: TimerService
    announcer: Announcer
    urls: list[str] = field(default_factory=list)  # LAN URLs satellites open, best first
    fingerprint: str | None = None  # SHA-256 of the hub's TLS certificate


# --- in-process registration ------------------------------------------------------------------

_local_lock = threading.Lock()
_local: tuple[HubServices, asyncio.AbstractEventLoop] | None = None


def register_local_services(services: HubServices, loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Make ``services`` reachable in-process; ``loop`` (default: the running one) owns them."""
    global _local
    if loop is None:
        loop = asyncio.get_running_loop()
    with _local_lock:
        _local = (services, loop)


def unregister_local_services(services: HubServices | None = None) -> None:
    """Forget the in-process hub (only if it is still ``services``, when given)."""
    global _local
    with _local_lock:
        if _local is not None and (services is None or _local[0] is services):
            _local = None


def local_services() -> tuple[HubServices, asyncio.AbstractEventLoop] | None:
    with _local_lock:
        current = _local
    if current is None or current[1].is_closed():
        return None
    return current


# --- ops ----------------------------------------------------------------------------------------


def _need(services: HubServices, name: str) -> Any:
    value = getattr(services, name, None)
    if value is None:
        raise RuntimeError(f"the hub's {name} service is not running")
    return value


def _given(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _text(args: dict, key: str, *, required: bool = False, max_len: int = MAX_TEXT) -> str:
    value = args.get(key)
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{key} is required")
    return value[:max_len]


def _known_rooms(services: HubServices) -> dict[str, str]:
    """slug -> display name: rooms in the store plus any room a device is online in."""
    rooms = {r.slug: r.name for r in _need(services, "store").list_rooms()}
    for state in services.registry.snapshot():
        if state.room:
            rooms.setdefault(state.room, state.room)
    return dict(sorted(rooms.items()))


def _listed(known: dict[str, str]) -> str:
    return ", ".join(known) or "none yet"


def _resolve_room(services: HubServices, room: Any, session_room: Any = None, *, required: bool = True) -> str | None:
    """Explicit room, else the speaking room from the session, else the only room there is."""
    known = _known_rooms(services)
    if room is not None and not isinstance(room, str):
        raise ValueError("room must be a room name")
    if _given(room):
        try:
            slug = slugify(room)
        except ValueError:
            slug = None
        if slug not in known:
            raise ValueError(f"unknown room {room.strip()!r}; known rooms: {_listed(known)}")
        return slug
    if _given(session_room):
        try:
            slug = slugify(session_room)
        except ValueError:
            slug = None
        if slug in known:
            return slug
    if not required:
        return None
    if len(known) == 1:
        return next(iter(known))
    if not known:
        raise ValueError(
            "there are no rooms yet: open the hub address on a phone or laptop, pair it and pick its room"
        )
    raise ValueError(f"which room? known rooms: {_listed(known)}")


def _seconds(value: Any) -> float:
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
        value = float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("seconds must be a number of seconds, e.g. 600 for ten minutes")
    return value


def _timer_view(timer: Timer, tz) -> dict[str, Any]:
    due = datetime.fromtimestamp(timer.due_at, tz) if tz is not None else datetime.fromtimestamp(timer.due_at).astimezone()
    return {
        "id": timer.id,
        "room": timer.room,
        "kind": timer.kind,
        "label": timer.label,
        "state": timer.state,
        "remaining_s": timer.remaining_s,
        "due": due.isoformat(timespec="seconds"),
        "missed": timer.missed,
    }


def _timer_views(services: HubServices, timers: list) -> list[dict[str, Any]]:
    tz = getattr(services.timers, "tz", None)
    return [_timer_view(t, tz) for t in timers]


async def _op_rooms(s: HubServices, a: dict) -> dict:
    known = _known_rooms(s)
    live = {state.room: state for state in s.registry.snapshot() if state.room}
    groups = {r.slug: r.earshot_group for r in s.store.list_rooms()}
    by_room: dict[str, list] = {}
    for view in _timer_views(s, _need(s, "timers").list()):
        by_room.setdefault(view["room"], []).append(view)
    rooms = []
    for slug, name in known.items():
        state = live.get(slug)
        rooms.append({
            "room": slug,
            "name": name,
            "earshot_group": groups.get(slug),
            "online": state is not None,
            "state": state.state if state else "offline",
            "devices": [d.to_dict() for d in state.devices] if state else [],
            "timers": by_room.get(slug, []),
        })
    unassigned = []
    hub = s.hub
    if hub is not None:
        for device_id in hub.online_devices():
            if s.registry.get(device_id) is None:
                device = s.store.get_device(device_id)
                unassigned.append({"device_id": device_id, "name": device.name if device else device_id})
    hub_url = s.urls[0] if s.urls else (hub.url if hub is not None else None)
    return {"ok": True, "rooms": rooms, "unassigned_devices": unassigned, "hub_url": hub_url}


async def _op_devices(s: HubServices, a: dict) -> dict:
    hub = _need(s, "hub")
    online = set(hub.online_devices())
    devices = []
    for d in _need(s, "store").list_devices():
        conn = s.registry.get(d.device_id)
        devices.append({
            "device_id": d.device_id,
            "name": d.name,
            "room": d.room,
            "paired": d.paired,
            "online": d.device_id in online,
            "state": conn.state if conn is not None else ("idle" if d.device_id in online else "offline"),
        })
    return {"ok": True, "devices": devices}


async def _op_pair_approve(s: HubServices, a: dict) -> dict:
    raw = a.get("code")
    code = re.sub(r"\D", "", str(raw if raw is not None else ""))
    if len(code) != 6:
        raise ValueError("a pairing code has 6 digits (it is shown on the device's screen)")
    if not await _need(s, "hub").approve_pairing(code):
        return {
            "ok": False,
            "error": f"no device is waiting with code {code}; codes expire after 10 minutes, "
            "reload the page on the device for a new one",
        }
    return {"ok": True, "code": code}


async def _op_revoke(s: HubServices, a: dict) -> dict:
    device_id = _text(a, "device_id", required=True, max_len=128)
    if not await _need(s, "hub").revoke_device(device_id):
        known = ", ".join(d.device_id for d in s.store.list_devices()) or "none"
        return {"ok": False, "error": f"no device {device_id!r}; known devices: {known}"}
    return {"ok": True, "device_id": device_id}


async def _op_announce(s: HubServices, a: dict) -> dict:
    text = _text(a, "text", required=True)
    rooms = a.get("rooms", "all")
    if rooms is None or (isinstance(rooms, str) and not rooms.strip()):
        rooms = "all"
    if isinstance(rooms, str) and rooms.strip().lower() != "all":
        rooms = [rooms]
    if not isinstance(rooms, (str, list)) or (isinstance(rooms, list) and not all(isinstance(r, str) for r in rooms)):
        raise ValueError('rooms must be "all" or a list of room names')
    from_room = a.get("from_room") if _given(a.get("from_room")) else a.get("session_room")
    result = await _need(s, "announcer").announce(text, rooms, from_room if _given(from_room) else None)
    return result.to_dict()


async def _op_show(s: HubServices, a: dict) -> dict:
    title = _text(a, "title", required=True, max_len=MAX_TITLE)
    body = _text(a, "body", required=True)
    room = _resolve_room(s, a.get("room"), a.get("session_room"))
    if not await _need(s, "announcer").show(room, title, body):
        return {"ok": False, "room": room, "error": f"no device is online in {room}"}
    return {"ok": True, "room": room}


async def _op_timer_set(s: HubServices, a: dict) -> dict:
    seconds = _seconds(a.get("seconds"))
    label = _text(a, "label", max_len=64)
    room = _resolve_room(s, a.get("room"), a.get("session_room"))
    timer = _need(s, "timers").add(room, seconds=seconds, label=label)
    return {"ok": True, "timer": _timer_views(s, [timer])[0]}


async def _op_alarm_set(s: HubServices, a: dict) -> dict:
    when = _text(a, "time", required=True, max_len=32)
    try:
        parse_hhmm(when)
    except ValueError:
        raise ValueError(f"not a time of day: {when!r}; use 24-hour HH:MM, e.g. 07:30") from None
    label = _text(a, "label", max_len=64)
    room = _resolve_room(s, a.get("room"), a.get("session_room"))
    timer = _need(s, "timers").add_alarm(room, when, label)
    return {"ok": True, "timer": _timer_views(s, [timer])[0]}


async def _op_timer_list(s: HubServices, a: dict) -> dict:
    room = _resolve_room(s, a.get("room"), required=False)
    return {"ok": True, "room": room, "timers": _timer_views(s, _need(s, "timers").list(room))}


async def _op_timer_cancel(s: HubServices, a: dict) -> dict:
    timers = _need(s, "timers")
    label = _text(a, "label", max_len=64) or None
    timer_id = _text(a, "timer_id", max_len=64) or None
    room = _resolve_room(s, a.get("room"), required=False)
    if not (label or timer_id or room):
        room = _resolve_room(s, None, a.get("session_room"), required=False)
    if not (label or timer_id or room):
        raise ValueError("say which timer to cancel: a label, a room or a timer id")
    cancelled = timers.cancel(timer_id=timer_id, room=room, label=label)
    if not cancelled:
        return {
            "ok": False,
            "error": "no running timer or alarm matches that",
            "timers": _timer_views(s, timers.list()),
        }
    return {"ok": True, "cancelled": _timer_views(s, cancelled)}


async def _op_room_add(s: HubServices, a: dict) -> dict:
    room = _need(s, "store").ensure_room(_text(a, "name", required=True, max_len=64))
    return {"ok": True, "room": {"slug": room.slug, "name": room.name}}


async def _op_room_remove(s: HubServices, a: dict) -> dict:
    store = _need(s, "store")
    name = _text(a, "name", required=True, max_len=64)
    room = store.get_room(name)
    if room is None:
        raise ValueError(f"no room {name!r}; known rooms: {_listed(_known_rooms(s))}")
    online = s.registry.devices_in(room.slug)
    if online:
        names = ", ".join(c.name for c in online)
        raise ValueError(f"{room.name} still has a device online ({names}); move it to another room first")
    if store.list_timers(states=ACTIVE_STATES, room=room.slug):
        raise ValueError(f"{room.name} has timers running; cancel them first")
    store.remove_room(room.slug)
    return {"ok": True, "room": {"slug": room.slug, "name": room.name}}


async def _op_room_group(s: HubServices, a: dict) -> dict:
    store = _need(s, "store")
    name = _text(a, "room", required=True, max_len=64)
    room = store.get_room(name)
    if room is None:
        raise ValueError(f"no room {name!r}; known rooms: {_listed(_known_rooms(s))}")
    group = _text(a, "group", max_len=64) or None
    store.set_earshot_group(room.slug, group)
    return {"ok": True, "room": room.slug, "group": group}


async def _op_info(s: HubServices, a: dict) -> dict:
    hub = s.hub
    return {
        "ok": True,
        "hub_url": s.urls[0] if s.urls else (hub.url if hub is not None else None),
        "urls": list(s.urls),
        "fingerprint": s.fingerprint,
    }


_OPS: dict[str, Callable[[HubServices, dict], Awaitable[dict]]] = {
    "rooms": _op_rooms,
    "announce": _op_announce,
    "timer_set": _op_timer_set,
    "alarm_set": _op_alarm_set,
    "timer_list": _op_timer_list,
    "timer_cancel": _op_timer_cancel,
    "show": _op_show,
    "pair_approve": _op_pair_approve,
    "devices": _op_devices,
    "revoke": _op_revoke,
    "room_add": _op_room_add,
    "room_remove": _op_room_remove,
    "room_group": _op_room_group,
    "info": _op_info,
}
OPS: tuple[str, ...] = tuple(_OPS)


async def run_op(services: HubServices, op: str, args: Any = None) -> dict:
    """Run one op on the hub's event loop. Never raises (cancellation aside)."""
    handler = _OPS.get(op)
    if handler is None:
        return {"ok": False, "error": f"unknown op {op!r}; ops: {', '.join(OPS)}"}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {"ok": False, "error": "arguments must be a JSON object"}
    try:
        return await handler(services, args)
    except asyncio.CancelledError:
        raise
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:
        log.exception("housemate: control op %s failed", op)
        return {"ok": False, "error": f"{op} failed: {exc}"}


# --- loopback HTTP server ------------------------------------------------------------------------


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` readable by this user only (0600 where the OS has modes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, flags, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(text.encode("utf-8"))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    try:
        os.replace(tmp, path)
    except PermissionError:  # Windows: a reader has the old file open right now
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
        tmp.unlink(missing_ok=True)


class ControlPortInUse(OSError):
    """Something else already listens on the control port."""


def bind_loopback(port: int) -> socket.socket:
    """A listening TCP socket on 127.0.0.1:``port`` that nobody else can share.

    Windows lets a second process bind a port that is in use when it asks for SO_REUSEADDR, so
    the socket takes SO_EXCLUSIVEADDRUSE there; elsewhere SO_REUSEADDR only skips TIME_WAIT.
    A clash raises ``ControlPortInUse`` instead of quietly sharing or moving ports.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((LOOPBACK, port))
        except OSError as exc:
            raise ControlPortInUse(
                exc.errno,
                f"housemate control API: {LOOPBACK}:{port} is already in use ({exc.strerror or exc}); "
                "stop the other program or set the housemate control_port to a free port",
            ) from exc
        sock.listen(64)
        return sock
    except BaseException:
        sock.close()
        raise


class ControlServer:
    """``POST /v1/<op>`` on 127.0.0.1 with a bearer token. Start it on the hub's event loop.

    ``start()`` raises ``ControlPortInUse`` when the port is taken; the tools keep working
    in-process, only other processes (the CLI) lose their way in.
    """

    host = LOOPBACK

    def __init__(self, services: HubServices, token_path: Path, port: int = DEFAULT_PORT) -> None:
        self.services = services
        self.token_path = Path(token_path)
        self.port = port
        self._token: str | None = None
        self._runner = None
        self._sockets: list[socket.socket] = []

    @property
    def token(self) -> str | None:
        return self._token

    @property
    def url(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    @property
    def port_file(self) -> Path:
        return self.token_path.with_name(PORT_FILE)

    def bound_addresses(self) -> list[tuple[str, int]]:
        return [tuple(s.getsockname()[:2]) for s in self._sockets]

    async def start(self) -> None:
        from aiohttp import web

        self._token = secrets.token_urlsafe(32)
        app = web.Application(client_max_size=MAX_BODY)
        app.router.add_post("/v1/{op}", self._handle)
        sock = bind_loopback(self.port)
        self._runner = web.AppRunner(app, access_log=None)
        try:
            await self._runner.setup()
            site = web.SockSite(self._runner, sock)
            await site.start()
        except BaseException:
            await self._runner.cleanup()
            self._runner = None
            sock.close()
            raise
        self._sockets = [sock]
        self.port = sock.getsockname()[1]
        # Only after the bind succeeded: a second gateway must not rotate a live hub's token.
        write_private(self.token_path, self._token + "\n")
        write_private(self.port_file, json.dumps({"port": self.port, "pid": os.getpid()}))
        log.info("housemate control API on %s", self.url)

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._sockets = []
        try:
            info = json.loads(self.port_file.read_text(encoding="utf-8"))
            if info.get("pid") == os.getpid() and info.get("port") == self.port:
                self.port_file.unlink()
        except (OSError, ValueError, AttributeError):
            pass

    def _authorized(self, header: str) -> bool:
        scheme, _, value = (header or "").partition(" ")
        if scheme.lower() != "bearer" or not self._token:
            return False
        return hmac.compare_digest(value.strip().encode("utf-8"), self._token.encode("utf-8"))

    async def _handle(self, request):
        from aiohttp import web

        if request.headers.get("Origin") is not None:
            return web.json_response({"ok": False, "error": "browser requests are not accepted"}, status=403)
        if not self._authorized(request.headers.get("Authorization", "")):
            return web.json_response({"ok": False, "error": "missing or wrong control token"}, status=401)
        op = request.match_info["op"]
        if op not in _OPS:
            return web.json_response({"ok": False, "error": f"unknown op {op!r}"}, status=404)
        raw = await request.read()
        try:
            args = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return web.json_response({"ok": False, "error": "body must be JSON"}, status=400)
        result = await run_op(self.services, op, args)
        return web.json_response(result)
