"""LAN hub: serves the satellite web app and speaks the satellite WebSocket protocol.

Pure aiohttp, no Hermes imports. The adapter owns one ``Hub`` and plugs in through
``on_audio`` / ``on_control`` (inbound) and ``send_control`` / ``send_pcm`` (outbound).

Trust model: a satellite says ``hello``. With a valid device key it is trusted; with a
wrong key the socket is closed with 4401; without a key it only gets a 6-digit pairing
code and nothing it sends is accepted until the owner approves that code out of band
(``Hub.approve_pairing``: Desktop Rooms page or ``hermes housemate pair``). A denied device
(``Hub.deny_pairing``) is closed with 4403 and gets no new code for ``deny_ttl_s``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from aiohttp import WSMsgType, web

from .protocol import FRAME_BYTES, MAX_MESSAGE_BYTES, MIC_PCM, PLAY_PCM, SATELLITE_TYPES, ProtocolError, decode, encode
from .rooms import DEVICE_STATES, DeviceConn, RoomRegistry
from .store import HousemateStore

__all__ = ["Hub", "HubPortInUse", "bind_exclusive", "server_ssl_context"]

log = logging.getLogger(__name__)

# Close codes the satellite acts on.
CLOSE_BAD_KEY = 4401  # forget the stored key and pair again
CLOSE_DENIED = 4403  # the owner denied this pairing request
CLOSE_TIMEOUT = 4408  # hello never came, or the pairing code expired
CLOSE_REPLACED = 4409  # the same device connected again
CLOSE_RATE_LIMITED = 4429
CLOSE_ABUSE = 4400

PLAYOUT_CHUNK = 32_000  # bytes per 0x02 frame, well under the 64 KB message cap
MAX_REFUSALS = 100

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".webmanifest": "application/manifest+json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".wav": "audio/wav",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}


def server_ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    """A TLS *server* context from the standard library's own ``SSLContext``.

    Hermes swaps ``ssl.SSLContext`` for truststore's client-side class at start-up (system trust
    store for outgoing HTTPS). That class verifies the *peer's* certificate after every handshake,
    so as a server it drops each satellite right after the request. Walk past it to the stdlib class
    (and leave its Python-level property setters alone: with the class swapped they recurse; the
    server protocol already defaults to TLS 1.2+).
    """
    cls = next((c for c in ssl.SSLContext.__mro__ if c.__module__ == "ssl" and c.__name__ == "SSLContext"),
               ssl.SSLContext)
    ctx = cls(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert), str(key))
    return ctx


class HubPortInUse(OSError):
    """Another program already listens on the hub's port."""


def _loopback_answers(port: int, family: int) -> bool:
    host = "::1" if family == socket.AF_INET6 else "127.0.0.1"
    probe = socket.socket(family, socket.SOCK_STREAM)
    probe.settimeout(0.3)
    try:
        return probe.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        probe.close()


def bind_exclusive(host: str, port: int) -> socket.socket:
    """A listening socket on ``host:port`` that no other program shares.

    Windows lets a wildcard bind succeed while another program holds ``127.0.0.1:<port>`` (a
    wrangler dev server on 8787 did exactly that): LAN phones would reach the hub but the PC's
    own browser would reach the other program. So the socket takes SO_EXCLUSIVEADDRUSE there, a
    wildcard bind first checks that nothing answers on loopback, and any clash raises
    ``HubPortInUse`` instead of quietly sharing the port.
    """
    family = socket.AF_INET6 if ":" in (host or "") else socket.AF_INET
    if port and host in ("", "0.0.0.0", "::") and _loopback_answers(port, family):
        raise HubPortInUse(
            98, f"housemate hub: port {port} is already in use on this computer; stop the other "
            "program or set platforms.housemate.port to a free port"
        )
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
            if exclusive is not None:
                sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise HubPortInUse(
                exc.errno,
                f"housemate hub: {host or '0.0.0.0'}:{port} is already in use ({exc.strerror or exc}); "
                "stop the other program or set platforms.housemate.port to a free port",
            ) from exc
        sock.listen(128)
        return sock
    except BaseException:
        sock.close()
        raise


@dataclass(eq=False)
class _Session:
    ws: web.WebSocketResponse
    ip: str
    device_id: str | None = None
    name: str = ""
    hello: dict | None = None
    authed: bool = False
    code: str | None = None
    code_at: float = 0.0  # when the pairing code was handed out (hub clock)
    expiry: asyncio.Task | None = None
    conn: DeviceConn | None = None
    dropped: int = 0
    refusals: int = 0
    cb_errors: int = 0
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Hub:
    def __init__(
        self,
        store: HousemateStore,
        rooms: RoomRegistry,
        *,
        host: str,
        port: int,
        cert: Path | None,
        key: Path | None,
        web_root: Path,
        tls: bool = True,
        pairing_ttl_s: float = 600,
        hello_timeout_s: float = 10.0,
        max_pending_per_ip: int = 5,
        deny_ttl_s: float = 600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if tls and (cert is None or key is None):
            raise ValueError("tls=True needs cert and key paths")
        self.store = store
        self.rooms = rooms
        self.host = host
        self.port = port
        self.tls = tls
        self._cert, self._key = cert, key
        self._web_root = Path(web_root).resolve()
        self.pairing_ttl_s = pairing_ttl_s
        self.hello_timeout_s = hello_timeout_s
        self.max_pending_per_ip = max_pending_per_ip
        self.deny_ttl_s = deny_ttl_s
        self._clock = clock
        self.stats: dict[str, int] = {"dropped_frames": 0, "bad_messages": 0}

        self._audio_cbs: list[Callable[[str, bytes], None]] = []
        self._control_cbs: list[Callable[[str, dict], Awaitable[None] | None]] = []
        self._disconnect_cbs: list[Callable[[str], None]] = []
        self._pairing_cbs: list[Callable[[], None]] = []
        self._denied: dict[str, float] = {}  # device id -> refused until (hub clock)
        self._sessions: dict[str, _Session] = {}  # trusted, by device id
        self._waiting: dict[str, _Session] = {}  # waiting for approval, by code
        self._pending_by_ip: dict[str, dict[str, float]] = {}  # ip -> {code: expires_at}
        self._all: set[_Session] = set()
        self._bg: set[asyncio.Task] = set()
        self._runner: web.AppRunner | None = None

    # --- lifecycle ------------------------------------------------------------------

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "", None) else self.host
        if ":" in host:
            host = f"[{host}]"
        return f"{'https' if self.tls else 'http'}://{host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return ("wss" if self.tls else "ws") + self.url[self.url.index(":"):] + "/ws"

    async def start(self) -> None:
        app = web.Application(client_max_size=MAX_MESSAGE_BYTES)
        app.router.add_get("/ws", self._ws_handler)
        app.router.add_get("/{path:.*}", self._static)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        ssl_ctx = None
        if self.tls:
            ssl_ctx = server_ssl_context(self._cert, self._key)
        try:
            sock = bind_exclusive(self.host, self.port)
        except BaseException:
            await self._runner.cleanup()
            self._runner = None
            raise
        site = web.SockSite(self._runner, sock, ssl_context=ssl_ctx)
        await site.start()
        self.port = sock.getsockname()[1]
        log.info("housemate hub listening on %s", self.url)

    async def stop(self) -> None:
        sessions = list(self._all)
        await asyncio.gather(
            *(s.ws.close(code=1001, message=b"hub stopping") for s in sessions if not s.ws.closed),
            return_exceptions=True,
        )
        for task in list(self._bg):
            task.cancel()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    # --- adapter surface ---------------------------------------------------------------

    def on_audio(self, cb: Callable[[str, bytes], None]) -> None:
        """``cb(device_id, frame)`` for every valid 640-byte mic frame from a trusted device."""
        self._audio_cbs.append(cb)

    def on_control(self, cb: Callable[[str, dict], Awaitable[None]]) -> None:
        """``await cb(device_id, msg)`` for control messages from trusted devices.

        ``hello`` (without the key) is delivered whenever a device becomes trusted, so the
        adapter can replay queued captions. Handlers run inline on the socket's read loop:
        keep them quick.
        """
        self._control_cbs.append(cb)

    def on_disconnect(self, cb: Callable[[str], None]) -> None:
        """``cb(device_id)`` when a trusted device's live connection goes away (or is revoked)."""
        self._disconnect_cbs.append(cb)

    def on_pairing_change(self, cb: Callable[[], None]) -> None:
        """``cb()`` whenever the list of devices waiting for approval changes. Keep it quick."""
        self._pairing_cbs.append(cb)

    async def send_control(self, device_id: str, msg: dict) -> bool:
        """Send to a trusted, connected device. False if it is not there (never raises for that).

        A ``state`` message also updates the room registry.
        """
        text = encode(msg)
        s = self._sessions.get(device_id)
        if s is None or s.ws.closed:
            return False
        if msg["type"] == "state" and msg["state"] in DEVICE_STATES:
            self.rooms.set_device_state(device_id, msg["state"])
        return await self._send_text(s, text)

    async def send_pcm(self, device_id: str, pcm: bytes) -> bool:
        """Send playout PCM (0x02 frames, split to stay under the message cap)."""
        s = self._sessions.get(device_id)
        if s is None or s.ws.closed:
            return False
        data = bytes(pcm)
        prefix = bytes([PLAY_PCM])
        try:
            async with s.send_lock:
                for i in range(0, len(data), PLAYOUT_CHUNK):
                    await s.ws.send_bytes(prefix + data[i : i + PLAYOUT_CHUNK])
        except Exception as exc:  # the device went away mid-stream
            log.debug("housemate: pcm to %s failed: %s", device_id, exc)
            return False
        return True

    async def approve_pairing(self, code: str, room: str | None = None) -> bool:
        """Approve a code shown on a waiting satellite; that socket gets ``paired{device_key}``.

        With ``room`` the device is placed there before it hears back, so ``paired`` already
        carries its room (the Rooms page's drag-into-a-room)."""
        code = re.sub(r"\D", "", str(code))
        s = self._waiting.get(code)
        if s is None or s.ws.closed:
            return False
        result = self.store.approve_pairing(code, self._clock())
        if result is None:
            return False
        device_id, device_key = result
        self._forget_waiting(s)
        self.store.upsert_device(device_id, s.name)
        if room:
            self.store.set_room(device_id, room)
        await self._authenticate(s, device_key)
        return True

    async def deny_pairing(self, code: str) -> bool:
        """Refuse a waiting satellite: its socket is closed and it gets no new code for a while."""
        code = re.sub(r"\D", "", str(code))
        s = self._waiting.get(code)
        if s is None:
            return False
        self._forget_waiting(s)
        self.store.cancel_pairing(code)
        if s.device_id:
            self._denied[s.device_id] = self._clock() + self.deny_ttl_s
        await self._send(s, {"type": "error", "code": "pairing_denied", "message": "the owner did not approve this device"})
        self._close_later(s, CLOSE_DENIED, "pairing denied")
        return True

    def pending_pairings(self) -> list[dict[str, Any]]:
        """Devices waiting for approval, oldest first: code, device id, name, browser, address."""
        now = self._clock()
        out = []
        for code, s in self._waiting.items():
            if s.ws.closed:
                continue
            hello = s.hello or {}
            wants = hello.get("room")
            out.append({
                "code": code,
                "device_id": s.device_id,
                "name": s.name,
                "ua": str(hello.get("ua") or "")[:200],
                "ip": s.ip,
                "wants": wants.strip()[:64] if isinstance(wants, str) and wants.strip() else None,
                "since_s": max(0, round(now - s.code_at)),
                "expires_in_s": max(0, round(s.code_at + self.pairing_ttl_s - now)),
            })
        return sorted(out, key=lambda p: -p["since_s"])

    async def assign_room(self, device_id: str, room: str) -> bool:
        """Move a paired device to ``room``; an online one is told at once. False if unknown."""
        device = self.store.get_device(device_id)
        if device is None or not device.paired:
            return False
        target = self.store.ensure_room(room)
        self.store.set_room(device_id, target.slug)
        s = self._sessions.get(device_id)
        if s is not None and s.authed and not s.ws.closed and s.conn is not None:
            self.rooms.attach(device_id, target.slug, s.conn)
            await self._send(s, {"type": "state", "state": s.conn.state, "room": target.slug, "room_name": target.name})
            await self._emit_control(device_id, {"type": "room.set", "room": target.slug, "room_name": target.name})
        return True

    async def revoke_device(self, device_id: str) -> bool:
        """Forget a device's key and drop its live connection."""
        existed = self.store.revoke_device(device_id)
        s = self._sessions.pop(device_id, None)
        if s is not None:
            self.rooms.detach(device_id)
            self._close_later(s, CLOSE_BAD_KEY, "device revoked")
            self._disconnected(device_id)
        return existed

    def online_devices(self) -> list[str]:
        return [d for d, s in self._sessions.items() if not s.ws.closed]

    # --- HTTP ----------------------------------------------------------------------------

    async def _static(self, request: web.Request) -> web.StreamResponse:
        rel = request.match_info.get("path", "")
        if "\\" in rel or "\x00" in rel or ":" in rel:
            raise web.HTTPNotFound()
        parts = [p for p in rel.split("/") if p]
        if any(p.startswith(".") for p in parts):  # "..", ".", dotfiles
            raise web.HTTPNotFound()
        target = self._web_root.joinpath(*parts) if parts else self._web_root / "index.html"
        try:
            resolved = target.resolve(strict=True)
        except (OSError, RuntimeError):
            raise web.HTTPNotFound() from None
        if not resolved.is_relative_to(self._web_root) or not resolved.is_file():
            raise web.HTTPNotFound()
        ctype = _MIME.get(resolved.suffix.lower(), "application/octet-stream")
        return web.FileResponse(
            resolved,
            headers={"Content-Type": ctype, "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    # --- WebSocket -----------------------------------------------------------------------------

    async def _ws_handler(self, request: web.Request) -> web.StreamResponse:
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() != request.host.lower():
            raise web.HTTPForbidden(text="cross-origin satellite connection refused")
        ws = web.WebSocketResponse(max_msg_size=MAX_MESSAGE_BYTES, heartbeat=20.0, timeout=3.0)
        await ws.prepare(request)
        s = _Session(ws=ws, ip=request.remote or "?")
        self._all.add(s)
        try:
            await self._serve(s)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: satellite socket failed")
        finally:
            self._all.discard(s)
            self._drop(s)
            if not ws.closed:
                await ws.close()
        return ws

    async def _serve(self, s: _Session) -> None:
        loop = asyncio.get_running_loop()
        hello_deadline = loop.time() + self.hello_timeout_s
        while not s.ws.closed:
            timeout = None if s.hello is not None else max(0.0, hello_deadline - loop.time())
            try:
                msg = await s.ws.receive(timeout=timeout)
            except asyncio.TimeoutError:
                await s.ws.close(code=CLOSE_TIMEOUT, message=b"hello timeout")
                return
            if msg.type == WSMsgType.TEXT:
                await self._on_text(s, msg.data)
            elif msg.type == WSMsgType.BINARY:
                self._on_binary(s, msg.data)
            else:  # CLOSE / CLOSING / CLOSED / ERROR (e.g. 1009 message too big)
                return

    def _on_binary(self, s: _Session, data: bytes) -> None:
        if not s.authed or len(data) != 1 + FRAME_BYTES or data[0] != MIC_PCM:
            s.dropped += 1
            self.stats["dropped_frames"] += 1
            return
        frame = data[1:]
        for cb in self._audio_cbs:
            try:
                cb(s.device_id, frame)
            except Exception:
                if s.cb_errors % 500 == 0:  # 50 frames/s: don't flood the log
                    log.exception("housemate: audio callback failed")
                s.cb_errors += 1

    async def _on_text(self, s: _Session, text: str) -> None:
        try:
            msg = decode(text)
        except ProtocolError as exc:
            await self._refuse(s, "bad_message", str(exc))
            return
        mtype = msg["type"]
        if mtype not in SATELLITE_TYPES:
            await self._refuse(s, "bad_message", f"{mtype!r} is not a satellite message")
            return
        if mtype == "hello":
            if s.hello is not None:
                await self._refuse(s, "bad_message", "hello already sent")
                return
            await self._on_hello(s, msg)
            return
        if s.hello is None:
            await self._refuse(s, "hello_first", "send hello first")
            return
        if not s.authed:
            await self._refuse(s, "not_paired", "this device is waiting for pairing approval")
            return
        if mtype == "room.set":
            msg = await self._on_room_set(s, msg)
            if msg is None:
                return
        await self._emit_control(s.device_id, msg)

    async def _on_hello(self, s: _Session, msg: dict) -> None:
        s.device_id = msg["device_id"]
        s.name = msg["name"].strip() or s.device_id
        s.hello = {k: v for k, v in msg.items() if k != "device_key"}
        key = msg.get("device_key")
        if key:
            if not self.store.verify_device(s.device_id, key):
                await s.ws.close(code=CLOSE_BAD_KEY, message=b"bad device key")
                return
            self.store.upsert_device(s.device_id, s.name)
            await self._authenticate(s, key)
            return

        now = self._clock()
        refused_until = self._denied.get(s.device_id)
        if refused_until is not None:
            if refused_until > now:
                await self._send(s, {"type": "error", "code": "pairing_denied", "message": "the owner did not approve this device"})
                await s.ws.close(code=CLOSE_DENIED, message=b"pairing denied")
                return
            del self._denied[s.device_id]
        pending = self._pending_by_ip.setdefault(s.ip, {})
        for code, expires_at in list(pending.items()):
            if expires_at <= now:
                del pending[code]
        if len(pending) >= self.max_pending_per_ip:
            await self._send(s, {"type": "error", "code": "rate_limited", "message": "too many devices waiting to pair from this address"})
            await s.ws.close(code=CLOSE_RATE_LIMITED, message=b"rate limited")
            return
        for other in [w for w in self._waiting.values() if w.device_id == s.device_id]:
            self._forget_waiting(other)
            self._close_later(other, CLOSE_REPLACED, "replaced by a newer connection")
        if self.store.get_device(s.device_id) is None:
            self.store.upsert_device(s.device_id, s.name)
        code = self.store.create_pairing_code(s.device_id, now, self.pairing_ttl_s)
        s.code = code
        s.code_at = now
        self._waiting[code] = s
        pending[code] = now + self.pairing_ttl_s
        s.expiry = asyncio.create_task(self._expire_pairing(s, code))
        self._pairing_changed()
        await self._send(s, {"type": "pair.code", "code": code})

    async def _expire_pairing(self, s: _Session, code: str) -> None:
        await asyncio.sleep(self.pairing_ttl_s)
        if s.code != code or s.authed:
            return
        s.expiry = None  # this task is finishing; _drop must not cancel it
        self._forget_waiting(s)
        self.store.cancel_pairing(code)
        await self._send(s, {"type": "error", "code": "pairing_expired", "message": "pairing code expired; reconnect for a new one"})
        await s.ws.close(code=CLOSE_TIMEOUT, message=b"pairing code expired")

    async def _authenticate(self, s: _Session, device_key: str) -> None:
        s.authed = True
        device_id = s.device_id
        old = self._sessions.get(device_id)
        self._sessions[device_id] = s
        if old is not None and old is not s:
            self._close_later(old, CLOSE_REPLACED, "replaced by a newer connection")
        s.conn = DeviceConn(device_id=device_id, name=s.name, transport=s)
        device = self.store.get_device(device_id)
        room_slug = device.room if device else None
        room_name = None
        if room_slug:
            room_name = self.store.ensure_room(room_slug).name
            self.rooms.attach(device_id, room_slug, s.conn)
        else:
            self.rooms.detach(device_id)
        await self._send(
            s,
            {
                "type": "paired",
                "device_key": device_key,
                "room": room_slug,
                "room_name": room_name,
                "rooms": [{"slug": r.slug, "name": r.name} for r in self.store.list_rooms()],
            },
        )
        await self._emit_control(device_id, dict(s.hello or {"type": "hello", "device_id": device_id}))

    async def _on_room_set(self, s: _Session, msg: dict) -> dict | None:
        try:
            room = self.store.ensure_room(msg["room"])
        except ValueError:
            await self._refuse(s, "bad_room", "room name needs letters or digits")
            return None
        self.store.set_room(s.device_id, room.slug)
        self.rooms.attach(s.device_id, room.slug, s.conn)
        await self._send(s, {"type": "state", "state": s.conn.state, "room": room.slug, "room_name": room.name})
        return {**msg, "room": room.slug, "room_name": room.name}

    # --- helpers -------------------------------------------------------------------------------

    async def _emit_control(self, device_id: str, msg: dict) -> None:
        for cb in self._control_cbs:
            try:
                result = cb(device_id, msg)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("housemate: control callback failed for %s", msg.get("type"))

    async def _refuse(self, s: _Session, code: str, message: str) -> None:
        self.stats["bad_messages"] += 1
        s.refusals += 1
        if s.refusals > MAX_REFUSALS:
            await s.ws.close(code=CLOSE_ABUSE, message=b"too many bad messages")
            return
        await self._send(s, {"type": "error", "code": code, "message": message[:1024]})

    async def _send(self, s: _Session, msg: dict) -> bool:
        return await self._send_text(s, encode(msg))

    async def _send_text(self, s: _Session, text: str) -> bool:
        if s.ws.closed:
            return False
        try:
            async with s.send_lock:
                await s.ws.send_str(text)
        except Exception as exc:
            log.debug("housemate: send to %s failed: %s", s.device_id, exc)
            return False
        return True

    def _forget_waiting(self, s: _Session) -> None:
        code = s.code
        if code is None:
            return
        s.code = None
        if self._waiting.get(code) is s:
            del self._waiting[code]
        pending = self._pending_by_ip.get(s.ip)
        if pending is not None:
            pending.pop(code, None)
            if not pending:
                del self._pending_by_ip[s.ip]
        if s.expiry is not None:
            s.expiry.cancel()
            s.expiry = None
        self._pairing_changed()

    def _pairing_changed(self) -> None:
        for cb in list(self._pairing_cbs):
            try:
                cb()
            except Exception:
                log.exception("housemate: pairing change callback failed")

    def _drop(self, s: _Session) -> None:
        code = s.code
        self._forget_waiting(s)
        if code is not None:
            self.store.cancel_pairing(code)
        if s.authed and self._sessions.get(s.device_id) is s:
            del self._sessions[s.device_id]
            self.rooms.detach(s.device_id)
            self._disconnected(s.device_id)

    def _disconnected(self, device_id: str) -> None:
        for cb in list(self._disconnect_cbs):
            try:
                cb(device_id)
            except Exception:
                log.exception("housemate: disconnect callback failed")

    def _close_later(self, s: _Session, code: int, reason: str) -> None:
        if s.ws.closed:
            return
        task = asyncio.create_task(s.ws.close(code=code, message=reason.encode()))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
