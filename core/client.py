"""``HousemateClient``: the hub's ops from any Hermes process (tools, slash commands, the CLI).

In the gateway process the hub registered itself (``control.register_local_services``) and calls
go straight onto its event loop. Anywhere else they go to the loopback control API with the token
from ``<data dir>/control.token``. Every call returns a dict with ``ok``; when the hub cannot be
reached it is ``{"ok": False, "error": HUB_DOWN}``. Nothing here raises.

Stdlib only (urllib), so importing it costs nothing in processes that never call the hub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .control import DEFAULT_PORT, LOOPBACK, PORT_FILE, local_services, run_op

__all__ = ["HUB_DOWN", "TOKEN_FILE", "HousemateClient"]

log = logging.getLogger(__name__)

HUB_DOWN = "hub not running: start the gateway with the housemate platform enabled"
TOKEN_FILE = "control.token"
DEFAULT_TIMEOUT = 20.0
_OP_TIMEOUTS = {"announce": 90.0}  # chime + text-to-speech + playout


def _drop_none(args: dict) -> dict:
    return {k: v for k, v in args.items() if v is not None}


class HousemateClient:
    def __init__(self, data_dir: Path | None, *, timeout: float | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.timeout = timeout

    # --- the ops -------------------------------------------------------------------------------

    def rooms(self) -> dict:
        return self.call("rooms")

    def announce(self, text: str, rooms: list[str] | str = "all", from_room: str | None = None) -> dict:
        return self.call("announce", _drop_none({"text": text, "rooms": rooms, "from_room": from_room}))

    def timer_set(self, seconds: int, label: str = "", room: str | None = None, session_room: str | None = None) -> dict:
        return self.call("timer_set", _drop_none({"seconds": seconds, "label": label, "room": room, "session_room": session_room}))

    def alarm_set(self, time: str, label: str = "", room: str | None = None, session_room: str | None = None) -> dict:
        return self.call("alarm_set", _drop_none({"time": time, "label": label, "room": room, "session_room": session_room}))

    def timer_list(self, room: str | None = None) -> dict:
        return self.call("timer_list", _drop_none({"room": room}))

    def timer_cancel(self, label: str | None = None, room: str | None = None, timer_id: str | None = None,
                     session_room: str | None = None) -> dict:
        return self.call("timer_cancel", _drop_none(
            {"label": label, "room": room, "timer_id": timer_id, "session_room": session_room}))

    def show(self, title: str, body: str, room: str | None = None, session_room: str | None = None) -> dict:
        return self.call("show", _drop_none({"title": title, "body": body, "room": room, "session_room": session_room}))

    def pair_approve(self, code: str, room: str | None = None) -> dict:
        return self.call("pair_approve", _drop_none({"code": code, "room": room}))

    def pair_deny(self, code: str) -> dict:
        return self.call("pair_deny", {"code": code})

    def assign(self, device_id: str, room: str) -> dict:
        return self.call("assign", {"device_id": device_id, "room": room})

    def mode(self, mode: str | None = None) -> dict:
        return self.call("mode", _drop_none({"mode": mode}))

    def desk_state(self, qr: bool = False) -> dict:
        return self.call("desk_state", {"qr": qr})

    def devices(self) -> dict:
        return self.call("devices")

    def revoke(self, device_id: str) -> dict:
        return self.call("revoke", {"device_id": device_id})

    def timings(self) -> dict:
        return self.call("timings")

    def info(self) -> dict:
        return self.call("info")

    # --- transport -------------------------------------------------------------------------------

    def _timeout(self, op: str, timeout: float | None) -> float:
        if timeout is not None:
            return timeout
        if self.timeout is not None:
            return self.timeout
        return _OP_TIMEOUTS.get(op, DEFAULT_TIMEOUT)

    def _local(self):
        local = local_services(self.data_dir)
        if local is not None and local[1].is_running():
            return local
        return None

    def call(self, op: str, args: dict | None = None, *, timeout: float | None = None) -> dict:
        """Blocking call. Do not use it on the hub's own event loop (``acall`` there)."""
        args = dict(args or {})
        timeout = self._timeout(op, timeout)
        local = self._local()
        if local is not None:
            services, loop = local
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is loop:
                return {"ok": False, "error": "called on the hub's event loop; use acall() there"}
            future = asyncio.run_coroutine_threadsafe(run_op(services, op, args), loop)
            try:
                return future.result(timeout)
            except TimeoutError:
                future.cancel()
                return {"ok": False, "error": f"the hub did not answer within {timeout:.0f} s"}
            except Exception as exc:  # the loop went away under us
                log.debug("housemate: in-process %s failed: %s", op, exc)
                return {"ok": False, "error": HUB_DOWN}
        return self._http(op, args, timeout)

    async def acall(self, op: str, args: dict | None = None, *, timeout: float | None = None) -> dict:
        """Call from async code, including the hub's own event loop."""
        args = dict(args or {})
        timeout = self._timeout(op, timeout)
        local = self._local()
        if local is None:
            return await asyncio.to_thread(self._http, op, args, timeout)
        services, loop = local
        try:
            if asyncio.get_running_loop() is loop:
                return await asyncio.wait_for(run_op(services, op, args), timeout)
            future = asyncio.run_coroutine_threadsafe(run_op(services, op, args), loop)
            try:
                return await asyncio.wait_for(asyncio.wrap_future(future), timeout)
            finally:
                if not future.done():
                    future.cancel()
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"the hub did not answer within {timeout:.0f} s"}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("housemate: in-process %s failed: %s", op, exc)
            return {"ok": False, "error": HUB_DOWN}

    def _http(self, op: str, args: dict, timeout: float) -> dict:
        if self.data_dir is None:
            return {"ok": False, "error": HUB_DOWN}
        try:
            token = (self.data_dir / TOKEN_FILE).read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if not token:
            return {"ok": False, "error": HUB_DOWN}
        port = DEFAULT_PORT
        try:
            port = int(json.loads((self.data_dir / PORT_FILE).read_text(encoding="utf-8"))["port"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        request = urllib.request.Request(
            f"http://{LOOPBACK}:{port}/v1/{urllib.parse.quote(op, safe='')}",
            data=json.dumps(args).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback: never a proxy
        status = 0
        try:
            with opener.open(request, timeout=timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
            if status == 401:
                return {"ok": False, "error": "the hub rejected the control token; if the gateway just restarted, try again"}
        except (TimeoutError, socket.timeout):
            return {"ok": False, "error": f"the hub did not answer within {timeout:.0f} s"}
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                return {"ok": False, "error": f"the hub did not answer within {timeout:.0f} s"}
            return {"ok": False, "error": HUB_DOWN}
        except OSError:
            return {"ok": False, "error": HUB_DOWN}
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if isinstance(body, dict) and "ok" in body:
            return body
        return {"ok": False, "error": f"unexpected answer from the hub (HTTP {status})"}
