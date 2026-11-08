"""REST routes of the Desktop Rooms page, for Hermes builds without plugin RPC (0.23.0).

Hermes mounts this router at ``/api/plugins/housemate/`` in the gateway process and the page
reaches it with ``ctx.rest``. Newer Hermes builds serve the same calls as plugin RPC methods
(``desktop_api.py``); the page tries those first. Each route forwards to one hub control op over
the loopback control API (token file in the plugin's data dir, 127.0.0.1 only), the same door
``hermes housemate`` uses, and returns its JSON (``ok`` + data or ``error``).

Hermes loads this file on its own, outside the plugin package, so it carries its own small
client instead of importing the package.
"""

import asyncio
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()

PLUGIN = "housemate"
HUB_DOWN = "hub not running: start the gateway with the housemate platform enabled"
TIMEOUTS = {"announce": 90.0}

# route -> (control op, body fields it forwards)
ACTIONS = {
    "announce": ("announce", ("text", "rooms")),
    "timer_cancel": ("timer_cancel", ("timer_id", "room", "label")),
    "pair_approve": ("pair_approve", ("code", "room")),
    "pair_deny": ("pair_deny", ("code",)),
    "assign": ("assign", ("device_id", "room")),
    "mode": ("mode", ("mode",)),
}


def _data_dir() -> Path | None:
    try:
        from plugins.plugin_storage import plugin_data_dir

        return Path(plugin_data_dir(PLUGIN))
    except Exception:
        return None


def _call(op: str, args: dict) -> dict:
    """POST one op to the loopback control API. Never raises."""
    data_dir = _data_dir()
    if data_dir is None:
        return {"ok": False, "error": HUB_DOWN}
    try:
        token = (data_dir / "control.token").read_text(encoding="utf-8").strip()
        port = int(json.loads((data_dir / "control.json").read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return {"ok": False, "error": HUB_DOWN}
    timeout = TIMEOUTS.get(op, 20.0)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/{op}",
        data=json.dumps(args).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # loopback: never a proxy
    try:
        with opener.open(request, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
    except (TimeoutError, socket.timeout):
        return {"ok": False, "error": f"the hub did not answer within {timeout:.0f} s"}
    except OSError:
        return {"ok": False, "error": HUB_DOWN}
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    return body if isinstance(body, dict) and "ok" in body else {"ok": False, "error": "unexpected answer from the hub"}


@router.get("/state")
async def state() -> dict:
    return await asyncio.to_thread(_call, "desk_state", {"qr": True})


@router.post("/{action}")
async def action(action: str, request: Request) -> dict:
    spec = ACTIONS.get(action)
    if spec is None:
        return {"ok": False, "error": f"unknown action {action!r}"}
    try:
        body = await request.json()
    except Exception:
        body = {}
    op, keys = spec
    body = body if isinstance(body, dict) else {}
    args = {k: body[k] for k in keys if body.get(k) is not None}
    return await asyncio.to_thread(_call, op, args)
