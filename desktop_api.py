"""Backend of the Desktop Rooms page (``desktop/plugin.js``): plugin RPC methods.

On a Hermes whose plugin context has ``register_rpc`` the page calls these over the gateway
socket as ``plugin.housemate.rooms.<name>``; each one forwards to a hub control op through
``HousemateClient`` (straight onto the hub's loop in the gateway process, else the loopback
control API), so it works in whichever process serves the call. Hermes 0.23.0 has no plugin
RPC yet: there ``dashboard/plugin_api.py`` serves the same calls as REST routes under
``/api/plugins/housemate/`` and the page falls back to them.

Live updates do not go through here: the adapter pushes ``plugin.housemate.rooms.changed``
(see ``core/desk.py``) and the page polls when no event arrives.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

__all__ = ["RPC_METHODS", "call", "register_desktop"]

log = logging.getLogger(__name__)

# RPC name -> (control op, the params it forwards)
RPC_METHODS: dict[str, tuple[str, tuple[str, ...]]] = {
    "rooms.state": ("desk_state", ("qr",)),
    "rooms.announce": ("announce", ("text", "rooms")),
    "rooms.timer_cancel": ("timer_cancel", ("timer_id", "room", "label")),
    "rooms.pair_approve": ("pair_approve", ("code", "room")),
    "rooms.pair_deny": ("pair_deny", ("code",)),
    "rooms.assign": ("assign", ("device_id", "room")),
    "rooms.mode": ("mode", ("mode",)),
}


def call(method: str, params: Any = None, client: Any = None) -> dict:
    """Run one Rooms-page call. Returns the op's JSON dict (``ok`` + data or ``error``); never raises."""
    spec = RPC_METHODS.get(method)
    if spec is None:
        return {"ok": False, "error": f"unknown method {method!r}"}
    op, keys = spec
    params = params if isinstance(params, dict) else {}
    args = {k: params[k] for k in keys if params.get(k) is not None}
    if method == "rooms.state":
        args = {"qr": bool(params.get("qr", True))}
    if client is None:
        from .tools import client as make_client

        client = make_client()
    try:
        return client.call(op, args)
    except Exception as exc:  # the client never raises; belt and braces for the RPC lane
        log.exception("housemate: desktop call %s failed", method)
        return {"ok": False, "error": f"{method} failed: {exc}"}


def _handler(method: str) -> Callable[..., dict]:
    def handle(params: Any = None, *, session_id: str | None = None, **kwargs: Any) -> dict:
        return call(method, params)

    handle.__name__ = "housemate_" + method.replace(".", "_")
    return handle


def register_desktop(ctx: Any) -> list[str]:
    """Register the Rooms page's RPC methods when this Hermes has plugin RPC; else nothing."""
    register = getattr(ctx, "register_rpc", None)
    if not callable(register):
        return []
    done = []
    for method in RPC_METHODS:
        try:
            register(method, _handler(method))
            done.append(method)
        except Exception:
            log.warning("housemate: registering the RPC method %s failed", method, exc_info=True)
    return done
