"""Housemate agent tools (toolset ``housemate``), registered in every Hermes process.

``kind: platform`` plugins are deferred by Hermes when they are bundled; ``provides_tools`` in
plugin.yaml makes discovery import just this module and call ``register_tools(ctx)``. A normal
install calls the package's ``register(ctx)``, which calls ``register_tools`` too. Either way the
tools, slash commands, CLI, blueprints and skill all register here, once per plugin load.

Handlers take the model's arguments, call the hub through ``HousemateClient`` (in-process when the
gateway owns the hub, loopback HTTP otherwise) and return a JSON string. They never raise.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

__all__ = ["HANDLERS", "TOOLSET", "data_dir", "dispatch_tool", "register_tools", "registered_for", "session_room"]

log = logging.getLogger(__name__)

PLUGIN_NAME = "housemate"
TOOLSET = "housemate"
EMOJI = "\U0001F3E0"


# --- where the hub is, which room is speaking -------------------------------------------------------


def data_dir() -> Path | None:
    """``<hermes home>/plugin-data/housemate`` (holds the control token), or None without Hermes."""
    try:
        from plugins.plugin_storage import plugin_data_dir

        return plugin_data_dir(PLUGIN_NAME)
    except Exception:
        return None


def session_room() -> str | None:
    """The room the user is speaking from, when this call comes from a Housemate session.

    The adapter puts the room in the session source's ``user_name`` (and, per-room mode, the
    chat id is the room). Any other platform, or no session at all, gives None.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:
        return None
    try:
        if (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower() != PLUGIN_NAME:
            return None
        room = (get_session_env("HERMES_SESSION_USER_NAME", "") or "").strip()
        if not room:
            chat = (get_session_env("HERMES_SESSION_CHAT_ID", "") or "").strip()
            room = "" if chat.lower() in ("", "household") else chat
        return room or None
    except Exception:
        return None


def client():
    from .core.client import HousemateClient

    return HousemateClient(data_dir())


# --- Hermes tools the hub itself uses (text-to-speech for announcements and prompts) ------------------

_dispatch: Callable[..., Any] | None = None


def remember_dispatch(ctx) -> None:
    """Keep the plugin context's ``dispatch_tool`` so the adapter can call Hermes tools."""
    global _dispatch
    fn = getattr(ctx, "dispatch_tool", None)
    if callable(fn):
        _dispatch = fn


def dispatch_tool(name: str, args: dict) -> Any:
    """Run a Hermes tool (blocking) and return its JSON result: through the plugin context when
    registration handed us one, else through the public tool registry."""
    if _dispatch is not None:
        return _dispatch(name, args)
    from tools.registry import registry

    return registry.dispatch(name, args)


# --- handlers ------------------------------------------------------------------------------------


def _str_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _run(op: str, args: Any, build: Callable[[dict], dict]) -> str:
    try:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return json.dumps({"ok": False, "error": "arguments must be an object"})
        payload = {k: v for k, v in build(args).items() if v is not None}
        result = client().call(op, payload)
    except Exception as exc:  # never let a tool call raise into the agent loop
        log.exception("housemate: tool %s failed", op)
        result = {"ok": False, "error": f"housemate {op} failed: {exc}"}
    try:
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"ok": False, "error": f"housemate {op} returned something unreadable: {exc}"})


def housemate_rooms(args: dict | None = None, **kwargs: Any) -> str:
    return _run("rooms", args, lambda a: {})


def _rooms_arg(value: Any) -> Any:
    if isinstance(value, str):
        return "all" if value.strip().lower() == "all" else [value]
    if isinstance(value, list) and any(isinstance(v, str) and v.strip().lower() == "all" for v in value):
        return "all"
    return value


def housemate_announce(args: dict | None = None, **kwargs: Any) -> str:
    return _run("announce", args, lambda a: {
        "text": a.get("text"),
        "rooms": _rooms_arg(a.get("rooms", "all")),
        "from_room": session_room(),
    })


def housemate_timer_set(args: dict | None = None, **kwargs: Any) -> str:
    return _run("timer_set", args, lambda a: {
        "seconds": a.get("seconds"),
        "label": a.get("label"),
        "room": _str_or_none(a.get("room")),
        "session_room": session_room(),
    })


def housemate_alarm_set(args: dict | None = None, **kwargs: Any) -> str:
    return _run("alarm_set", args, lambda a: {
        "time": a.get("time"),
        "label": a.get("label"),
        "room": _str_or_none(a.get("room")),
        "session_room": session_room(),
    })


def housemate_timer_list(args: dict | None = None, **kwargs: Any) -> str:
    return _run("timer_list", args, lambda a: {"room": _str_or_none(a.get("room"))})


def housemate_timer_cancel(args: dict | None = None, **kwargs: Any) -> str:
    return _run("timer_cancel", args, lambda a: {
        "label": _str_or_none(a.get("label")),
        "room": _str_or_none(a.get("room")),
        "timer_id": _str_or_none(a.get("timer_id")),
        "session_room": session_room(),
    })


def housemate_show(args: dict | None = None, **kwargs: Any) -> str:
    return _run("show", args, lambda a: {
        "title": a.get("title"),
        "body": a.get("body"),
        "room": _str_or_none(a.get("room")),
        "session_room": session_room(),
    })


# --- schemas ---------------------------------------------------------------------------------------


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


_ROOM = _string("Room name, e.g. 'kitchen'. Leave it out to use the room the user is speaking from.")

# name -> (handler, description, properties, required)
_TOOLS: dict[str, tuple[Callable[..., str], str, dict, list[str]]] = {
    "housemate_rooms": (
        housemate_rooms,
        "List the rooms of the user's home that have a Housemate speaker: which devices are online in "
        "each room, what each room is doing (idle, listening, thinking, speaking, ringing, offline) and "
        "the timers and alarms running there. Use it to answer questions about the rooms or speakers, "
        "and to check room names before announcing.",
        {},
        [],
    ),
    "housemate_announce": (
        housemate_announce,
        "Speak a message out loud on the Housemate speakers in other rooms (intercom): a chime, the "
        "text as a caption, then the spoken message. Use it when the user asks to tell, call or "
        "announce something to another room or to everyone ('tell the kitchen dinner is ready'). "
        "rooms ['all'] reaches every room except the one the user is speaking from. Do not use it to "
        "answer the person talking to you; just reply. Write the text exactly as it should be heard.",
        {
            "text": _string("What to say, as it should be spoken, e.g. 'Dinner is ready.'"),
            "rooms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Rooms to speak in, e.g. ['kitchen', 'office'], or ['all'] for every other room.",
            },
        },
        ["text", "rooms"],
    ),
    "housemate_timer_set": (
        housemate_timer_set,
        "Start a countdown timer that rings in a room (chime plus the spoken label) when it ends. It is "
        "deterministic: it rings even if the conversation has moved on and survives a restart. Use it "
        "for 'set a timer for 10 minutes' or 'pasta timer, 8 minutes'; convert the duration to "
        "seconds. For a time of day use housemate_alarm_set. For repeating reminders or anything that "
        "needs you to think when it fires, use a cron job instead.",
        {
            "seconds": {"type": "integer", "minimum": 1, "maximum": 604800,
                        "description": "Duration in seconds, e.g. 600 for 10 minutes."},
            "label": _string("Short name said when it rings, e.g. 'pasta'. Optional."),
            "room": _ROOM,
        },
        ["seconds"],
    ),
    "housemate_alarm_set": (
        housemate_alarm_set,
        "Set a one-off alarm at a local time of day (24-hour HH:MM). It rings at the next occurrence: "
        "today if that time is still ahead, otherwise tomorrow. Use it for 'wake me at 7' or 'alarm at "
        "18:30'. For daily or weekly alarms use a cron job instead.",
        {
            "time": _string("Local time of day, 24-hour HH:MM, e.g. '07:30'."),
            "label": _string("Short name said when it rings, e.g. 'school run'. Optional."),
            "room": _ROOM,
        },
        ["time"],
    ),
    "housemate_timer_list": (
        housemate_timer_list,
        "List running timers and alarms: label, room, kind, time left in seconds (remaining_s) and the "
        "local time it is due. Leave room out for the whole home. Use it for 'how long is left on the "
        "pasta?' or 'what alarms are set?'.",
        {"room": _string("Only this room, e.g. 'kitchen'. Leave it out for every room.")},
        [],
    ),
    "housemate_timer_cancel": (
        housemate_timer_cancel,
        "Cancel timers or alarms, and silence them if they are ringing. Match by label ('cancel the "
        "pasta timer'), by room (everything in that room) or by timer_id from housemate_timer_list. "
        "With no arguments it cancels the timers in the room the user is speaking from.",
        {
            "label": _string("Label of the timer or alarm, e.g. 'pasta'."),
            "room": _string("Cancel everything in this room, e.g. 'kitchen'."),
            "timer_id": _string("Exact id from housemate_timer_list."),
        },
        [],
    ),
    "housemate_show": (
        housemate_show,
        "Put a card with a title and plain text on a room's screen. Use it for anything better read "
        "than heard: recipes, ingredient lists, steps, numbers, addresses, links, tables. Then say one "
        "short sentence about it instead of reading it out. Defaults to the room the user is speaking "
        "from.",
        {
            "title": _string("Card title, e.g. 'Pancakes'."),
            "body": _string("Card text. Plain text; new lines and simple '-' lists are fine."),
            "room": _ROOM,
        },
        ["title", "body"],
    ),
}

HANDLERS: dict[str, Callable[..., str]] = {name: spec[0] for name, spec in _TOOLS.items()}


def _available() -> bool:
    return True  # unreachable hub is reported by each call, so the model can tell the user


# --- registration ------------------------------------------------------------------------------------

_registered_manifests: list[Any] = []


def registered_for(ctx) -> bool:
    """True when ``register_tools`` already ran for this plugin load (same manifest object)."""
    manifest = getattr(ctx, "manifest", None)
    return manifest is not None and any(m is manifest for m in _registered_manifests)


def register_tools(ctx) -> None:
    """Tools, slash commands, CLI, blueprints and skill: everything that must exist in every process."""
    for name, (handler, description, properties, required) in _TOOLS.items():
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = list(required)
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema={"name": name, "description": description, "parameters": parameters},
            handler=handler,
            check_fn=_available,
            description=description,
            emoji=EMOJI,
        )
    from .commands import register_surface

    register_surface(ctx)
    try:  # the Desktop Rooms page's RPC methods (Hermes builds with ctx.register_rpc)
        from .desktop_api import register_desktop

        register_desktop(ctx)
    except Exception:
        log.warning("housemate: registering the Rooms page methods failed", exc_info=True)
    remember_dispatch(ctx)
    manifest = getattr(ctx, "manifest", None)
    if manifest is not None:
        _registered_manifests.append(manifest)
