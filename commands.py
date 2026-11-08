"""Slash commands, the ``hermes housemate`` CLI, Automation Blueprints and the bundled skill.

``register_surface(ctx)`` is called from ``tools.register_tools`` so all of it exists in every
process, also when Hermes defers the platform adapter. Each part registers on its own: one that
an older or newer Hermes rejects is logged and the rest still load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from . import cli, tools

__all__ = ["BLUEPRINTS", "announce_command", "housemate_command", "register_surface"]

log = logging.getLogger(__name__)

SKILL_DIR = Path(__file__).resolve().parent / "skills"


# --- slash commands ------------------------------------------------------------------------------


async def housemate_command(raw_args: str = "", **kwargs: Any) -> str:
    """/housemate: rooms, devices and timers."""
    client = tools.client()
    rooms = await client.acall("rooms")
    if not rooms.get("ok"):
        return f"Housemate: {rooms.get('error')}"
    devices = await client.acall("devices")
    return cli.format_status(rooms, devices if devices.get("ok") else None)


async def announce_command(raw_args: str = "", **kwargs: Any) -> str:
    """/announce <text>: speak it in every room (except the room it was typed or said in)."""
    text = (raw_args or "").strip()
    if not text:
        return "Usage: /announce <text>   (spoken in every room)"
    args: dict[str, Any] = {"text": text, "rooms": "all"}
    if str(kwargs.get("platform") or "").strip().lower() == tools.PLUGIN_NAME:
        room = getattr(kwargs.get("source"), "user_name", None)
        if isinstance(room, str) and room.strip():
            args["from_room"] = room.strip()
    result = await tools.client().acall("announce", args)
    if not result.get("ok"):
        return f"Couldn't announce: {result.get('error')}"
    message = f"Announced in {', '.join(result.get('rooms') or [])}."
    if result.get("offline"):
        message += f" No device online in {', '.join(result['offline'])}."
    return message


# --- Automation Blueprints ------------------------------------------------------------------------

_SPOKEN = "It will be read aloud by a speaker, so use plain spoken sentences: no markdown, lists or URLs."


def _time(default: str) -> dict:
    return {"name": "time", "type": "time", "label": "What time?", "default": default,
            "help": "24-hour local time, e.g. 07:00"}


_RECURRENCE = {"name": "recurrence", "type": "weekdays", "label": "Repeat on", "default": "everyday",
               "options": ["everyday", "weekdays", "weekends"]}

_DELIVER = {
    "name": "deliver", "type": "enum", "label": "Which room speaks it?", "default": "housemate",
    "options": ["housemate", "origin", "local"], "strict": False,
    "help": "housemate = your Housemate home room (HOUSEMATE_HOME_CHANNEL); housemate:<room> = one "
            "room, e.g. housemate:kitchen; origin = the chat you set this up from",
}

BLUEPRINTS: dict[str, dict[str, Any]] = {
    "morning": dict(
        title="Spoken morning briefing",
        description="A 30-second briefing spoken in one room: weather, your first calendar item and "
                    "today's timers and alarms.",
        category="daily",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Give me a 30-second spoken morning briefing: weather for my location, my first calendar item "
            "if a calendar tool is available, and today's running timers or alarms. " + _SPOKEN
        ),
        slots=[_time("07:00"), _RECURRENCE, _DELIVER],
        deliver_default="housemate",
        tags=["housemate", "daily", "voice"],
    ),
    "bedtime": dict(
        title="Spoken goodnight",
        description="A short wind-down spoken in one room: tomorrow's first calendar item and the alarms "
                    "set for the morning.",
        category="daily",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Give me a 20-second spoken goodnight: my first calendar item tomorrow if a calendar tool is "
            "available, the alarms set for tomorrow morning and any timer still running (say so plainly if "
            "no alarm is set). " + _SPOKEN
        ),
        slots=[_time("22:30"), _RECURRENCE, _DELIVER],
        deliver_default="housemate",
        tags=["housemate", "daily", "voice"],
    ),
    "remind": dict(
        title="Spoken reminder",
        description="Say a reminder out loud in one room at a set time, e.g. 'take your vitamins' "
                    "every morning.",
        category="reminders",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template="Remind me out loud, in one short friendly sentence, to {what}. " + _SPOKEN,
        slots=[
            {"name": "what", "type": "text", "label": "Remind me to...", "default": "take your vitamins"},
            _time("09:00"),
            _RECURRENCE,
            _DELIVER,
        ],
        deliver_default="housemate",
        tags=["housemate", "reminder", "voice"],
    ),
}


# --- registration ----------------------------------------------------------------------------------


def _register_commands(ctx) -> None:
    ctx.register_command("housemate", housemate_command, description="Housemate rooms, devices and timers")
    ctx.register_command("announce", announce_command, description="Say something out loud in every room",
                         args_hint="<text>")


def _register_cli(ctx) -> None:
    ctx.register_cli_command(
        name="housemate",
        help="Housemate rooms, pairing, devices and the hub certificate",
        setup_fn=cli.setup_parser,
        handler_fn=cli.main,
        description=cli.DESCRIPTION,
    )


def _register_blueprints(ctx) -> None:
    for key, fields in BLUEPRINTS.items():
        ctx.register_automation_blueprint(key, **fields)


def _register_skill(ctx) -> None:
    for child in sorted(SKILL_DIR.iterdir()):
        skill_md = child / "SKILL.md"
        if child.is_dir() and skill_md.is_file():
            ctx.register_skill(child.name, skill_md,
                               description="Speaking through Housemate room speakers: speech vs cards, "
                                           "timers vs cron, announcements, room names")


def register_surface(ctx) -> None:
    parts: list[tuple[str, Callable[[Any], None]]] = [
        ("slash commands", _register_commands),
        ("CLI command", _register_cli),
        ("automation blueprints", _register_blueprints),
        ("skill", _register_skill),
    ]
    for what, register in parts:
        try:
            register(ctx)
        except Exception:
            log.warning("housemate: registering %s failed", what, exc_info=True)
