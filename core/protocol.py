"""Satellite <-> hub wire protocol.

Binary frames carry a one-byte prefix: ``0x01`` mic PCM16 LE mono 16 kHz (satellite -> hub,
exactly one 20 ms / 640-byte frame each), ``0x02`` playout PCM16 LE mono at the rate given
in ``play.begin`` (hub -> satellite). Text frames are JSON control messages with a ``type``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

__all__ = [
    "FRAME_BYTES",
    "HUB_TYPES",
    "MAX_MESSAGE_BYTES",
    "MIC_PCM",
    "PLAY_PCM",
    "SAMPLE_RATE",
    "SATELLITE_TYPES",
    "ProtocolError",
    "decode",
    "encode",
]

MIC_PCM = 0x01
PLAY_PCM = 0x02
SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2  # 640
MAX_MESSAGE_BYTES = 64 * 1024


class ProtocolError(ValueError):
    """A control message that is not valid JSON, has an unknown type or lacks fields."""


_DEVICE_ID = re.compile(r"^[A-Za-z0-9_.:-]{4,128}$")


def _str(max_len: int, *, min_len: int = 0) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and min_len <= len(v) <= max_len


def _bool(v: Any) -> bool:
    return isinstance(v, bool)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _list(v: Any) -> bool:
    return isinstance(v, list)


def _scalar(v: Any) -> bool:
    return v is None or isinstance(v, (str, int, float, bool))


def _device_id(v: Any) -> bool:
    return isinstance(v, str) and bool(_DEVICE_ID.match(v))


_ID = _str(128, min_len=1)
_TEXT = _str(16_000)

# type -> (required {field: check}, optional {field: check})
SATELLITE_TYPES: dict[str, tuple[dict, dict]] = {
    # ``room`` (optional): the room the user picked before pairing; the Rooms page shows it.
    "hello": ({"device_id": _device_id, "name": _str(64), "ua": _str(512)}, {"device_key": _str(256), "room": _str(64)}),
    "room.set": ({"room": _str(64, min_len=1)}, {}),
    "ptt": ({"on": _bool}, {}),
    "mute": ({"on": _bool}, {}),
    "playback.done": ({"stream_id": _ID}, {}),
    "ui.action": ({"kind": _str(64, min_len=1), "id": _str(128)}, {"value": _scalar}),
}

HUB_TYPES: dict[str, tuple[dict, dict]] = {
    "pair.code": ({"code": _str(16, min_len=1)}, {}),
    "paired": ({"device_key": _str(256, min_len=1)}, {"room": lambda v: v is None or _str(64)(v)}),
    "state": ({"state": _str(32, min_len=1)}, {}),
    "caption": ({"role": _str(32, min_len=1), "text": _TEXT}, {}),
    "card": ({"id": _ID, "title": _str(512), "body": _TEXT}, {}),
    "play.begin": ({"stream_id": _ID, "sample_rate": _int}, {}),
    "play.end": ({"stream_id": _ID}, {}),
    "play.abort": ({"stream_id": _ID}, {}),
    "timers": ({"items": _list}, {}),
    "prompt": ({"kind": _str(32, min_len=1), "id": _ID, "text": _TEXT, "choices": _list}, {}),
    "prompt.clear": ({"id": _ID}, {}),
    "chime": ({"kind": _str(32, min_len=1)}, {}),
    # Not in the original list: tells a satellite why its message was refused.
    "error": ({"code": _str(64, min_len=1), "message": _str(1024)}, {}),
}

_ALL_TYPES = {**SATELLITE_TYPES, **HUB_TYPES}


def _validate(msg: Any) -> dict:
    if not isinstance(msg, dict):
        raise ProtocolError("control message must be a JSON object")
    mtype = msg.get("type")
    if not isinstance(mtype, str):
        raise ProtocolError("control message needs a string 'type'")
    spec = _ALL_TYPES.get(mtype)
    if spec is None:
        raise ProtocolError(f"unknown message type: {mtype[:64]!r}")
    required, optional = spec
    for name, check in required.items():
        if name not in msg:
            raise ProtocolError(f"{mtype}: missing field {name!r}")
        if not check(msg[name]):
            raise ProtocolError(f"{mtype}: bad value for {name!r}")
    for name, check in optional.items():
        if name in msg and not check(msg[name]):
            raise ProtocolError(f"{mtype}: bad value for {name!r}")
    return msg


def encode(msg: dict) -> str:
    """Validate and serialise a control message (extra fields are kept)."""
    return json.dumps(_validate(msg), separators=(",", ":"), ensure_ascii=False)


def decode(text: str) -> dict:
    """Parse and validate a control message; raises ProtocolError."""
    try:
        msg = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from None
    return _validate(msg)
