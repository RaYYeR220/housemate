"""Housemate settings, read from the gateway's ``platforms.housemate`` block.

Hermes promotes every non-typed key of a platform block into ``PlatformConfig.extra``, so a user
writes them directly under the platform::

    platforms:
      housemate:
        enabled: true
        conversation: household     # or per_room
        port: 8742
        rooms: [Kitchen, Office]

``HousemateConfig.from_extra`` never raises on bad input: a wrong value falls back to its default
and is listed in ``warnings`` so the adapter can log it once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

__all__ = ["CONVERSATION_MODES", "DEFAULT_PORT", "DEFAULT_CONTROL_PORT", "HousemateConfig"]

CONVERSATION_MODES = ("household", "per_room")
DEFAULT_PORT = 8742
DEFAULT_CONTROL_PORT = 8743
DEFAULT_WAKE_MODEL = "hey_jarvis"  # flips to "hey_hermes" once that head ships (falls back anyway)


def _port(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        port = int(value)
    except (TypeError, ValueError):
        return None
    return port if 1 <= port <= 65535 else None


def _number(value: Any, lo: float, hi: float) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if lo <= number <= hi else None


def _flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on", "1"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "off", "0"):
        return False
    return None


def _text(value: Any, max_len: int = 128) -> str | None:
    if isinstance(value, str) and value.strip() and len(value.strip()) <= max_len:
        return value.strip()
    return None


@dataclass(frozen=True)
class HousemateConfig:
    conversation: Literal["household", "per_room"] = "household"
    bind_host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    control_port: int = DEFAULT_CONTROL_PORT
    tls: bool = True
    wake_model: str = DEFAULT_WAKE_MODEL
    wake_threshold: float = 0.5
    vad: Literal["silero", "energy"] = "silero"
    followup_s: float = 6.0
    answer_window_s: float = 10.0  # no-wake window for "approve" / "deny" / a clarify choice
    turn_timeout_s: float = 180.0  # safety net: a household turn never blocks the queue longer
    rooms: tuple[str, ...] = ("Kitchen", "Office")
    earshot_groups: dict[str, str] = field(default_factory=dict)
    speak_replies: bool = True
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_extra(cls, extra: dict | None) -> "HousemateConfig":
        extra = extra if isinstance(extra, dict) else {}
        cfg = cls()
        warnings: list[str] = []
        changes: dict[str, Any] = {}

        def bad(key: str, why: str) -> None:
            warnings.append(f"platforms.housemate.{key}: {why}; using the default")

        if "conversation" in extra:
            mode = str(extra["conversation"]).strip().lower().replace("-", "_")
            if mode in CONVERSATION_MODES:
                changes["conversation"] = mode
            else:
                bad("conversation", f"must be one of {', '.join(CONVERSATION_MODES)}")
        if "bind_host" in extra:
            host = _text(extra["bind_host"], 64)
            if host is not None:
                changes["bind_host"] = host
            else:
                bad("bind_host", "must be an address such as 0.0.0.0")
        for key in ("port", "control_port"):
            if key in extra:
                port = _port(extra[key])
                if port is not None:
                    changes[key] = port
                else:
                    bad(key, "must be a TCP port number")
        for key, lo, hi in (
            ("wake_threshold", 0.05, 0.99),
            ("followup_s", 0.0, 30.0),
            ("answer_window_s", 3.0, 60.0),
            ("turn_timeout_s", 20.0, 1800.0),
        ):
            if key in extra:
                number = _number(extra[key], lo, hi)
                if number is not None:
                    changes[key] = number
                else:
                    bad(key, f"must be a number from {lo:g} to {hi:g}")
        for key in ("tls", "speak_replies"):
            if key in extra:
                flag = _flag(extra[key])
                if flag is not None:
                    changes[key] = flag
                else:
                    bad(key, "must be true or false")
        if "wake_model" in extra:
            model = _text(extra["wake_model"], 260)
            if model is not None:
                changes["wake_model"] = model
            else:
                bad("wake_model", "must be a model name or a path to a .onnx head")
        if "vad" in extra:
            vad = str(extra["vad"]).strip().lower()
            if vad in ("silero", "energy"):
                changes["vad"] = vad
            else:
                bad("vad", "must be silero or energy")
        if "rooms" in extra:
            raw = extra["rooms"]
            if isinstance(raw, str):
                raw = [part for part in raw.split(",")]
            names = [r.strip() for r in raw if isinstance(r, str) and r.strip()] if isinstance(raw, (list, tuple)) else None
            if names is not None:
                changes["rooms"] = tuple(dict.fromkeys(n[:64] for n in names))
            else:
                bad("rooms", "must be a list of room names")
        if "earshot_groups" in extra:
            raw = extra["earshot_groups"]
            if isinstance(raw, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
                changes["earshot_groups"] = {k.strip(): v.strip() for k, v in raw.items() if k.strip() and v.strip()}
            else:
                bad("earshot_groups", "must map room names to group names")
        return replace(cfg, **changes, warnings=tuple(warnings))
