"""Stand-ins for the few public Hermes gateway classes the adapter subclasses and builds.

Installed only when Hermes itself is not importable, so the adapter's unit tests run on a plain
Python with the plugin's own dependencies. With Hermes installed the real classes are used (and
the ``hermes``-marked tests exercise them).
"""

from __future__ import annotations

import enum
import importlib.util
import sys
import tempfile
import types
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

HERMES_AVAILABLE = importlib.util.find_spec("gateway") is not None and importlib.util.find_spec("gateway.platforms") is not None


def _module(name: str, **attrs: Any) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # a package: unknown submodules fail with ModuleNotFoundError
    mod.__dict__.update(attrs)
    return mod


class Platform(str):
    @property
    def value(self) -> str:
        return str(self)


@dataclass
class PlatformConfig:
    enabled: bool = True
    extra: dict = field(default_factory=dict)
    home_channel: Any = None


@dataclass
class AudioFormat:
    sample_rate: int = 24000
    channels: int = 1
    sample_width: int = 2


@dataclass
class StreamingTTSHandle:
    chat_id: str = ""
    audio_format: AudioFormat = field(default_factory=AudioFormat)
    audible: bool = False
    aborted: bool = False


@dataclass
class ExecApprovalPrompt:
    chat_id: str
    session_key: str
    text: str
    actions: list
    command: str
    description: str
    smart_denied: bool
    metadata: Optional[dict] = None
    request_id: Optional[str] = None

    @property
    def choices(self) -> list:
        return [choice for _, choice, _ in self.actions]


@dataclass
class SendResult:
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class MessageType(enum.Enum):
    TEXT = "text"
    VOICE = "voice"


@dataclass
class SessionSource:
    platform: Any
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"
    user_id: Optional[str] = None
    user_name: Optional[str] = None


@dataclass
class MessageEvent:
    text: str
    message_type: MessageType = MessageType.TEXT
    source: Any = None
    message_id: Optional[str] = None
    media_urls: list = field(default_factory=list)
    media_types: list = field(default_factory=list)
    channel_prompt: Optional[str] = None
    channel_context: Optional[str] = None
    internal: bool = False
    metadata: dict = field(default_factory=dict)


class BasePlatformAdapter:
    gateway_runner = None

    def __init__(self, config, platform) -> None:
        self.config = config
        self.platform = platform
        self._active_sessions: dict = {}
        self._running = False
        self.fatal: tuple | None = None

    def build_source(self, chat_id, chat_name=None, chat_type="dm", user_id=None, user_name=None, **_: Any):
        return SessionSource(self.platform, str(chat_id), chat_name, chat_type, user_id, user_name)

    async def handle_message(self, event) -> None:
        raise NotImplementedError("tests replace handle_message")

    def _event_session_key(self, event) -> str:
        return f"agent:main:{self.platform}:dm:{event.source.chat_id}"

    def _mark_connected(self, **_: Any) -> None:
        self._running = True

    def _mark_disconnected(self) -> None:
        self._running = False

    def _set_fatal_error(self, code, message, *, retryable) -> None:
        self.fatal = (code, message, retryable)
        self._running = False

    def _should_auto_tts_for_chat(self, chat_id) -> bool:
        return False

    @property
    def is_connected(self) -> bool:
        return self._running


class OwnAccessPolicyMixin:
    ALLOW_ALL_ENV_PREFIX = ""

    @property
    def enforces_own_access_policy(self) -> bool:
        return True

    def _entry_matches(self, entries, target) -> bool:
        return target in entries

    def _live_dm_allow_from(self):
        return self._allow_from

    def _is_dm_allowed(self, sender_id) -> bool:
        if self._dm_policy == "allowlist":
            return self._entry_matches(self._live_dm_allow_from(), sender_id)
        return False


_CACHE = Path(tempfile.mkdtemp(prefix="housemate-shim-audio-"))


def cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
    path = _CACHE / f"audio_{uuid.uuid4().hex[:12]}{ext}"
    path.write_bytes(data)
    return str(path)


async def cache_audio_from_bytes_async(data: bytes, ext: str = ".ogg") -> str:
    return cache_audio_from_bytes(data, ext)


def get_scoped_secret(name: str, default: Any = None, **_: Any) -> Any:
    """Hermes reads env through the profile's secret scope; without Hermes that is the process env."""
    import os

    value = os.environ.get(name)
    return default if value is None else value


def install() -> bool:
    """Put the stand-ins in place unless Hermes is importable. True when the shim is active."""
    if HERMES_AVAILABLE:
        return False
    if "gateway" in sys.modules and getattr(sys.modules["gateway"], "__housemate_shim__", False):
        return True
    base = _module(
        "gateway.platforms.base",
        AudioFormat=AudioFormat,
        BasePlatformAdapter=BasePlatformAdapter,
        ExecApprovalPrompt=ExecApprovalPrompt,
        SendResult=SendResult,
        StreamingTTSHandle=StreamingTTSHandle,
        cache_audio_from_bytes=cache_audio_from_bytes,
        cache_audio_from_bytes_async=cache_audio_from_bytes_async,
    )
    event = _module("gateway.platforms.event", MessageEvent=MessageEvent, MessageType=MessageType)
    mixin = _module("gateway.platforms.access_policy_mixin", OwnAccessPolicyMixin=OwnAccessPolicyMixin)
    shared = _module("gateway.platforms._shared", get_scoped_secret=get_scoped_secret)
    platforms = _module("gateway.platforms", base=base, event=event, access_policy_mixin=mixin, _shared=shared)
    config = _module("gateway.config", Platform=Platform, PlatformConfig=PlatformConfig)
    gateway = _module("gateway", platforms=platforms, config=config, __housemate_shim__=True)
    sys.modules.update({
        "gateway": gateway,
        "gateway.config": config,
        "gateway.platforms": platforms,
        "gateway.platforms.base": base,
        "gateway.platforms.event": event,
        "gateway.platforms.access_policy_mixin": mixin,
        "gateway.platforms._shared": shared,
    })
    return True
