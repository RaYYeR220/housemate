"""The Housemate gateway platform: room speakers in, spoken answers out.

``connect()`` starts everything the gateway needs for voice rooms:

- the LAN hub (HTTPS + WebSocket, self-signed certificate) that serves the satellite web app and
  takes each paired device's microphone stream;
- one ``DeviceListener`` per paired device (wake word, endpointing, cross-device arbitration);
- timers, announcements and the loopback control API the agent tools talk to.

Inbound, an endpointed utterance becomes a VOICE ``MessageEvent`` (Hermes transcribes it with the
user's STT). Its room reaches the model as ``channel_prompt`` (this turn) and ``channel_context``
(kept in history), never as event metadata. ``SessionRouter`` decides the chat (one household
conversation by default) and queues a second room that speaks while a turn is running.

Outbound, Hermes's streaming TTS is paced to the device that asked (``Playout``), the reply text
becomes a caption in that room, and a short follow-up window opens when the device reports that
playback finished. Approvals and clarify prompts are spoken and shown as cards; a spoken answer goes
through the public side-chat route in ``core.answers``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import socket
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    AudioFormat,
    BasePlatformAdapter,
    ExecApprovalPrompt,
    SendResult,
    StreamingTTSHandle,
    cache_audio_from_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType

from .core import answers
from .core.announce import PLAY_RATE, Announcer, decode_to_pcm, tts_paths_from_result
from .core.arbiter import WakeArbiter
from .core.audio import pcm_to_wav
from .core.config import HousemateConfig
from .core.desk import DeskFeed, DeskLog, desk_state, persist_mode
from .core.intents import is_skip, is_stop, normalize, parse_choice, parse_yes_no, spoken_label
from .core.playout import PcmRealigner, Playout
from .core.protocol import SAMPLE_RATE as MIC_RATE
from .core.router import HOUSEHOLD_CHAT, SessionRouter
from .core.store import slugify

__all__ = ["HousemateAdapter", "PLATFORM_NAME"]

log = logging.getLogger("housemate")

PLATFORM_NAME = "housemate"
WEB_ROOT = Path(__file__).resolve().parent / "web"
SIDE_CHAT = answers.SIDE_CHAT

# Gateway notices that make no sense out loud on a kitchen speaker.
_SWALLOWED_PREFIXES = ("\U0001F4EC", "♻")  # "📬 No home channel is set…", "♻️ Gateway restarted…"
_TRANSCRIPT_PREFIX = "\U0001F399"  # "🎙️ "<transcript>"": the STT echo
# Housekeeping notices: caption only. 💾 self-improvement, ⚡ interrupt, 💡 tip, ⏳/⌛ waits, 🔄 retry.
_STATUS_PREFIXES = ("\U0001F4BE", "⚡", "\U0001F4A1", "⏳", "⌛", "\U0001F504")
_QUOTES = "\"'“”‘’ ️"

CAPTION_TTL_S = 15 * 60
CAPTION_KEEP = 5
SESSION_IDLE_WAIT_S = 20.0  # a spoken answer waits this long for the side chat to go idle
SESSION_BUSY_MAX_WAIT_S = 300.0  # a queued question waits this long for Hermes, then is given up
TURN_TIMEOUT_MARGIN_S = 60.0  # the turn safety net sits this far past Hermes's approval timeout
APPROVAL_ANSWER_TTL_S = 310.0  # a little past Hermes's default 5-minute approval timeout
STOP_ANSWER_TTL_S = 200.0
PLAYBACK_GRACE_S = 3.0
RECENT_TURN_S = 60.0
MAX_REPROMPTS = 2
INTERCOM_WINDOW_S = 8.0  # after a tap on another room: start talking within this, it plays there
INTERCOM_TTL_S = INTERCOM_WINDOW_S + 17.0  # window + the longest utterance + endpointing
TURN_LOG_SIZE = 100  # finished turns whose stage times the control API reports
# A spoken approval or clarify answer is taken only from a clip that began this long after the last
# sound any Housemate device played (and while none plays): never from our own audio.
ECHO_TAIL_S = 1.0
CHIME_S = 1.5  # a satellite chime, counted as playback (it is played on the device, not streamed)
ANNOUNCED_TTL_S = 600.0  # what we announced this recently is never an approval answer


def wake_phrase(model: str) -> str:
    """The wake model's name as people say it: "hey_jarvis" / "hey_jarvis_v0.1.onnx" -> "Hey Jarvis"."""
    name = str(model or "").replace("\\", "/").rsplit("/", 1)[-1]
    if name.lower().endswith(".onnx"):
        name = name[:-5]
    name = re.sub(r"_v\d+(?:\.\d+)*$", "", name)
    return " ".join(w.capitalize() for w in re.split(r"[_\-\s]+", name) if w)[:64]


# --- small records -------------------------------------------------------------------------------


@dataclass
class _Utterance:
    message_id: str
    device_id: str
    room: str
    kind: str  # wake | followup | ptt
    pcm: bytes
    endpoint_at: float  # wall clock: when speech ended
    captured_at: float  # wall clock: when the listener handed it over


@dataclass
class _Stream:
    """One playout on one device (a reply, a spoken prompt or a fallback reply)."""

    stream_id: str
    device_id: str
    kind: str  # rep | ask
    rate: int
    playout: Playout
    ref: str | None = None  # the prompt id for "ask" streams
    task: asyncio.Task | None = None
    started_at: float = field(default_factory=time.monotonic)
    on_first_pcm: Callable[[], None] | None = None


@dataclass
class _Playing:
    """What a device is playing right now (ours or an announcer stream)."""

    stream_id: str
    kind: str  # rep | ask | say | ann
    rate: int
    started_at: float
    bytes: int = 0
    end_timer: asyncio.TimerHandle | None = None


@dataclass
class _Prompt:
    kind: str  # approval | clarify
    id: str  # card id: the approval request id or the clarify id
    chat_id: str
    session_key: str
    room: str | None
    device_id: str | None
    question: str
    choices: list[str]
    request_id: str | None = None
    tries: int = 0
    skippable: bool = False  # a clarify question the user may skip ("Or say skip.")
    title: str = ""  # approval card: the short question, what it is, the command itself
    detail: str = ""
    command: str = ""
    spoken: str = ""  # what is said when the prompt comes up
    smart_denied: bool = False  # Hermes's smart approval flagged the command: tap only
    presented: bool = False  # shown (and spoken, armed): the first open prompt of its room
    shown_to: set[str] = field(default_factory=set)  # devices that got the card: only they may tap
    turn_id: str | None = None  # message id of the turn that asked (retired when Hermes is done with it)


class _ReplyHandle(StreamingTTSHandle):
    """The streaming-TTS handle Hermes passes back to write/finish/abort."""

    def __init__(self, chat_id: str, audio_format: AudioFormat, stream: _Stream) -> None:
        super().__init__(chat_id=chat_id, audio_format=audio_format)
        self.stream = stream
        self.realign = PcmRealigner()
        self.chunks = 0
        self.turn: Any = None  # the router turn this stream answers


class _HubTap:
    """Wraps the hub for the announcer so the adapter sees its playback (echo guard, follow-ups)."""

    def __init__(self, hub: Any, adapter: "HousemateAdapter") -> None:
        self._hub = hub
        self._adapter = adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._hub, name)

    async def send_control(self, device_id: str, msg: dict) -> bool:
        ok = await self._hub.send_control(device_id, msg)
        if ok:
            self._adapter._observe_control(device_id, msg)
        return ok

    async def send_pcm(self, device_id: str, pcm: bytes) -> bool:
        ok = await self._hub.send_pcm(device_id, pcm)
        if ok:
            self._adapter._observe_pcm(device_id, len(pcm))
        return ok


class _NullWake:
    """Wake detector used when the wake models could not be loaded: tap-to-talk still works."""

    def feed(self, frame: bytes) -> float | None:
        return None


# --- the adapter ---------------------------------------------------------------------------------


class HousemateAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        if not isinstance(getattr(config, "extra", None), dict):
            config.extra = {}
        self.cfg = HousemateConfig.from_extra(getattr(config, "extra", None))
        turn_timeout = self.cfg.turn_timeout_s
        if "turn_timeout_s" not in config.extra:
            turn_timeout = max(turn_timeout, _hermes_approval_timeout() + TURN_TIMEOUT_MARGIN_S)
        self.router = SessionRouter(self.cfg.conversation, turn_timeout_s=turn_timeout)
        self._dispatching: set[str] = set()  # chats whose next question waits for Hermes to go idle
        self.arbiter = WakeArbiter()
        # What the Desktop Rooms page shows; every change pokes the feed (plugin event to Desktop).
        self.desk = DeskLog(on_change=self._desk_poke)
        self.arbiter.on_decide(self.desk.wake)
        self._feed: DeskFeed | None = None
        self._persist_mode: Callable[[str], Optional[str]] = persist_mode
        self.store: Any = None
        self.registry: Any = None
        self.hub: Any = None
        self.timers: Any = None
        self.announcer: Any = None
        self.services: Any = None
        self.control: Any = None
        self.models: Any = None
        self.urls: list[str] = []
        self.fingerprint: str | None = None
        self.listeners: dict[str, Any] = {}
        self.book = answers.BOOK
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streams: dict[str, _Stream] = {}
        self._playing: dict[str, _Playing] = {}
        self._played_until: dict[str, float] = {}  # device -> wall clock when its last sound ended
        self._announced: deque[tuple[float, str]] = deque(maxlen=32)  # (monotonic, normalised text)
        self._captions: dict[str, deque] = {}
        self._prompts: dict[str, _Prompt] = {}
        self._idle_timers: dict[str, asyncio.TimerHandle] = {}
        self._intercom: dict[str, tuple[str, float]] = {}  # device -> (target room, until): next utterance plays there
        self._states: dict[str, str] = {}  # last state asked of each device
        self._timings: dict[str, dict[str, float]] = {}
        self._turn_log: deque[dict] = deque(maxlen=TURN_LOG_SIZE)  # finished turns, for the bench
        self._timing_meta: dict[str, tuple[str, str]] = {}  # message id -> (room, trigger kind)
        self._bg: set[asyncio.Task] = set()
        self._timer_task: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._db_conn: Any = None
        # Seams for tests; the real ones are Hermes's public helpers.
        self._cache_audio: Callable[..., Any] = cache_audio_from_bytes_async
        self._resolve_approval: Callable[[str, str, Optional[str]], int] | None = None
        self._resolve_clarify: Callable[[str, str], bool] | None = None
        self._tts_files: Callable[[str], Any] | None = None
        self._decode: Callable[..., Any] = decode_to_pcm

    # ======================================================================== lifecycle

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        for warning in self.cfg.warnings:
            log.warning("housemate: %s", warning)
        from plugins.plugin_storage import plugin_data_dir, plugin_db

        from .core.control import ControlPortInUse, ControlServer
        from .core.hub import Hub, HubPortInUse
        from .core.rooms import RoomRegistry
        from .core.store import HousemateStore
        from .core.tls import ensure_self_signed_cert, hub_urls, lan_hosts

        self._loop = asyncio.get_running_loop()
        data_dir = Path(plugin_data_dir(PLATFORM_NAME))
        try:
            self._db_conn = plugin_db(PLATFORM_NAME)
            store = HousemateStore(self._db_conn)
            registry = RoomRegistry()
            cert = key = None
            if self.cfg.tls:
                advertised = [self.cfg.advertise_host] if self.cfg.advertise_host else []
                cert, key, self.fingerprint = await asyncio.to_thread(
                    ensure_self_signed_cert, data_dir, lan_hosts(advertised))
            hub = Hub(store, registry, host=self.cfg.bind_host, port=self.cfg.port, cert=cert, key=key,
                      web_root=WEB_ROOT, tls=self.cfg.tls)
            hub.info = self._satellite_info
            await hub.start()
        except HubPortInUse as exc:
            log.error("housemate: %s", exc)
            self._set_fatal_error("port_in_use", str(exc), retryable=False)
            return False
        except Exception as exc:
            log.exception("housemate: the hub could not start")
            self._set_fatal_error("hub_failed", f"housemate hub could not start: {exc}", retryable=True)
            return False

        scheme = "https" if self.cfg.tls else "http"
        self.urls = hub_urls(scheme, hub.port, self.cfg.advertise_host)
        self._wire(store, registry, hub)
        self._seed_rooms()
        self._refresh_allowlist()

        from .core.control import HubServices, register_local_services

        self.services = HubServices(store, registry, hub, self.timers, self.announcer, list(self.urls), self.fingerprint,
                                    timings=self.recent_timings, desk=self.desk, mode=lambda: self.router.mode,
                                    set_mode=self._set_mode)
        register_local_services(self.services, self._loop)
        self._feed = DeskFeed(self._loop, lambda: desk_state(self.services))
        self.desk.info["wake_word"] = self.cfg.wake_model
        self.control = ControlServer(self.services, data_dir / "control.token", port=self.cfg.control_port)
        try:
            await self.control.start()
        except ControlPortInUse as exc:
            log.warning("housemate: %s (agent tools still work inside the gateway)", exc)
            self.control = None
        self._timer_task = asyncio.create_task(self.timers.run(), name="housemate-timers")
        self._watchdog = asyncio.create_task(self._watch(), name="housemate-watchdog")
        self._spawn(self._load_models(data_dir))
        self._mark_connected(listener_base=hub.url)
        log.info("housemate: hub on %s (open it on a phone or laptop to add a room speaker)",
                 ", ".join(self.urls[:3]))
        return True

    def _wire(self, store: Any, registry: Any, hub: Any, *, timers: Any = None, announcer: Any = None) -> None:
        """Connect the hub's callbacks and build the timer and announcement services."""
        from .core.timers import TimerService

        self.store, self.registry, self.hub = store, registry, hub
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        hub.on_audio(self._on_audio)
        hub.on_control(self._on_control)
        if hasattr(hub, "on_disconnect"):
            hub.on_disconnect(self._on_disconnect)
        if hasattr(hub, "on_pairing_change"):
            hub.on_pairing_change(self._desk_poke)
        if hasattr(registry, "on_change"):
            registry.on_change(lambda _rooms: self._desk_poke())
        self.announcer = announcer or Announcer(_HubTap(hub, self), registry, self._tts, on_speak=self._note_announced)
        self.timers = timers or TimerService(
            store,
            time.time,
            self._on_timer_ring,
            play_chime=lambda room: self.announcer.chime(room, "timer"),
            speak=self.announcer.say,
            broadcast_timers=self._broadcast_timers,
        )

    def _seed_rooms(self) -> None:
        for name in self.cfg.rooms:
            with contextlib.suppress(Exception):
                self.store.ensure_room(name)
        for room, group in self.cfg.earshot_groups.items():
            with contextlib.suppress(Exception):
                self.store.ensure_room(room)
                self.store.set_earshot_group(slugify(room), group)

    async def _load_models(self, data_dir: Path) -> None:
        from .core.models import load_audio_models

        started = time.monotonic()
        try:
            self.models = await asyncio.to_thread(
                load_audio_models, data_dir, wake_model=self.cfg.wake_model, vad=self.cfg.vad)
            log.info("housemate: wake word %s ready in %.1f s", self.models.wake_model, time.monotonic() - started)
            self.desk.info["wake_word"] = self.models.wake_model
            self._desk_poke()
        except Exception as exc:
            log.error("housemate: wake-word models unavailable (%s); tap-to-talk only", exc)
            self.models = None
        for device_id in list(self.hub.online_devices()):
            await self._ensure_listener(device_id)
        if self.cfg.prewarm:
            await self._prewarm()

    async def _prewarm(self) -> None:
        """Warm Hermes's speech models before the first question; a cold load adds seconds to the
        first answer after a restart. Never a turn in the household conversation."""
        await self._prewarm_stt()  # hands a clip to Hermes and returns; the model loads meanwhile
        await self._prewarm_tts()

    async def _prewarm_tts(self) -> None:
        """One short synthesis through the text_to_speech tool, the path replies take: a local voice
        (Piper loads in ~5 s) is in memory afterwards. The file is thrown away, nothing is played."""
        started = time.monotonic()
        try:
            paths = await self._tts("Ready.")
        except Exception:
            log.debug("housemate: text-to-speech warm-up failed", exc_info=True)
            return
        for path in paths:
            with contextlib.suppress(OSError):
                Path(path).unlink()
        log.info("housemate: text-to-speech warmed up in %.1f s", time.monotonic() - started)

    async def _prewarm_stt(self) -> None:
        """A short, quiet clip goes to the answer side chat under a one-time marker; whatever it
        transcribes to is ignored and the side turn is vetoed."""
        try:
            device_id = next((d.device_id for d in self.store.list_devices() if d.paired), None)
        except Exception:
            device_id = None
        if device_id is None:
            return  # nobody may talk yet; the first real question warms it up
        token = answers.new_token()
        self.book.add(answers.PendingAnswer(
            token=token, kind="warmup", device_id="~warmup", room="", expires_at=time.monotonic() + 120.0,
            handle=lambda transcript: "speech-to-text warmed up", key="warmup"))
        # 0.8 s of a quiet 220 Hz tone: enough for a real decode, nothing anyone would say.
        import math

        samples = (int(300 * math.sin(2 * math.pi * 220 * i / 16_000)) for i in range(12_800))
        pcm = b"".join(v.to_bytes(2, "little", signed=True) for v in samples)
        try:
            path = await self._cache_audio(pcm_to_wav(pcm), ".wav")
            source = self.build_source(chat_id=SIDE_CHAT, chat_name="Housemate answers", chat_type="dm",
                                       user_id=device_id, user_name="Housemate")
            await self.handle_message(MessageEvent(
                text="", message_type=MessageType.VOICE, source=source, message_id=uuid.uuid4().hex,
                media_urls=[str(path)], media_types=["audio/wav"], channel_context=answers.make_marker(token)))
            log.info("housemate: warming up speech-to-text")
        except Exception:
            log.debug("housemate: warm-up skipped", exc_info=True)

    async def disconnect(self) -> None:
        from .core.control import unregister_local_services

        for task in (self._watchdog, self._timer_task):
            if task is not None:
                task.cancel()
        if self._feed is not None:
            self._feed.close()
            self._feed = None
        if self.timers is not None:
            with contextlib.suppress(Exception):
                self.timers.close()
        if self.services is not None:
            unregister_local_services(self.services)
        if self.control is not None:
            with contextlib.suppress(Exception):
                await self.control.stop()
            self.control = None
        for stream in list(self._streams.values()):
            stream.playout.abort()
        for listener in list(self.listeners.values()):
            with contextlib.suppress(Exception):
                listener.close()
        self.listeners.clear()
        for handle in list(self._idle_timers.values()):
            handle.cancel()
        for task in list(self._bg):
            task.cancel()
        if self.hub is not None:
            with contextlib.suppress(Exception):
                await self.hub.stop()
        if self._db_conn is not None:
            with contextlib.suppress(Exception):
                self._db_conn.close()
            self._db_conn = None
        self._mark_disconnected()

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        if chat_id == HOUSEHOLD_CHAT:
            return {"name": "Household", "type": "dm"}
        if chat_id == SIDE_CHAT:
            return {"name": "Housemate answers", "type": "dm"}
        return {"name": self._room_name(chat_id), "type": "dm"}

    # ======================================================================== access

    def _refresh_allowlist(self) -> None:
        """Paired devices are the allowlist. Hermes reads ``config.extra["allow_from"]`` live (at
        intake and again at admission), so pairing a phone grants access and revoking it takes
        access away at once. ``HOUSEMATE_ALLOWED_DEVICES``, when set, replaces this list."""
        if self.store is None:
            return
        try:
            paired = sorted(d.device_id for d in self.store.list_devices() if d.paired)
        except Exception:
            log.exception("housemate: reading paired devices failed")
            return
        self.config.extra["allow_from"] = paired

    def _device_allowed(self, device_id: str | None) -> bool:
        """May this device talk to Hermes (and so answer its prompts)? The same order Hermes's own
        check uses: ``HOUSEMATE_ALLOW_ALL_DEVICES``, else the env allowlists when any is set
        (``HOUSEMATE_ALLOWED_DEVICES``, ``GATEWAY_ALLOWED_USERS``), else every paired device."""
        if not device_id or self.store is None:
            return False
        try:
            device = self.store.get_device(device_id)
        except Exception:
            return False
        if device is None or not device.paired:
            return False
        if _truthy(_secret("HOUSEMATE_ALLOW_ALL_DEVICES")):
            return True
        listed = _id_set(_secret("HOUSEMATE_ALLOWED_DEVICES")) | _id_set(_secret("GATEWAY_ALLOWED_USERS"))
        if listed:
            return "*" in listed or device_id in listed
        return True

    def _allowed_in(self, room: str | None) -> list[str]:
        """Online devices in ``room`` that may talk to Hermes."""
        if not room or self.registry is None:
            return []
        return sorted(c.device_id for c in self.registry.devices_in(room) if self._device_allowed(c.device_id))

    def _may_tap(self, device_id: str, prompt: "_Prompt") -> bool:
        """A tap answers a prompt only from a device that was shown it, is still in its room and is
        still allowed (a paired tablet outside the allowlist, or in another room, cannot approve)."""
        return (prompt.presented and device_id in prompt.shown_to
                and self._device_room(device_id) == prompt.room and self._device_allowed(device_id))

    # ======================================================================== voice mode

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        return self.cfg.speak_replies and chat_id != SIDE_CHAT

    # ======================================================================== inbound audio

    def _on_audio(self, device_id: str, frame: bytes) -> None:
        listener = self.listeners.get(device_id)
        if listener is not None:
            listener.feed(frame)

    def _on_wake(self, device_id: str) -> None:
        """A device won a wake round: stop what it plays (barge-in), stop a ring, listen."""
        self._intercom.pop(device_id, None)
        room = self._device_room(device_id)
        if room and self.timers is not None and self.timers.is_ringing(room):
            self.timers.stop_ringing(room)
            self.book.drop(kind="stop", room=room)
        self._barge_in(device_id)
        self._set_state(device_id, "listening")

    def _barge_in(self, device_id: str) -> None:
        """Silence the device now (the caller sets the next state)."""
        stream = self._streams.get(device_id)
        if stream is not None and not stream.playout.ended:
            stream.playout.abort()  # its pump sends play.abort
        playing = self._playing.get(device_id)
        if playing is None:
            return
        if playing.kind in ("say", "ann"):
            # announcer audio went out in one piece: tell the device to drop it
            self._spawn(self._send(device_id, {"type": "play.abort", "stream_id": playing.stream_id}))
        self._playback_finished(device_id, playing.stream_id, aborted=True, set_state=False)

    def _on_utterance(self, device_id: str, utt: Any, kind: str) -> None:
        captured = time.time()
        room = self._device_room(device_id)
        if room is None:
            log.debug("housemate: utterance from %s ignored (no room yet)", device_id)
            return
        self._cancel_idle(device_id)
        intercom = self._intercom.pop(device_id, None)
        if intercom is not None and kind == "followup" and time.monotonic() < intercom[1]:
            self._set_state(device_id, "thinking")
            self._spawn(self._deliver_intercom(device_id, room, intercom[0], bytes(utt.pcm)))
            return
        pending = self.book.for_device(device_id)
        if pending is not None and pending.kind in ("approval", "clarify") and self._heard_ourselves(utt):
            # Our own audio (a reply, a ring, an announcement in any room) may be in this clip: it
            # never answers a prompt. A ringing timer's "stop" still may.
            stop = self.book.for_device(device_id, kinds=("stop",))
            if stop is None:
                log.info("housemate: an answer from %s overlapped our own audio; not taken", room)
                self.book.drop(token=pending.token)
                self._reprompt(pending.key, overlapped=True)
                return
            pending = stop
        if pending is not None:
            self._set_state(device_id, "thinking")
            self._spawn(self._send_answer(device_id, room, utt, pending))
            return
        item = _Utterance(uuid.uuid4().hex, device_id, room, kind, bytes(utt.pcm),
                          float(getattr(utt, "ended_at", captured)), captured)
        self._timings[item.message_id] = {"endpoint": item.endpoint_at, "captured": captured}
        self._timing_meta[item.message_id] = (room, kind)
        chat_id = self.router.chat_id_for(room)
        self._set_state(device_id, "thinking")
        if self.router.household_busy(chat_id):
            self.router.enqueue(item, chat_id)
            log.info("housemate: %s spoke while a turn is running; queued (%d waiting)",
                     room, self.router.queued(chat_id))
            self._settle_dropped()
            return
        self.router.begin_turn(chat_id, room, item.message_id, device_id)
        self._spawn(self._dispatch(chat_id, item))

    async def _dispatch(self, chat_id: str, item: _Utterance) -> None:
        try:
            path = await self._cache_audio(pcm_to_wav(item.pcm), ".wav")
            room_name = self._room_name(item.room)
            device_name = self._device_name(item.device_id)
            source = self.build_source(
                chat_id=chat_id,
                chat_name="Household" if chat_id == HOUSEHOLD_CHAT else room_name,
                chat_type="dm",
                user_id=item.device_id,
                user_name=room_name,
            )
            event = MessageEvent(
                text="",
                message_type=MessageType.VOICE,
                source=source,
                message_id=item.message_id,
                media_urls=[str(path)],
                media_types=["audio/wav"],
                channel_prompt=self._room_prompt(room_name, device_name),
                channel_context=f"[room: {room_name}]",
            )
            self._dispatching.add(chat_id)
            try:
                idle = await self._wait_session_idle(event, SESSION_BUSY_MAX_WAIT_S)
            finally:
                self._dispatching.discard(chat_id)
            if not idle:
                # Handing it over now would interrupt Hermes's running turn (and cancel its approval).
                log.warning("housemate: Hermes is still busy; gave up the question from %s", item.room)
                self._set_state(item.device_id, "idle")
                await self._caption_room(item.room, "status", "Sorry, I couldn't get to that. Please ask again.")
                self._end_turn(chat_id, item.message_id)
                return
            self._mark(item.message_id, "dispatch")
            await self.handle_message(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: handing the utterance from %s to Hermes failed", item.room)
            self._set_state(item.device_id, "idle")
            self._end_turn(chat_id, item.message_id)

    def _room_prompt(self, room_name: str, device_name: str) -> str:
        where = f"The user is speaking from the {room_name}"
        if device_name and device_name.lower() != room_name.lower():
            where += f" (device: {device_name})"
        if self.cfg.conversation == "household":
            return where + ". This is one household conversation across rooms; your reply is spoken in that room."
        return where + "."

    async def _wait_session_idle(self, event: MessageEvent, timeout: float = SESSION_IDLE_WAIT_S) -> bool:
        """A queued utterance waits until Hermes has really finished the previous turn: handing
        it over while that session is still busy would interrupt the running turn. False when the
        session is still busy after ``timeout``."""
        try:
            key = self._event_session_key(event)
        except Exception:
            return True
        active = getattr(self, "_active_sessions", None)
        if not isinstance(active, dict) or key not in active:
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while key in active and loop.time() < deadline:
            await asyncio.sleep(0.05)
        if key in active:
            return False
        await asyncio.sleep(0.1)
        return True

    def _end_turn(self, chat_id: str, message_id: str | None = None) -> None:
        turn = self.router.active_turn(chat_id)
        if turn is None or (message_id is not None and turn.message_id != message_id):
            return
        self._log_timing(turn.message_id)
        nxt = self.router.on_turn_end(chat_id, message_id)
        self._settle_dropped()
        if nxt is not None:
            self.router.begin_turn(chat_id, nxt.room, nxt.message_id, nxt.device_id)
            log.info("housemate: answering the queued question from %s", nxt.room)
            self._spawn(self._dispatch(chat_id, nxt))

    def _settle_dropped(self) -> None:
        """Questions the queue gave up on, and turns that timed out, leave no room "thinking"."""
        dropped, expired = self.router.take_dropped()
        for item in dropped:
            log.info("housemate: dropped a queued question from %s", item.room)
            self._set_state(item.device_id, "idle")
            self._spawn(self._caption_room(item.room, "status", "Sorry, I couldn't get to that. Please ask again."))
        for turn in expired:
            if turn.device_id and turn.device_id not in self._playing:
                self._set_state(turn.device_id, "idle")

    async def on_processing_complete(self, event: MessageEvent, outcome: Any) -> None:
        """Backup turn-end signal (fires after delivery); the final ``send()`` usually came first."""
        with contextlib.suppress(Exception):
            await super().on_processing_complete(event, outcome)
        source = getattr(event, "source", None)
        chat_id = getattr(source, "chat_id", None)
        message_id = getattr(event, "message_id", None)
        if chat_id and message_id and chat_id != SIDE_CHAT:
            # Hermes is done with this turn, so every prompt it asked is settled, even one answered
            # on Desktop or cancelled by /stop: clear those cards so none blocks the room's next one.
            stale = [p for p in self._prompts.values() if p.turn_id == message_id]
            for prompt in stale:
                self._prompts.pop(prompt.id, None)
            for prompt in stale:
                self._retire_prompt(prompt, "")
            turn = self.router.active_turn(chat_id)
            if turn is not None and turn.message_id == message_id:
                if turn.device_id and turn.device_id not in self._playing:
                    self._set_state(turn.device_id, "idle")
                self._end_turn(chat_id, message_id)

    async def _watch(self) -> None:
        """Every few seconds: end turns that never finished, forget stale timing records."""
        while True:
            await asyncio.sleep(5.0)
            try:
                self._watch_step()
            except Exception:
                log.exception("housemate: watchdog step failed")

    def _watch_step(self) -> None:
        for chat_id, item in self.router.expire(hold=self._turn_held):
            log.warning("housemate: a turn in %s timed out; answering the next question", chat_id)
            self.router.begin_turn(chat_id, item.room, item.message_id, item.device_id)
            self._spawn(self._dispatch(chat_id, item))
        self._settle_dropped()
        cutoff = time.time() - 900
        for mid in [m for m, t in self._timings.items() if t.get("captured", 0) < cutoff]:
            self._timings.pop(mid, None)
            self._timing_meta.pop(mid, None)

    def _turn_held(self, chat_id: str) -> bool:
        """The chat's turn waits on the user (an open approval or clarify prompt) or its question
        waits for Hermes to go idle: the safety timeout must not hand the next question over."""
        return chat_id in self._dispatching or any(p.chat_id == chat_id for p in self._prompts.values())

    # ======================================================================== device control

    async def _on_control(self, device_id: str, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "hello":
            self._spawn(self._on_hello(device_id))
        elif mtype == "room.set":
            self._spawn(self._ensure_listener(device_id))
            self._spawn(self._push_timers(device_id))
        elif mtype == "ptt":
            listener = self.listeners.get(device_id)
            self._intercom.pop(device_id, None)
            if msg.get("on"):
                self._barge_in(device_id)
                self._set_state(device_id, "listening")
            if listener is not None:
                listener.set_ptt(bool(msg.get("on")))
        elif mtype == "mute":
            listener = self.listeners.get(device_id)
            if listener is not None:
                listener.set_muted(bool(msg.get("on")))
        elif mtype == "playback.done":
            self._playback_finished(device_id, str(msg.get("stream_id")))
        elif mtype == "ui.action":
            self._on_ui_action(device_id, msg)

    async def _on_hello(self, device_id: str) -> None:
        self._refresh_allowlist()  # a device that just paired may talk now
        await self._ensure_listener(device_id)
        room = self._device_room(device_id)
        ringing = bool(room and self.timers is not None and self.timers.is_ringing(room))
        await self._send_state(device_id, "ringing" if ringing else "idle")
        await self._push_timers(device_id)
        if room and self._device_allowed(device_id):
            for prompt in list(self._prompts.values()):
                if prompt.room == room and prompt.presented and await self._send(device_id, self._prompt_msg(prompt)):
                    prompt.shown_to.add(device_id)
        queue = self._captions.pop(device_id, None)
        if queue:
            now = time.monotonic()
            for at, role, text in queue:
                if now - at <= CAPTION_TTL_S:
                    await self._send(device_id, {"type": "caption", "role": role, "text": text})

    async def _ensure_listener(self, device_id: str) -> None:
        room = self._device_room(device_id)
        if room is None or self.hub is None or device_id not in self.hub.online_devices():
            return
        group = self._earshot_group(room)
        listener = self.listeners.get(device_id)
        if listener is not None:
            listener.set_room(room, group)
            return
        from .core.listener import DeviceListener

        if self.models is not None:
            wake = await asyncio.to_thread(self.models.new_wake)
            endpointer = await asyncio.to_thread(self.models.new_endpointer, end_silence_ms=self.cfg.end_silence_ms)
        else:
            from .core.endpointer import Endpointer
            from .core.vad import EnergyVad

            wake, endpointer = _NullWake(), Endpointer(EnergyVad(), end_silence_ms=self.cfg.end_silence_ms)
        if device_id in self.listeners or device_id not in self.hub.online_devices():
            return  # raced with another hello, or the device left while models loaded
        listener = DeviceListener(device_id, wake, endpointer, self.arbiter, self.cfg.wake_threshold,
                                  room=room, group=group, loop=self._loop)
        listener.on_utterance(self._on_utterance)
        listener.on_wake(self._on_wake)
        self.listeners[device_id] = listener

    def _on_disconnect(self, device_id: str) -> None:
        self._refresh_allowlist()  # a revoked device loses access at once
        self._states.pop(device_id, None)
        self._intercom.pop(device_id, None)
        listener = self.listeners.pop(device_id, None)
        if listener is not None:
            listener.close()
        stream = self._streams.get(device_id)
        if stream is not None:
            stream.playout.abort()
        playing = self._playing.pop(device_id, None)
        if playing is not None and playing.end_timer is not None:
            playing.end_timer.cancel()
        self._cancel_idle(device_id)
        self.book.drop(device_id=device_id, kind="stop")

    def _on_ui_action(self, device_id: str, msg: dict) -> None:
        kind = str(msg.get("kind") or "")
        ident = str(msg.get("id") or "")
        value = msg.get("value")
        if kind == "stop_timer":
            room = self._device_room(device_id)
            if room and self.timers is not None:
                self.timers.stop_ringing(room)
            return
        if kind == "intercom":
            self._start_intercom(device_id, ident or str(value or ""))
            return
        if kind in ("approve", "deny"):
            kind, value = "prompt", kind
        if kind != "prompt":
            return
        prompt = self._prompts.get(ident)
        if prompt is None:
            self._spawn(self._send(device_id, {"type": "prompt.clear", "id": ident or "-"}))
            return
        if not self._may_tap(device_id, prompt):
            log.warning("housemate: ignored a tap on %s %s from %s (not shown there, not in %s, or not allowed)",
                        prompt.kind, prompt.id, device_id, prompt.room)
            return
        log.info("housemate: %s %s answered by a tap on %s", prompt.kind, prompt.id, device_id)
        if prompt.kind == "approval":
            choice = parse_yes_no(str(value or ""))
            if choice is None:
                return  # an unknown button never approves
            self._apply_approval(prompt, choice, via="tap")
        else:
            label = str(value or "")
            pick = label if label in prompt.choices else parse_choice(label, prompt.choices)
            if pick is None and isinstance(value, int) and 1 <= value <= len(prompt.choices):
                pick = prompt.choices[value - 1]
            if pick is not None:
                self._apply_clarify(prompt, pick, via="tap")

    def _satellite_info(self) -> dict:
        """What a satellite shows besides its room (sent with ``paired``): the wake phrase of the model
        that is really loaded, the conversation mode, this computer's name."""
        model = self.models.wake_model if self.models is not None else self.cfg.wake_model
        return {"wake": wake_phrase(model), "conversation": getattr(self.router, "mode", self.cfg.conversation),
                "hub_name": socket.gethostname()[:64]}

    # ======================================================================== intercom

    def _start_intercom(self, device_id: str, target: str) -> None:
        """A tap on another room on the satellite: what this device hears next (no wake word, a few
        seconds to start) is played in that room as a voice message. Nothing goes to Hermes."""
        room = self._device_room(device_id)
        if room is None:
            return
        found = None
        with contextlib.suppress(Exception):
            found = self.store.get_room(target) if target else None
        if found is None or found.slug == room:
            self._spawn(self._send(device_id, {"type": "caption", "role": "status",
                                               "text": f"There is no other room called {target[:40] or 'that'}."}))
            return
        listener = self.listeners.get(device_id)
        if listener is None:
            self._spawn(self._send(device_id, {"type": "caption", "role": "status",
                                               "text": "Still starting up. Try again in a moment."}))
            return
        self._intercom[device_id] = (found.slug, time.monotonic() + INTERCOM_TTL_S)
        self._barge_in(device_id)
        listener.open_followup(INTERCOM_WINDOW_S)
        self._set_state(device_id, "listening")
        self._idle_later(device_id, INTERCOM_WINDOW_S + 1.0)

    async def _deliver_intercom(self, device_id: str, room: str, target: str, pcm: bytes) -> None:
        to_name, from_name = self._room_name(target), self._room_name(room)
        sent: list[str] = []
        try:
            sent = await self.announcer.voice_message(target, pcm, MIC_RATE, from_name)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: playing the voice message in %s failed", target)
        log.info("housemate: intercom %s -> %s (%.1f s, %d device(s))", room, target,
                 len(pcm) / (2 * MIC_RATE), len(sent))
        note = f"Sent to the {to_name}." if sent else f"No speaker is on in the {to_name}."
        await self._send(device_id, {"type": "caption", "role": "status", "text": note})
        if device_id not in self._playing:
            self._set_state(device_id, "idle")

    # ======================================================================== outbound text

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[dict[str, Any]] = None) -> SendResult:
        message_id = "hm-" + uuid.uuid4().hex[:12]
        try:
            await self._deliver_text(chat_id, str(content or ""), reply_to)
        except Exception:
            log.exception("housemate: delivering a message to %s failed", chat_id)
        return SendResult(success=True, message_id=message_id)

    async def _deliver_text(self, chat_id: str, content: str, reply_to: Optional[str]) -> None:
        text = content.strip()
        if chat_id == SIDE_CHAT or not text:
            return  # the spoken-answer side chat never talks back
        if text.startswith(_SWALLOWED_PREFIXES):
            return
        if text.startswith(_STATUS_PREFIXES):
            # Gateway housekeeping ("💾 Self-improvement review: …", "⚡ Interrupting …"): show it on
            # the screen of the room that last spoke, never say it out loud, never end a turn on it.
            turn = self.router.active_turn(chat_id) or self.router.latest_turn(chat_id)
            room = turn.room if turn is not None else self._chat_room(chat_id)
            if room:
                await self._caption_room(room, "status", _plain(text))
            return
        if text.startswith(_TRANSCRIPT_PREFIX):
            said = text[len(_TRANSCRIPT_PREFIX):].strip(_QUOTES)
            turn = self.router.active_turn(chat_id) or self.router.latest_turn(chat_id)
            if turn is not None:
                self._mark(turn.message_id, "transcript")
                await self._caption_room(turn.room, "user", said)
            return
        turn = self.router.turn_for(reply_to)
        if turn is not None and turn.chat_id == chat_id:
            self._mark(turn.message_id, "reply_text")
            await self._caption_room(turn.room, "assistant", text)
            if not turn.audio and not turn.stream_open and self.cfg.speak_replies and turn.device_id:
                turn.audio = True  # once per turn, even if Hermes sends the answer in parts
                self._spawn(self._speak_reply(turn.device_id, text, turn.message_id))
            elif turn.device_id and turn.device_id not in self._playing:
                self._set_state(turn.device_id, "idle")
            self._end_turn(chat_id, turn.message_id)
            return
        active = self.router.active_turn(chat_id)
        if active is not None:  # a notice in the middle of a turn: show it, don't speak it
            await self._caption_room(active.room, "assistant", text)
            return
        recent = self.router.latest_turn(chat_id)
        if recent is not None and self._chat_room(chat_id) is None and recent.age_s() < RECENT_TURN_S:
            # Right after a turn (e.g. Hermes reporting that it failed): tell the room that asked.
            await self._caption_room(recent.room, "assistant", text)
            if self.cfg.speak_replies and recent.device_id:
                self._spawn(self._speak_reply(recent.device_id, text))
            return
        # Nobody asked: a cron delivery, a reminder or a message from another platform.
        rooms = self._delivery_rooms(chat_id)
        for room in rooms:
            await self._caption_room(room, "announcement", text)
        if self.cfg.speak_replies and rooms:
            self._spawn(self._speak_rooms(rooms, text))

    async def edit_message(self, chat_id: str, message_id: str, content: str, *, finalize: bool = False) -> SendResult:
        prompt = self._prompts.get(message_id)
        if prompt is None:
            return SendResult(success=False, error="Not supported")
        # Hermes edits the approval card when it times out: clear it, say why.
        self._retire_prompt(prompt, _plain(content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        turn = self.router.active_turn(chat_id)
        if turn is not None and turn.device_id and turn.device_id not in self._playing:
            self._set_state(turn.device_id, "thinking")

    async def stop_typing(self, chat_id: str) -> None:
        return None

    # ======================================================================== outbound audio

    def supports_streaming_tts(self, chat_id: str, audio_format: AudioFormat) -> bool:
        if chat_id == SIDE_CHAT or not self.cfg.speak_replies:
            return False
        if getattr(audio_format, "channels", 1) != 1 or getattr(audio_format, "sample_width", 2) != 2:
            return False
        return self._reply_device(chat_id) is not None

    async def begin_streaming_tts(self, chat_id: str, audio_format: AudioFormat,
                                  metadata: Optional[dict[str, Any]] = None) -> Optional[StreamingTTSHandle]:
        try:
            if not self.supports_streaming_tts(chat_id, audio_format):
                return None
            device_id = self._reply_device(chat_id)
            turn = self.router.active_turn(chat_id)
            stream = await self._start_stream(device_id, int(audio_format.sample_rate), "rep")
            if stream is None:
                return None
            handle = _ReplyHandle(chat_id, audio_format, stream)
            if turn is not None:
                turn.stream_open = True
                handle.turn = turn
                mid = turn.message_id
                self._mark(mid, "tts_begin")
                stream.on_first_pcm = lambda: self._mark(mid, "first_pcm_out")
            return handle
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: could not open the audio stream; Hermes falls back to whole files")
            return None

    async def write_streaming_tts(self, handle: StreamingTTSHandle, chunk: bytes) -> None:
        if not isinstance(handle, _ReplyHandle) or handle.aborted or not chunk:
            return
        try:
            pcm = handle.realign.feed(chunk)
            if handle.chunks == 0:
                turn = self.router.active_turn(handle.chat_id)
                if turn is not None:
                    self._mark(turn.message_id, "first_pcm_in")
            handle.chunks += 1
            if pcm:
                handle.stream.playout.write(pcm)
                handle.audible = True
                if handle.turn is not None:
                    handle.turn.audio = True
        except Exception:
            log.exception("housemate: streaming audio failed")

    async def finish_streaming_tts(self, handle: StreamingTTSHandle, *, interrupted: bool = False) -> None:
        if not isinstance(handle, _ReplyHandle):
            return
        handle.realign.flush()
        self._close_turn_stream(handle)
        if interrupted or not handle.audible:
            handle.stream.playout.abort()  # nothing was heard: no follow-up window after silence
        else:
            handle.stream.playout.finish()

    async def abort_streaming_tts(self, handle: StreamingTTSHandle, error: Optional[str] = None) -> None:
        if not isinstance(handle, _ReplyHandle):
            return
        handle.aborted = True
        self._close_turn_stream(handle)
        reason = (error or "").lower()
        if error and handle.audible and "interrupt" not in reason and "cancel" not in reason:
            # Synthesis gave up (e.g. the finalisation timeout on a long answer): play what came.
            handle.stream.playout.finish()
        else:
            handle.stream.playout.abort()

    @staticmethod
    def _close_turn_stream(handle: "_ReplyHandle") -> None:
        if handle.turn is not None:
            handle.turn.stream_open = False

    async def send_voice(self, chat_id: str, audio_path: str, caption: Optional[str] = None,
                         reply_to: Optional[str] = None, metadata: Optional[dict[str, Any]] = None,
                         **kwargs: Any) -> SendResult:
        """Whole-file audio (TTS providers without streaming, e.g. Edge): decode and play it."""
        message_id = "hm-" + uuid.uuid4().hex[:12]
        if chat_id == SIDE_CHAT:
            return SendResult(success=True, message_id=message_id)
        turn = self.router.turn_for(reply_to) or self.router.active_turn(chat_id)
        device_id = self._reply_device(chat_id, reply_to)
        try:
            if device_id is not None:
                if turn is not None:
                    self._mark(turn.message_id, "tts_begin")  # the whole file is ready
                pcm = await self._decode(Path(audio_path), PLAY_RATE)
                stream = await self._start_stream(device_id, PLAY_RATE, "rep")
                if stream is not None:
                    if turn is not None:
                        turn.audio = True
                        mid = turn.message_id
                        stream.on_first_pcm = lambda: self._mark(mid, "first_pcm_out")
                    stream.playout.write(pcm[: len(pcm) - len(pcm) % 2])
                    stream.playout.finish()
            if caption:
                room = turn.room if turn is not None else (self._device_room(device_id) if device_id else None)
                if room:
                    await self._caption_room(room, "assistant", _plain(caption))
        except Exception as exc:
            log.warning("housemate: playing %s failed: %s", Path(audio_path).name, exc)
            return SendResult(success=False, error=str(exc))
        return SendResult(success=True, message_id=message_id)

    async def _start_stream(self, device_id: str | None, rate: int, kind: str, ref: str | None = None) -> _Stream | None:
        if not device_id or self.hub is None:
            return None
        old = self._streams.get(device_id)
        if old is not None and not old.playout.ended:
            old.playout.abort()  # one voice per device: the newest wins
        stream_id = f"{kind}-{uuid.uuid4().hex[:12]}"
        if not await self.hub.send_control(device_id, {"type": "play.begin", "stream_id": stream_id, "sample_rate": rate}):
            return None
        raced = self._streams.get(device_id)
        if raced is not None and not raced.playout.ended:
            raced.playout.abort()  # another stream started while play.begin was in flight
        hub = self.hub
        stream: _Stream

        async def send_pcm(pcm: bytes) -> bool:
            ok = await hub.send_pcm(device_id, pcm)
            if ok and stream.on_first_pcm is not None:
                callback, stream.on_first_pcm = stream.on_first_pcm, None
                callback()
            return ok

        stream = _Stream(stream_id, device_id, kind, rate, Playout(send_pcm, sample_rate=rate), ref=ref)
        self._streams[device_id] = stream
        self._playback_started(device_id, stream_id, kind, rate)
        stream.task = self._spawn(self._run_stream(stream))
        return stream

    async def _run_stream(self, stream: _Stream) -> None:
        result = "failed"
        try:
            result = await stream.playout.run()
        except asyncio.CancelledError:
            result = "aborted"
            raise
        finally:
            if self._streams.get(stream.device_id) is stream:
                del self._streams[stream.device_id]
            if result == "done":
                await self._send(stream.device_id, {"type": "play.end", "stream_id": stream.stream_id})
                self._expect_playback_end(stream.device_id, stream.stream_id, stream.playout)
            else:
                if result == "aborted":
                    await self._send(stream.device_id, {"type": "play.abort", "stream_id": stream.stream_id})
                self._playback_finished(stream.device_id, stream.stream_id, aborted=True)

    # --- what each device is playing (echo guard, follow-up windows, state) ----------------------

    def _observe_control(self, device_id: str, msg: dict) -> None:
        mtype = msg.get("type")
        sid = str(msg.get("stream_id") or "")
        if mtype == "play.begin" and not sid.startswith(("rep-", "ask-")):
            self._playback_started(device_id, sid, sid.split("-", 1)[0] or "ann", int(msg.get("sample_rate") or PLAY_RATE))
        elif mtype == "play.end" and device_id in self._playing and self._playing[device_id].stream_id == sid:
            playing = self._playing[device_id]
            remaining = playing.bytes / (2 * playing.rate) - (time.monotonic() - playing.started_at)
            self._arm_playback_timer(device_id, sid, max(0.0, remaining) + PLAYBACK_GRACE_S)
        elif mtype == "play.abort":
            self._playback_finished(device_id, sid, aborted=True)
        elif mtype == "chime":
            self._note_chime(device_id)

    def _note_chime(self, device_id: str) -> None:
        """A chime plays on the device itself; count it as our audio for the echo guard."""
        until = time.time() + CHIME_S
        self._played_until[device_id] = max(self._played_until.get(device_id, 0.0), until)

    def _heard_ourselves(self, utt: Any) -> bool:
        """Could this clip hold our own audio? True while any Housemate device plays (in any room:
        earshot groups are a hint, not a guarantee) or when the clip began less than
        ``ECHO_TAIL_S`` after the last sound one played."""
        if self._playing:
            return True
        started = float(getattr(utt, "started_at", 0.0) or 0.0)
        last = max(self._played_until.values(), default=float("-inf"))
        return started < last + ECHO_TAIL_S

    def _note_announced(self, text: str) -> None:
        """The announcer is about to say ``text`` (announcements, timer rings, deliveries)."""
        said = normalize(text)
        if said:
            self._announced.append((time.monotonic(), said))

    def _was_announced(self, transcript: str) -> bool:
        """Is this transcript something we said ourselves a little while ago? (any thread)"""
        said = normalize(transcript)
        if not said:
            return False
        cutoff = time.monotonic() - ANNOUNCED_TTL_S
        return any(at >= cutoff and f" {said} " in f" {text} " for at, text in list(self._announced))

    def _observe_pcm(self, device_id: str, n: int) -> None:
        playing = self._playing.get(device_id)
        if playing is not None:
            playing.bytes += n

    def _playback_started(self, device_id: str, stream_id: str, kind: str, rate: int) -> None:
        previous = self._playing.get(device_id)
        if previous is not None and previous.end_timer is not None:
            previous.end_timer.cancel()
        self._playing[device_id] = _Playing(stream_id, kind, rate, time.monotonic())
        self._cancel_idle(device_id)
        listener = self.listeners.get(device_id)
        if listener is not None:
            listener.set_playing(True)
        self._set_state(device_id, "speaking")

    def _expect_playback_end(self, device_id: str, stream_id: str, playout: Playout) -> None:
        """Our playout is sent; the device reports ``playback.done`` when it is heard. In case
        that report never comes, finish it ourselves a little after it must have ended."""
        end = playout.play_end_at
        if end is None:
            end = (playout.first_sent_at or time.monotonic()) + playout.sent_s
        remaining = end - time.monotonic()
        self._arm_playback_timer(device_id, stream_id, max(0.0, remaining) + PLAYBACK_GRACE_S)

    def _arm_playback_timer(self, device_id: str, stream_id: str, delay: float) -> None:
        playing = self._playing.get(device_id)
        if playing is None or playing.stream_id != stream_id or self._loop is None:
            return
        if playing.end_timer is not None:
            playing.end_timer.cancel()
        playing.end_timer = self._loop.call_later(delay, self._playback_finished, device_id, stream_id)

    def _playback_finished(self, device_id: str, stream_id: str, *, aborted: bool = False,
                           set_state: bool = True) -> None:
        playing = self._playing.get(device_id)
        if playing is None or playing.stream_id != stream_id:
            return
        del self._playing[device_id]
        self._played_until[device_id] = max(self._played_until.get(device_id, 0.0), time.time())
        if playing.end_timer is not None:
            playing.end_timer.cancel()
        listener = self.listeners.get(device_id)
        if listener is not None:
            listener.set_playing(False)
        room = self._device_room(device_id)
        ringing = bool(room and self.timers is not None and self.timers.is_ringing(room))
        if aborted:
            if set_state:
                self._set_state(device_id, "ringing" if ringing else "idle")
            return
        if playing.kind == "rep" and self.cfg.followup_s > 0:
            self._open_window(device_id, self.cfg.followup_s)
        elif playing.kind == "ask":
            if self.book.for_device(device_id, kinds=("approval", "clarify")) is not None:
                self._open_window(device_id, self.cfg.answer_window_s)
            else:
                self._set_state(device_id, "ringing" if ringing else "idle")  # tap-only prompt
        elif ringing:
            if listener is not None:
                listener.open_followup(15.0)  # "stop" without the wake word between ring cycles
            self._set_state(device_id, "ringing")
        else:
            self._set_state(device_id, "idle")

    def _open_window(self, device_id: str, seconds: float) -> None:
        listener = self.listeners.get(device_id)
        if listener is None:
            self._set_state(device_id, "idle")
            return
        listener.open_followup(seconds)
        self._set_state(device_id, "listening")
        self._idle_later(device_id, seconds + 1.0)

    def _idle_later(self, device_id: str, seconds: float) -> None:
        self._cancel_idle(device_id)
        if self._loop is None:
            return

        def go_idle() -> None:
            self._idle_timers.pop(device_id, None)
            if self._state_of(device_id) == "listening" and device_id not in self._playing:
                self._set_state(device_id, "idle")

        self._idle_timers[device_id] = self._loop.call_later(seconds, go_idle)

    def _cancel_idle(self, device_id: str) -> None:
        handle = self._idle_timers.pop(device_id, None)
        if handle is not None:
            handle.cancel()

    # ======================================================================== speaking ourselves

    async def _tts(self, text: str) -> list[Path]:
        if self._tts_files is not None:
            result = await self._tts_files(text)
            return result if isinstance(result, list) else [Path(result)]
        from .tools import dispatch_tool

        raw = await asyncio.to_thread(dispatch_tool, "text_to_speech", {"text": text})
        return tts_paths_from_result(raw)

    async def _speech_pcm(self, text: str) -> bytes:
        paths = await self._tts(text)
        pcm = b"".join([await self._decode(Path(p), PLAY_RATE) for p in paths])
        return pcm[: len(pcm) - len(pcm) % 2]

    async def _speak_reply(self, device_id: str, text: str, message_id: str | None = None) -> None:
        """Speak a turn's answer when no audio came with it (TTS off or failed upstream)."""
        try:
            pcm = await self._speech_pcm(text)
        except Exception as exc:
            log.warning("housemate: could not speak the reply: %s", exc)
            self._set_state(device_id, "idle")
            return
        stream = await self._start_stream(device_id, PLAY_RATE, "rep")
        if stream is not None:
            stream.playout.write(pcm)
            stream.playout.finish()

    async def _speak_rooms(self, rooms: list[str], text: str) -> None:
        try:
            await self.announcer.say(rooms, text)
        except Exception as exc:
            log.warning("housemate: could not speak the delivery: %s", exc)

    # ======================================================================== approvals & clarify

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        chat_id = prompt.chat_id
        turn = self.router.active_turn(chat_id) or self.router.latest_turn(chat_id)
        room = turn.room if turn is not None else self._chat_room(chat_id)
        device_id = self._reply_device(chat_id)
        card_id = prompt.request_id or f"appr-{uuid.uuid4().hex[:12]}"
        what = " ".join(str(prompt.description or "").split()) or "run a command"
        flagged = bool(getattr(prompt, "smart_denied", False))
        if flagged:
            # Hermes's smart approval flagged it: say so, and take the answer from a tap only.
            question = f"Careful: Hermes's safety check flagged a command: {what}. Tap approve or deny on the screen."
        else:
            question = f"I need your OK to run a command: {what}. Say approve, or deny."
        entry = _Prompt("approval", card_id, chat_id, prompt.session_key, room, device_id, question,
                        ["Approve", "Deny"], request_id=prompt.request_id,
                        title="Flagged command: run it?" if flagged else "Run a command?",
                        detail=what, command=" ".join(str(prompt.command or "").split())[:400],
                        spoken=question, smart_denied=flagged, turn_id=turn.message_id if turn else None)
        if not await self._open_prompt(entry):
            log.warning("housemate: no device online in %s for the approval; it will time out (deny)", room)
            return SendResult(success=False, error=f"no Housemate device is online in {room}")
        return SendResult(success=True, message_id=card_id)

    async def send_clarify(self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
                           session_key: str, metadata: Optional[dict[str, Any]] = None) -> SendResult:
        turn = self.router.active_turn(chat_id) or self.router.latest_turn(chat_id)
        room = turn.room if turn is not None else self._chat_room(chat_id)
        device_id = self._reply_device(chat_id)
        labels = [str(c) for c in (choices or []) if str(c).strip()]
        text, skippable = _split_skip_hint(question)
        entry = _Prompt("clarify", clarify_id, chat_id, session_key, room, device_id, _plain(text), labels,
                        skippable=skippable, turn_id=turn.message_id if turn else None)
        spoken = entry.question
        if labels and len(labels) <= 6:
            spoken += " " + _read_choices(labels) + "."
        if skippable:
            spoken += " Or say skip."
        entry.spoken = spoken
        if not await self._open_prompt(entry):
            return SendResult(success=False, error=f"no Housemate device is online in {room}")
        return SendResult(success=True, message_id=clarify_id)

    async def _open_prompt(self, prompt: _Prompt) -> bool:
        """One prompt per room at a time: the first open one is on the screens, spoken and armed;
        later ones wait their turn (a spoken "approve" answers what was asked, never a newer card).
        False when no allowed device is online in the room."""
        if not self._allowed_in(prompt.room):
            return False
        self._prompts[prompt.id] = prompt
        if self._room_head(prompt.room) is not prompt:
            log.info("housemate: %s %s waits behind the open prompt in %s", prompt.kind, prompt.id, prompt.room)
            return True
        if await self._present(prompt):
            return True
        self._prompts.pop(prompt.id, None)
        self._promote(prompt.room)  # one may have queued behind it while the card was being sent
        return False

    def _room_head(self, room: str | None) -> _Prompt | None:
        return next((p for p in self._prompts.values() if p.room == room), None)

    async def _present(self, prompt: _Prompt) -> bool:
        """Show the card in its room, then speak it on one device and arm that device's answer."""
        prompt.presented = True
        if not await self._show_prompt(prompt):
            return False
        if prompt.device_id not in prompt.shown_to:
            prompt.device_id = min(prompt.shown_to) if prompt.shown_to else None
        if prompt.device_id is not None and prompt.id in self._prompts:
            if not prompt.smart_denied:
                self._arm_answer(prompt)
            self._spawn(self._ask(prompt, prompt.spoken or prompt.question))
        return True

    def _promote(self, room: str | None) -> None:
        """The room's open prompt was settled: bring up the next one, if any."""
        nxt = self._room_head(room)
        if nxt is not None and not nxt.presented:
            self._spawn(self._present_next(nxt))

    async def _present_next(self, prompt: _Prompt) -> None:
        if prompt.id not in self._prompts or prompt.presented:
            return  # settled (or brought up) meanwhile
        if not await self._present(prompt):
            log.warning("housemate: no device online in %s for the next %s; it shows when one connects",
                        prompt.room, prompt.kind)

    async def retire_clarify_card(self, clarify_id: str, notice: str) -> None:
        prompt = self._prompts.get(clarify_id)
        if prompt is not None:
            self._retire_prompt(prompt, _plain(notice))

    def _prompt_msg(self, prompt: _Prompt) -> dict:
        msg = {"type": "prompt", "kind": prompt.kind, "id": prompt.id, "text": prompt.question,
               "choices": list(prompt.choices)}
        for name in ("title", "detail", "command"):  # extra fields for the satellite's card
            if getattr(prompt, name):
                msg[name] = getattr(prompt, name)
        return msg

    async def _show_prompt(self, prompt: _Prompt) -> bool:
        """Send the card to the allowed devices in its room; only they may tap it."""
        sent = 0
        for device_id in self._allowed_in(prompt.room):
            if await self._send(device_id, self._prompt_msg(prompt)):
                prompt.shown_to.add(device_id)
                sent += 1
        return sent > 0

    async def _ask(self, prompt: _Prompt, text: str) -> None:
        """Speak the question on the asking device; the answer window opens when it has played."""
        try:
            pcm = await self._speech_pcm(text)
        except Exception as exc:
            log.warning("housemate: could not speak the %s prompt: %s", prompt.kind, exc)
            pcm = b""
        if prompt.id not in self._prompts or prompt.device_id is None:
            return
        stream = await self._start_stream(prompt.device_id, PLAY_RATE, "ask", ref=prompt.id) if pcm else None
        if stream is None:
            if await self._send(prompt.device_id, {"type": "chime", "kind": "prompt"}):
                self._note_chime(prompt.device_id)
            if self.book.for_device(prompt.device_id, kinds=("approval", "clarify")) is not None:
                self._open_window(prompt.device_id, self.cfg.answer_window_s)
            return
        stream.playout.write(pcm)
        stream.playout.finish()

    def _arm_answer(self, prompt: _Prompt) -> None:
        """The asking device's next utterance answers this prompt (spoken, via the side chat)."""
        if prompt.device_id is None:
            return
        loop = self._loop

        def handle(transcript: str) -> str:  # runs in Hermes's agent thread
            if prompt.kind == "approval":
                if self._was_announced(transcript):
                    loop.call_soon_threadsafe(self._reprompt, prompt.id)
                    return "matches something we announced: not an answer"
                choice = parse_yes_no(transcript)
                if choice is None:
                    loop.call_soon_threadsafe(self._reprompt, prompt.id)
                    return f"not an approve/deny answer: {transcript[:60]!r}"
                try:
                    resolved = self._resolve_approval_card(prompt, choice)
                except Exception:
                    log.exception("housemate: resolving the approval failed")
                    resolved = 0
                loop.call_soon_threadsafe(self._approval_done, prompt.id, choice, resolved)
                return f"{choice} (resolved {resolved})"
            if prompt.skippable and is_skip(transcript):
                try:
                    ok = self._resolve_clarify_card(prompt, clarify_skip_value())
                except Exception:
                    log.exception("housemate: skipping the clarify prompt failed")
                    ok = False
                loop.call_soon_threadsafe(self._clarify_done, prompt.id, "Skipped.", ok)
                return "clarify skipped"
            if prompt.choices:
                pick = parse_choice(transcript, prompt.choices)
                if pick is None:
                    loop.call_soon_threadsafe(self._reprompt, prompt.id)
                    return f"no choice matched {transcript[:60]!r}"
            else:
                pick = transcript.strip()
                if not pick:
                    loop.call_soon_threadsafe(self._reprompt, prompt.id)
                    return "empty answer"
            try:
                ok = self._resolve_clarify_card(prompt, pick)
            except Exception:
                log.exception("housemate: resolving the clarify prompt failed")
                ok = False
            loop.call_soon_threadsafe(self._clarify_done, prompt.id, pick, ok)
            return f"clarify -> {pick!r} ({'resolved' if ok else 'already closed'})"

        ttl = APPROVAL_ANSWER_TTL_S if prompt.kind == "approval" else 3600.0
        self.book.drop(key=prompt.id)
        self.book.add(answers.PendingAnswer(
            token=answers.new_token(), kind=prompt.kind, device_id=prompt.device_id, room=prompt.room or "",
            expires_at=time.monotonic() + ttl, handle=handle, key=prompt.id))

    async def _send_answer(self, device_id: str, room: str, utt: Any, pending: answers.PendingAnswer) -> None:
        """Send the clip to the side chat; Hermes transcribes it and our pre_llm_call hook answers."""
        try:
            path = await self._cache_audio(pcm_to_wav(bytes(utt.pcm)), ".wav")
            source = self.build_source(chat_id=SIDE_CHAT, chat_name="Housemate answers", chat_type="dm",
                                       user_id=device_id, user_name=self._room_name(room))
            event = MessageEvent(
                text="",
                message_type=MessageType.VOICE,
                source=source,
                message_id=uuid.uuid4().hex,
                media_urls=[str(path)],
                media_types=["audio/wav"],
                channel_context=answers.make_marker(pending.token),
            )
            await self._wait_session_idle(event)
            await self.handle_message(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: sending a spoken answer failed")

    def _resolve_approval_card(self, prompt: _Prompt, choice: str) -> int:
        resolve = self._resolve_approval
        if resolve is None:
            from tools.approval import resolve_gateway_approval_card as resolve
        return int(resolve(prompt.session_key, "once" if choice == "approve" else "deny", prompt.request_id) or 0)

    def _resolve_clarify_card(self, prompt: _Prompt, response: str) -> bool:
        resolve = self._resolve_clarify
        if resolve is None:
            from tools.clarify_gateway import resolve_gateway_clarify as resolve
        return bool(resolve(prompt.id, response))

    def _apply_approval(self, prompt: _Prompt, choice: str, *, via: str) -> None:
        resolved = 0
        try:
            resolved = self._resolve_approval_card(prompt, choice)
        except Exception:
            log.exception("housemate: resolving the approval failed")
        self._approval_done(prompt.id, choice, resolved)

    def _apply_clarify(self, prompt: _Prompt, pick: str, *, via: str) -> None:
        ok = False
        try:
            ok = self._resolve_clarify_card(prompt, pick)
        except Exception:
            log.exception("housemate: resolving the clarify prompt failed")
        self._clarify_done(prompt.id, pick, ok)

    def _approval_done(self, card_id: str, choice: str, resolved: int) -> None:
        prompt = self._prompts.get(card_id)
        if prompt is None:
            return
        note = "Approved." if choice == "approve" else "Denied."
        if not resolved:
            note = "That approval had already closed."
        log.info("housemate: approval %s -> %s (resolved %s)", card_id, choice, resolved)
        self._retire_prompt(prompt, note)
        if prompt.device_id and resolved:
            self._set_state(prompt.device_id, "thinking")

    def _clarify_done(self, clarify_id: str, pick: str, ok: bool) -> None:
        prompt = self._prompts.get(clarify_id)
        if prompt is None:
            return
        self._retire_prompt(prompt, pick if ok else "That question had already closed.")
        if prompt.device_id and ok:
            self._set_state(prompt.device_id, "thinking")

    def _reprompt(self, card_id: str, *, overlapped: bool = False) -> None:
        """Ask again (at most ``MAX_REPROMPTS`` times, then a tap is the only answer). ``overlapped``:
        the answer clip held our own audio; ask on the screen and listen again once it is quiet."""
        prompt = self._prompts.get(card_id)
        if prompt is None or prompt.device_id is None:
            return
        prompt.tries += 1
        if prompt.tries > MAX_REPROMPTS:
            self.book.drop(key=prompt.id)
            self._spawn(self._caption_room(prompt.room, "status", "No answer understood; tap a button on the screen."))
            self._set_state(prompt.device_id, "idle")
            return
        self._arm_answer(prompt)
        if overlapped:
            say = "say approve, or deny" if prompt.kind == "approval" else "answer again"
            self._spawn(self._caption_room(prompt.room, "status",
                                           f"Sorry, I couldn't hear you over the speaker; {say}, or tap."))
            if prompt.device_id not in self._playing:
                self._open_window(prompt.device_id, self.cfg.answer_window_s)
            return
        if prompt.kind == "approval":
            again = "Sorry, I didn't catch that. Say approve, or deny."
        elif prompt.choices:
            again = "Sorry, which one? " + _read_choices(prompt.choices[:6]) + "."
        else:
            again = "Sorry, I didn't catch that. " + prompt.question
        self._spawn(self._ask(prompt, again))

    def _retire_prompt(self, prompt: _Prompt, note: str) -> None:
        self._prompts.pop(prompt.id, None)
        self.book.drop(key=prompt.id)
        if prompt.device_id:
            stream = self._streams.get(prompt.device_id)
            if stream is not None and stream.kind == "ask" and stream.ref == prompt.id:
                stream.playout.abort()
        if prompt.room:
            if prompt.presented:
                self._spawn(self._clear_prompt(prompt, note))
            self._promote(prompt.room)

    async def _clear_prompt(self, prompt: _Prompt, note: str) -> None:
        devices = {conn.device_id for conn in self.registry.devices_in(prompt.room)} | prompt.shown_to
        for device_id in sorted(devices):
            await self._send(device_id, {"type": "prompt.clear", "id": prompt.id})
        if note:
            await self._caption_room(prompt.room, "status", note)

    # ======================================================================== timers

    async def _on_timer_ring(self, timer: Any) -> None:
        room = timer.room
        for conn in self.registry.devices_in(room):
            self._arm_stop(room, conn.device_id)
            self._set_state(conn.device_id, "ringing")
        try:
            from .core.timers import ring_text

            await self._caption_room(room, "timer", ring_text(timer, getattr(self.timers, "tz", None)))
        except Exception:
            log.debug("housemate: no ring caption", exc_info=True)

    def _arm_stop(self, room: str, device_id: str) -> None:
        loop = self._loop

        def handle(transcript: str) -> str:
            if is_stop(transcript):
                loop.call_soon_threadsafe(self._stop_ring, room)
                return "stopped the ring"
            loop.call_soon_threadsafe(self._rearm_stop, room, device_id)
            return f"not a stop: {transcript[:60]!r}"

        self.book.drop(kind="stop", device_id=device_id)
        self.book.add(answers.PendingAnswer(
            token=answers.new_token(), kind="stop", device_id=device_id, room=room,
            expires_at=time.monotonic() + STOP_ANSWER_TTL_S, handle=handle, key=room))

    def _rearm_stop(self, room: str, device_id: str) -> None:
        if self.timers is not None and self.timers.is_ringing(room):
            self._arm_stop(room, device_id)
            self._set_state(device_id, "ringing")

    def _stop_ring(self, room: str) -> None:
        if self.timers is not None:
            self.timers.stop_ringing(room)
        self.book.drop(kind="stop", room=room)
        for conn in self.registry.devices_in(room):
            if conn.device_id not in self._playing:
                self._set_state(conn.device_id, "idle")

    async def _broadcast_timers(self, room: str, items: list[dict]) -> None:
        self._desk_poke()
        await self.announcer.push_timers(room, items)
        if self.timers is not None and not self.timers.is_ringing(room):
            self.book.drop(kind="stop", room=room)
            for conn in self.registry.devices_in(room):
                if self._state_of(conn.device_id) == "ringing":
                    self._set_state(conn.device_id, "idle")

    async def _push_timers(self, device_id: str) -> None:
        room = self._device_room(device_id)
        if room is None or self.timers is None:
            return
        items = [t.to_dict() for t in self.timers.list(room)]
        await self._send(device_id, {"type": "timers", "items": items})

    # ======================================================================== Desktop Rooms page

    def _desk_poke(self) -> None:
        """Something the Rooms page shows changed (any thread)."""
        feed = self._feed
        if feed is not None:
            feed.poke()

    async def _set_mode(self, mode: str) -> dict:
        """Switch the conversation mode now (from the next question on) and save it in config.yaml
        (``platforms.housemate.conversation``) so the next gateway start keeps it."""
        previous = self.router.mode
        self.router.mode = mode
        self.cfg = replace(self.cfg, conversation=mode)
        with contextlib.suppress(Exception):
            self.config.extra["conversation"] = mode
        error = await asyncio.to_thread(self._persist_mode, mode)
        self._desk_poke()
        log.info("housemate: conversation mode %s -> %s (%s)", previous, mode,
                 "saved" if error is None else f"not saved: {error}")
        result = {"ok": True, "mode": mode, "previous": previous, "applies": "next_question", "persisted": error is None}
        if error is not None:
            result["persist_error"] = error
        return result

    # ======================================================================== helpers

    def _spawn(self, coro: Any) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._bg.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("housemate: background task failed", exc_info=task.exception())

    async def _send(self, device_id: str, msg: dict) -> bool:
        try:
            return bool(await self.hub.send_control(device_id, msg))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: sending %s to %s failed", msg.get("type"), device_id)
            return False

    async def _send_state(self, device_id: str, state: str) -> None:
        self._states[device_id] = state
        await self._send(device_id, {"type": "state", "state": state})

    def _state_of(self, device_id: str) -> str:
        return self._states.get(device_id, "idle")

    def _set_state(self, device_id: str | None, state: str) -> None:
        """Tell a device what it is doing. Sends go out in call order; the newest request wins."""
        if not device_id or self.hub is None or self._states.get(device_id) == state:
            return
        self._states[device_id] = state
        self._spawn(self._send(device_id, {"type": "state", "state": state}))

    async def _caption_room(self, room: str | None, role: str, text: str) -> None:
        if not room or not text:
            return
        msg = {"type": "caption", "role": role, "text": text[:16_000]}
        self.desk.caption(room, role, msg["text"])
        online = [c.device_id for c in self.registry.devices_in(room)]
        delivered = set()
        for device_id in online:
            if await self._send(device_id, msg):
                delivered.add(device_id)
        # Keep it for the room's devices that are offline right now (phone locked, Wi-Fi blip).
        for device_id in self._room_devices(room):
            if device_id not in delivered:
                queue = self._captions.setdefault(device_id, deque(maxlen=CAPTION_KEEP))
                queue.append((time.monotonic(), role, msg["text"]))

    def _room_devices(self, room: str) -> list[str]:
        try:
            return [d.device_id for d in self.store.list_devices() if d.paired and d.room == room]
        except Exception:
            return []

    def _device_room(self, device_id: str | None) -> str | None:
        if not device_id:
            return None
        conn = self.registry.get(device_id) if self.registry is not None else None
        if conn is not None and conn.room:
            return conn.room
        try:
            device = self.store.get_device(device_id) if self.store is not None else None
        except Exception:
            device = None
        return device.room if device is not None and device.room else None

    def _device_name(self, device_id: str) -> str:
        conn = self.registry.get(device_id) if self.registry is not None else None
        return conn.name if conn is not None and conn.name else device_id

    def _room_name(self, room: str) -> str:
        try:
            found = self.store.get_room(room) if self.store is not None else None
        except Exception:
            found = None
        if found is not None:
            return found.name
        return room.replace("-", " ").title()

    def _earshot_group(self, room: str) -> str | None:
        try:
            found = self.store.get_room(room)
        except Exception:
            return None
        return found.earshot_group if found is not None else None

    def _chat_room(self, chat_id: str) -> str | None:
        """A chat id that names a room (per-room mode, ``housemate:<room>`` cron delivery)."""
        if chat_id in (HOUSEHOLD_CHAT, SIDE_CHAT):
            return None
        try:
            slug = slugify(chat_id)
        except ValueError:
            return None
        known = {s.room for s in self.registry.snapshot()}
        with contextlib.suppress(Exception):
            known.update(r.slug for r in self.store.list_rooms())
        return slug if slug in known else None

    def _delivery_rooms(self, chat_id: str) -> list[str]:
        room = self._chat_room(chat_id)
        if room:
            return [room]
        home = (os.environ.get("HOUSEMATE_HOME_CHANNEL") or "").strip()
        if home and self._chat_room(home):
            return [self._chat_room(home)]
        latest = self.router.latest_turn(chat_id)
        if latest is not None:
            return [latest.room]
        return sorted({s.room for s in self.registry.snapshot() if s.room})

    def _reply_device(self, chat_id: str, reply_to: str | None = None) -> str | None:
        """The device that asked, if online; else another online device in that room."""
        online = set(self.hub.online_devices()) if self.hub is not None else set()
        device_id = self.router.reply_device(chat_id, reply_to)
        if device_id in online:
            return device_id
        room = self.router.reply_room(chat_id, reply_to) or self._chat_room(chat_id)
        if room is None:
            return None
        candidates = sorted(c.device_id for c in self.registry.devices_in(room) if c.device_id in online)
        return candidates[0] if candidates else None

    def _mark(self, message_id: str | None, what: str) -> None:
        if message_id and message_id in self._timings and what not in self._timings[message_id]:
            self._timings[message_id][what] = time.time()

    def _log_timing(self, message_id: str) -> None:
        marks = self._timings.get(message_id)
        if not marks or "endpoint" not in marks:
            return
        llm = answers.last_llm_call_after(marks.get("dispatch", marks["endpoint"]))
        if llm is not None:
            marks.setdefault("llm_start", llm)
        base = marks["endpoint"]
        rel = {k: round(v - base, 3) for k, v in sorted(marks.items(), key=lambda kv: kv[1])}
        log.info("housemate timing %s %s", message_id, json.dumps(rel))
        room, kind = self._timing_meta.get(message_id, (None, None))
        self._turn_log.append({"message_id": message_id, "room": room, "kind": kind,
                               "endpoint_at": round(base, 3), "stages": rel})

    def recent_timings(self) -> list[dict]:
        """Stage times of the last finished turns, oldest first (``hermes housemate bench``)."""
        return list(self._turn_log)


def _hermes_approval_timeout() -> float:
    """Hermes's ``approvals.timeout`` (default 300 s), read through its public config loader."""
    try:
        from hermes_cli.config import load_config

        approvals = (load_config() or {}).get("approvals") or {}
        return max(0.0, float(approvals.get("timeout", 300)))
    except Exception:
        return 300.0


def _secret(name: str) -> str:
    """An env setting read through Hermes's profile-scoped reader (a raw ``os.environ`` read sees the
    default profile's value under ``gateway.multiplex_profiles``)."""
    try:
        from gateway.platforms._shared import get_scoped_secret

        return str(get_scoped_secret(name, "") or "").strip()
    except Exception:
        return ""


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _id_set(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split(",") if part.strip()}


def clarify_skip_value() -> str:
    """What a skipped clarify question resolves to: Hermes's own skip sentinel when available."""
    try:
        from tools.clarify_gateway import SKIPPED

        return SKIPPED
    except Exception:
        return "skip"  # what a user types to skip; the clarify tool reports it to the model


def _split_skip_hint(question: str) -> tuple[str, bool]:
    """Hermes appends a typed-reply hint (Reply "skip" to skip this question.) on its own last line
    for chat platforms. A speaker says it differently, so take it off the question."""
    lines = str(question or "").rstrip().splitlines()
    if len(lines) > 1 and "skip" in lines[-1].lower():
        return "\n".join(lines[:-1]).strip(), True
    return str(question or ""), False


def _read_choices(labels: list[str]) -> str:
    return "; ".join(f"{i}: {spoken_label(label)}" for i, label in enumerate(labels, start=1))


def _plain(text: str) -> str:
    """Card/caption text: strip the markdown and emoji decorations chat platforms use."""
    out = str(text or "").replace("```", " ").replace("**", "").replace("`", "")
    out = out.lstrip("⌛⚠️✅❌\U0001F6AB ")
    return " ".join(out.split())
