"""Shared fakes for the adapter tests: a recording hub, listeners, timers, TTS and a wired adapter."""

from __future__ import annotations

import asyncio
import sqlite3
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import hermes_shim

hermes_shim.install()
if hermes_shim.HERMES_AVAILABLE:
    # Real Hermes: ``Platform("housemate")`` only resolves once the platform is registered.
    from gateway.platform_registry import PlatformEntry, platform_registry

    if not platform_registry.is_registered("housemate"):
        platform_registry.register(PlatformEntry(
            name="housemate", label="Housemate", adapter_factory=lambda cfg: None, check_fn=lambda: True,
            allowed_users_env="HOUSEMATE_ALLOWED_DEVICES", allow_all_env="HOUSEMATE_ALLOW_ALL_DEVICES",
        ))

from gateway.config import PlatformConfig  # noqa: E402  (after the shim)

from housemate.adapter import HousemateAdapter  # noqa: E402
from housemate.core.answers import AnswerBook  # noqa: E402
from housemate.core.protocol import encode  # noqa: E402
from housemate.core.rooms import DeviceConn, RoomRegistry  # noqa: E402
from housemate.core.store import HousemateStore  # noqa: E402

RATE = 24_000


class FakeHub:
    """Records control/PCM per device and validates every control message like the wire would."""

    def __init__(self, store: HousemateStore, registry: RoomRegistry) -> None:
        self.store = store
        self.rooms = registry
        self.online: set[str] = set()
        self.calls: list[tuple[str, str, object]] = []
        self.audio_cbs, self.control_cbs, self.disconnect_cbs = [], [], []
        self.url = "http://127.0.0.1:0"

    def on_audio(self, cb) -> None:
        self.audio_cbs.append(cb)

    def on_control(self, cb) -> None:
        self.control_cbs.append(cb)

    def on_disconnect(self, cb) -> None:
        self.disconnect_cbs.append(cb)

    def online_devices(self) -> list[str]:
        return sorted(self.online)

    async def send_control(self, device_id: str, msg: dict) -> bool:
        encode(msg)
        if device_id not in self.online:
            return False
        if msg["type"] == "state":
            self.rooms.set_device_state(device_id, msg["state"])
        self.calls.append((device_id, "control", dict(msg)))
        return True

    async def send_pcm(self, device_id: str, pcm: bytes) -> bool:
        if device_id not in self.online:
            return False
        self.calls.append((device_id, "pcm", bytes(pcm)))
        return True

    async def stop(self) -> None:
        return None

    # --- views for assertions
    def to(self, device_id: str) -> list[tuple[str, object]]:
        return [(kind, payload) for dev, kind, payload in self.calls if dev == device_id]

    def controls(self, device_id: str, mtype: str | None = None) -> list[dict]:
        return [p for k, p in self.to(device_id) if k == "control" and (mtype is None or p["type"] == mtype)]

    def pcm(self, device_id: str) -> bytes:
        return b"".join(p for k, p in self.to(device_id) if k == "pcm")


class FakeListener:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))

        return record

    def called(self, name: str) -> list[tuple]:
        return [c[1:] for c in self.calls if c[0] == name]


class FakeTimers:
    def __init__(self) -> None:
        self.ringing: set[str] = set()
        self.stopped: list[str] = []

    def is_ringing(self, room: str) -> bool:
        return room in self.ringing

    def stop_ringing(self, room: str) -> bool:
        self.stopped.append(room)
        was = room in self.ringing
        self.ringing.discard(room)
        return was

    def list(self, room=None):
        return []

    def close(self) -> None:
        return None

    async def run(self) -> None:
        await asyncio.Event().wait()


class FakeTTS:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.texts: list[str] = []

    async def __call__(self, text: str) -> list[Path]:
        self.texts.append(text)
        path = self.tmp / f"tts{len(self.texts)}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b"\x10\x00" * (RATE // 5))  # 0.2 s
        return [path]


def utterance(seconds: float = 1.0) -> SimpleNamespace:
    pcm = b"\x01\x00" * int(16_000 * seconds)
    now = time.time()
    return SimpleNamespace(pcm=pcm, started_at=now - seconds, ended_at=now, truncated=False)


class Kit:
    """A HousemateAdapter wired to fakes, with paired devices in rooms."""

    def __init__(self, tmp: Path, *, extra: dict | None = None, timers=None) -> None:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.store = HousemateStore(conn)
        self.registry = RoomRegistry()
        self.hub = FakeHub(self.store, self.registry)
        self.tts = FakeTTS(tmp)
        self.timers = timers or FakeTimers()
        self.adapter = HousemateAdapter(PlatformConfig(enabled=True, extra=dict(extra or {})))
        self.adapter.book = AnswerBook()
        self.adapter._tts_files = self.tts
        self.adapter._cache_audio = hermes_shim.cache_audio_from_bytes_async  # a temp dir, never a real home
        self.events: list = []
        self.resolved: list[tuple] = []
        self.clarified: list[tuple] = []

        async def handle_message(event) -> None:
            self.events.append(event)

        self.adapter.handle_message = handle_message
        self.adapter._resolve_approval = lambda sk, choice, rid: self.resolved.append((sk, choice, rid)) or 1
        self.adapter._resolve_clarify = lambda cid, response: self.clarified.append((cid, response)) or True
        self.listeners: dict[str, FakeListener] = {}

    def wire(self) -> "Kit":
        self.adapter._wire(self.store, self.registry, self.hub, timers=self.timers)
        return self

    def add_device(self, device_id: str, room: str, name: str | None = None) -> FakeListener:
        now = time.time()
        self.store.upsert_device(device_id, name or f"{room} phone")
        code = self.store.create_pairing_code(device_id, now)
        self.store.approve_pairing(code, now)
        slug = self.store.ensure_room(room).slug
        self.store.set_room(device_id, slug)
        self.connect(device_id, slug, name)
        listener = FakeListener()
        self.listeners[device_id] = listener
        self.adapter.listeners[device_id] = listener
        return listener

    def connect(self, device_id: str, slug: str, name: str | None = None) -> None:
        self.registry.attach(device_id, slug, DeviceConn(device_id=device_id, name=name or device_id))
        self.hub.online.add(device_id)

    def drop(self, device_id: str) -> None:
        self.hub.online.discard(device_id)
        self.registry.detach(device_id)
        self.adapter._on_disconnect(device_id)

    async def settle(self, rounds: int = 3) -> None:
        """Let background tasks (dispatch, playout, captions) run to completion."""
        for _ in range(rounds):
            pending = [t for t in self.adapter._bg if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=5)
            await asyncio.sleep(0)
