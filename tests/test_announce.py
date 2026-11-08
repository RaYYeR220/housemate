import importlib.util
import io
import json
import shutil
import sqlite3
import wave
from pathlib import Path

import pytest

from housemate.core.announce import AnnounceResult, Announcer, decode_to_pcm, tts_paths_from_result
from housemate.core.protocol import encode
from housemate.core.rooms import DeviceConn, RoomRegistry
from housemate.core.store import HousemateStore

ROOT = Path(__file__).resolve().parents[1]
PCM = b"\x01\x00\xff\x7f" * 50


class FakeHub:
    """Records what would go over the sockets; validates every control message."""

    def __init__(self, store: HousemateStore) -> None:
        self.store = store
        self.calls: list[tuple[str, str, object]] = []
        self.gone: set[str] = set()

    async def send_control(self, device_id: str, msg: dict) -> bool:
        encode(msg)
        if device_id in self.gone:
            return False
        self.calls.append((device_id, "control", msg))
        return True

    async def send_pcm(self, device_id: str, pcm: bytes) -> bool:
        if device_id in self.gone:
            return False
        self.calls.append((device_id, "pcm", bytes(pcm)))
        return True

    def to(self, device_id: str) -> list[tuple[str, object]]:
        return [(kind, payload) for dev, kind, payload in self.calls if dev == device_id]

    def devices(self) -> set[str]:
        return {dev for dev, _, _ in self.calls}


class FakeTTS:
    def __init__(self, tmp_path: Path, fail: bool = False) -> None:
        self.tmp_path = tmp_path
        self.texts: list[str] = []
        self.fail = fail

    async def __call__(self, text: str) -> Path:
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("provider down")
        path = self.tmp_path / f"tts{len(self.texts)}.mp3"
        path.write_bytes(b"not really mp3")
        return path


async def fake_decode(path, rate=24000):
    return PCM


@pytest.fixture
def store():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    s = HousemateStore(conn)
    for name in ("Kitchen", "Office", "Bedroom"):
        s.ensure_room(name)
    yield s
    conn.close()


@pytest.fixture
def registry():
    reg = RoomRegistry()
    for device_id, room in (("dev-k", "kitchen"), ("dev-o", "office"), ("dev-b", "bedroom")):
        reg.attach(device_id, room, DeviceConn(device_id=device_id, name=f"{room} device"))
    return reg


@pytest.fixture
def hub(store):
    return FakeHub(store)


@pytest.fixture
def tts(tmp_path):
    return FakeTTS(tmp_path)


@pytest.fixture
def announcer(hub, registry, tts):
    return Announcer(hub, registry, tts, decode=fake_decode)


def kinds(calls):
    return [payload["type"] if kind == "control" else "pcm" for kind, payload in calls]


# --- the plan's three ----------------------------------------------------------------


async def test_announce_all_rooms_except_origin_when_intercom(announcer, hub, tts):
    result = await announcer.announce("Dinner is ready", "all", from_room="Office")
    assert result.ok and result.error is None
    assert result.rooms == ["bedroom", "kitchen"]
    assert sorted(result.devices) == ["dev-b", "dev-k"]
    assert hub.devices() == {"dev-b", "dev-k"}
    assert hub.to("dev-o") == []
    assert tts.texts == ["Dinner is ready"]  # synthesised once, streamed to both


async def test_announce_unknown_room_returns_error_listing_rooms(announcer, hub, tts):
    result = await announcer.announce("hello", ["kitchen", "Garage"], from_room=None)
    assert result.ok is False
    assert "unknown room" in result.error and "Garage" in result.error
    assert result.rooms == ["bedroom", "kitchen", "office"]
    assert hub.calls == [] and tts.texts == []
    assert result.to_dict()["rooms"] == ["bedroom", "kitchen", "office"]


async def test_show_card_to_room(announcer, hub):
    assert await announcer.show("Kitchen", "Pasta", "1. Boil water\n2. Salt it") is True
    assert hub.devices() == {"dev-k"}
    [(kind, msg)] = hub.to("dev-k")
    assert kind == "control"
    assert msg["type"] == "card" and msg["title"] == "Pasta" and msg["body"].startswith("1. Boil")
    assert msg["id"]
    assert await announcer.show("garage", "x", "y") is False


# --- streaming details ----------------------------------------------------------------


async def test_announce_streams_chime_then_play_begin_pcm_end(announcer, hub):
    result = await announcer.announce("Package at the door", ["kitchen"], from_room="office")
    assert result.ok and result.rooms == ["kitchen"] and result.devices == ["dev-k"]
    calls = hub.to("dev-k")
    assert kinds(calls) == ["chime", "caption", "play.begin", "pcm", "play.end"]
    assert calls[0][1]["kind"] == "announce"
    assert calls[1][1]["text"] == "Package at the door"
    begin, end = calls[2][1], calls[4][1]
    assert begin["sample_rate"] == 24000
    assert begin["stream_id"] == end["stream_id"] and begin["stream_id"].startswith("ann-")
    assert calls[3][1] == PCM


async def test_announce_explicit_rooms_can_include_origin(announcer, hub):
    result = await announcer.announce("test", ["office", "kitchen", "office"], from_room="office")
    assert result.ok and result.rooms == ["office", "kitchen"]
    assert hub.devices() == {"dev-o", "dev-k"}


async def test_announce_room_without_online_device_fails_cleanly(hub, store, tts):
    store.ensure_room("Garage")
    reg = RoomRegistry()
    ann = Announcer(hub, reg, tts, decode=fake_decode)
    result = await ann.announce("anyone?", ["garage"], from_room=None)
    assert result.ok is False and "garage" in result.error
    assert result.offline == ["garage"]
    assert tts.texts == [] and hub.calls == []


async def test_announce_partial_delivery_reports_offline_rooms(hub, store, registry, tts):
    store.ensure_room("Garage")
    ann = Announcer(hub, registry, tts, decode=fake_decode)
    result = await ann.announce("hi", ["kitchen", "garage"], from_room=None)
    assert result.ok and result.devices == ["dev-k"] and result.offline == ["garage"]


async def test_announce_tts_failure_is_reported(hub, registry, tmp_path):
    ann = Announcer(hub, registry, FakeTTS(tmp_path, fail=True), decode=fake_decode)
    result = await ann.announce("hello", ["kitchen"], from_room=None)
    assert result.ok is False and "provider down" in result.error
    assert "play.begin" not in kinds(hub.to("dev-k"))


async def test_device_gone_mid_announce_does_not_raise(announcer, hub):
    hub.gone.add("dev-b")
    result = await announcer.announce("hi all", "all", from_room=None)
    assert result.ok and sorted(result.devices) == ["dev-k", "dev-o"]


async def test_say_reuses_tts_for_repeated_text(announcer, hub, tts):
    assert await announcer.say("kitchen", "Your tea timer is done.") == ["dev-k"]
    assert await announcer.say("kitchen", "Your tea timer is done.") == ["dev-k"]
    assert tts.texts == ["Your tea timer is done."]
    assert kinds(hub.to("dev-k")) == ["play.begin", "pcm", "play.end"] * 2
    begin = hub.to("dev-k")[0][1]
    assert begin["stream_id"].startswith("say-")


async def test_say_to_room_without_devices_skips_tts(announcer, tts):
    assert await announcer.say("garage", "nobody home") == []
    assert tts.texts == []


async def test_chime_and_push_timers_reach_room_devices(announcer, hub):
    assert await announcer.chime("kitchen", "timer") == 1
    items = [{"id": "t1", "state": "ringing"}]
    assert await announcer.push_timers("kitchen", items) == 1
    assert hub.to("dev-k") == [
        ("control", {"type": "chime", "kind": "timer"}),
        ("control", {"type": "timers", "items": items}),
    ]
    assert await announcer.chime("garage") == 0


def test_announce_result_to_dict():
    r = AnnounceResult(ok=False, rooms=["kitchen"], error="unknown room 'x'")
    assert r.to_dict() == {"ok": False, "rooms": ["kitchen"], "devices": [], "offline": [], "error": "unknown room 'x'"}


# --- TTS result parsing and decoding ------------------------------------------------------


def test_tts_paths_from_result(tmp_path):
    one = json.dumps({"success": True, "file_path": str(tmp_path / "a.mp3")})
    assert tts_paths_from_result(one) == [tmp_path / "a.mp3"]
    many = {"success": True, "file_path": "a.ogg", "file_paths": ["a.ogg", "b.ogg"]}
    assert tts_paths_from_result(many) == [Path("a.ogg"), Path("b.ogg")]
    with pytest.raises(RuntimeError, match="no key"):
        tts_paths_from_result(json.dumps({"success": False, "error": "no key"}))
    with pytest.raises(RuntimeError):
        tts_paths_from_result("not json")
    with pytest.raises(RuntimeError):
        tts_paths_from_result({"success": True})


def _wav(path: Path, rate: int, frames: bytes) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return path


async def test_decode_to_pcm_reads_matching_wav_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)  # proves ffmpeg is not needed
    frames = b"\x10\x00\x20\x00" * 1200
    path = _wav(tmp_path / "speech.wav", 24000, frames)
    assert await decode_to_pcm(path, rate=24000) == frames


async def test_decode_to_pcm_resamples_a_wav_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)  # a local voice's WAV never needs ffmpeg
    import numpy as np

    t = np.arange(22050) / 22050
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype("<i2").tobytes()  # 1 s of 440 Hz at 22.05 kHz
    pcm = await decode_to_pcm(_wav(tmp_path / "piper.wav", 22050, tone), rate=24000)
    out = np.frombuffer(pcm, "<i2").astype(np.float64)
    assert len(out) == 24000
    peak = np.argmax(np.abs(np.fft.rfft(out)))  # still a 440 Hz tone (1 Hz bins over 1 s)
    assert abs(peak - 440) <= 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_decode_to_pcm_uses_ffmpeg_for_other_formats(tmp_path):
    import subprocess

    wav = _wav(tmp_path / "speech16k.wav", 16000, b"\x00\x10" * 16000)  # 1 s
    mp3 = tmp_path / "speech.mp3"
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-i", str(wav), str(mp3)], check=True)
    pcm = await decode_to_pcm(mp3, rate=24000)
    assert abs(len(pcm) - 48000) <= 4000


async def test_decode_to_pcm_reports_bad_input(tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"definitely not audio")
    with pytest.raises(RuntimeError):
        await decode_to_pcm(bad)


# --- the chime asset -------------------------------------------------------------------------


def _load_make_chime():
    spec = importlib.util.spec_from_file_location("make_chime", ROOT / "scripts" / "make_chime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(data: bytes) -> tuple:
    with wave.open(io.BytesIO(data), "rb") as w:
        params = w.getparams()
        raw = w.readframes(w.getnframes())
    return params, [int.from_bytes(raw[i : i + 2], "little", signed=True) for i in range(0, len(raw), 2)]


def test_chime_wav_is_small_mono_pcm16():
    data = (ROOT / "web" / "chime.wav").read_bytes()
    assert len(data) <= 60 * 1024
    params, samples = _samples(data)
    assert params.nchannels == 1 and params.sampwidth == 2
    assert 0.5 <= params.nframes / params.framerate <= 2.0
    peak = max(abs(s) for s in samples)
    assert 8000 < peak < 32767  # audible, not clipped
    assert max(abs(s) for s in samples[-50:]) < 200  # fades out, no click at the end


def test_make_chime_reproduces_committed_wav():
    rendered = _load_make_chime().render()
    committed = (ROOT / "web" / "chime.wav").read_bytes()
    p1, a = _samples(rendered)
    p2, b = _samples(committed)
    assert p1 == p2
    assert max(abs(x - y) for x, y in zip(a, b)) <= 1
