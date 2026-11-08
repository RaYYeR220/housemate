import asyncio
import json
import os
import socket
import sqlite3
import time
from pathlib import Path

import aiohttp
import pytest

from housemate.core.announce import Announcer
from housemate.core.client import HUB_DOWN, HousemateClient
from housemate.core.control import (
    DEFAULT_PORT,
    ControlPortInUse,
    ControlServer,
    HubServices,
    bind_loopback,
    local_services,
    register_local_services,
    run_op,
    unregister_local_services,
)
from housemate.core.hub import Hub
from housemate.core.rooms import RoomRegistry
from housemate.core.store import HousemateStore
from housemate.core.timers import TimerService
from housemate.core.tls import lan_ipv4s

from fake_satellite import FakeSatellite

ROOT = Path(__file__).resolve().parents[1]
PCM = b"\x01\x00\xff\x7f" * 50


class FakeTTS:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.texts: list[str] = []

    async def __call__(self, text: str) -> Path:
        self.texts.append(text)
        path = self.tmp_path / f"tts{len(self.texts)}.mp3"
        path.write_bytes(b"fake")
        return path


async def fake_decode(path, rate=24000):
    return PCM


class Env:
    def __init__(self, services: HubServices, server: ControlServer, data_dir: Path, tts: FakeTTS):
        self.services, self.server, self.data_dir, self.tts = services, server, data_dir, tts
        self.sats: list[FakeSatellite] = []

    @property
    def hub(self) -> Hub:
        return self.services.hub

    def client(self) -> HousemateClient:
        return HousemateClient(self.data_dir)

    async def call(self, op: str, args: dict | None = None) -> dict:
        """HTTP loopback from another thread, the way the CLI and other processes reach the hub."""
        return await asyncio.to_thread(self.client().call, op, args or {})

    async def paired_sat(self, device_id="dev-kitchen", name="Kitchen phone", room="Kitchen"):
        first = FakeSatellite(self.hub.ws_url, device_id, name)
        await first.connect()
        self.sats.append(first)
        code = (await first.recv_control(type="pair.code"))["code"]
        assert (await self.call("pair_approve", {"code": code}))["ok"] is True
        await first.recv_control(type="paired")
        await first.close()
        sat = FakeSatellite(self.hub.ws_url, device_id, name, first.device_key)
        await sat.connect()
        self.sats.append(sat)
        await sat.recv_control(type="paired")
        if room:
            await sat.send_control({"type": "room.set", "room": room})
            await sat.recv_control(type="state")
        return sat


@pytest.fixture
async def env(tmp_path):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    store, registry = HousemateStore(conn), RoomRegistry()
    hub = Hub(store, registry, host="127.0.0.1", port=0, cert=None, key=None, web_root=ROOT / "web", tls=False)
    await hub.start()
    tts = FakeTTS(tmp_path)
    announcer = Announcer(hub, registry, tts, decode=fake_decode)
    timers = TimerService(store, time.time, None, play_chime=lambda r: announcer.chime(r, "timer"),
                          speak=announcer.say, broadcast_timers=announcer.push_timers)
    services = HubServices(store=store, registry=registry, hub=hub, timers=timers, announcer=announcer,
                           urls=["https://192.168.1.20:8742"], fingerprint="AB:CD")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    server = ControlServer(services, data_dir / "control.token", port=0)
    await server.start()
    e = Env(services, server, data_dir, tts)
    yield e
    unregister_local_services()
    for sat in e.sats:
        await sat.close()
    timers.close()
    await server.stop()
    await hub.stop()
    conn.close()


async def _post(env, op, body=None, token=None, headers=None):
    hdrs = dict(headers or {})
    if token is not None:
        hdrs["Authorization"] = f"Bearer {token}"
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{env.server.url}/v1/{op}", json=body or {}, headers=hdrs) as resp:
            return resp.status, await resp.json()


# --- the server itself ----------------------------------------------------------------------


async def test_control_rejects_missing_token(env):
    status, body = await _post(env, "rooms")
    assert status == 401 and body["ok"] is False
    status, _ = await _post(env, "rooms", token="not-the-token")
    assert status == 401
    status, body = await _post(env, "rooms", token=env.server.token)
    assert status == 200 and body["ok"] is True


async def test_control_rejects_browser_origin(env):
    status, body = await _post(env, "rooms", token=env.server.token, headers={"Origin": "http://evil.example"})
    assert status == 403 and body["ok"] is False


async def test_control_binds_loopback_only(env):
    assert env.server.host == "127.0.0.1"
    addrs = env.server.bound_addresses()
    assert addrs and all(host == "127.0.0.1" for host, _ in addrs)
    for ip in lan_ipv4s()[:1]:
        with pytest.raises(OSError):
            socket.create_connection((ip, env.server.port), timeout=1.0).close()


def test_default_control_port():
    assert DEFAULT_PORT == 8743
    assert ControlServer(None, Path("control.token")).port == 8743


async def test_second_server_on_same_port_fails_loudly(env, tmp_path):
    other_dir = tmp_path / "other"
    other = ControlServer(env.services, other_dir / "control.token", port=env.server.port)
    with pytest.raises(ControlPortInUse) as excinfo:
        await other.start()
    assert str(env.server.port) in str(excinfo.value) and "in use" in str(excinfo.value)
    assert not (other_dir / "control.token").exists()  # a live hub's token is never rotated away
    status, _ = await _post(env, "rooms", token=env.server.token)
    assert status == 200


def test_port_cannot_be_shared_even_with_reuseaddr():
    sock = bind_loopback(0)
    port = sock.getsockname()[1]
    thief = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        thief.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            thief.bind(("127.0.0.1", port))
    finally:
        thief.close()
        sock.close()


async def test_client_finds_non_default_port(env):
    assert env.server.port != DEFAULT_PORT
    assert (await env.call("info"))["ok"] is True


async def test_token_file_is_fresh_and_private(env):
    path = env.data_dir / "control.token"
    token = path.read_text(encoding="utf-8").strip()
    assert token == env.server.token and len(token) >= 32
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600
    info = json.loads((env.data_dir / "control.json").read_text(encoding="utf-8"))
    assert info["port"] == env.server.port


async def test_unknown_op_is_404(env):
    status, body = await _post(env, "self_destruct", token=env.server.token)
    assert status == 404 and "unknown" in body["error"]


async def test_stop_removes_port_file(tmp_path):
    services = HubServices(store=None, registry=RoomRegistry(), hub=None, timers=None, announcer=None)
    server = ControlServer(services, tmp_path / "control.token", port=0)
    await server.start()
    assert (tmp_path / "control.json").exists()
    await server.stop()
    assert not (tmp_path / "control.json").exists()


# --- client: loopback, in-process, hub down ---------------------------------------------------------


async def test_timer_set_via_client_reaches_service(env):
    env.services.store.ensure_room("Kitchen")
    result = await env.call("timer_set", {"seconds": 600, "label": "pasta", "room": "Kitchen"})
    assert result["ok"] is True
    assert result["timer"]["room"] == "kitchen" and result["timer"]["label"] == "pasta"
    [timer] = env.services.timers.list("kitchen")
    assert timer.label == "pasta" and 590 <= timer.remaining_s <= 600


async def test_client_uses_in_process_hub_from_another_thread(env):
    env.services.store.ensure_room("Office")
    register_local_services(env.services, asyncio.get_running_loop())
    assert local_services()[0] is env.services
    client = HousemateClient(None)  # no data dir: only the in-process path can work
    result = await asyncio.to_thread(client.call, "timer_set", {"seconds": 30, "room": "office"})
    assert result["ok"] is True
    assert [t.room for t in env.services.timers.list()] == ["office"]


async def test_client_acall_on_hub_loop_runs_directly(env):
    env.services.store.ensure_room("Office")
    register_local_services(env.services, asyncio.get_running_loop())
    result = await HousemateClient(None).acall("rooms")
    assert result["ok"] is True and [r["room"] for r in result["rooms"]] == ["office"]


async def test_sync_call_on_hub_loop_never_deadlocks(env):
    register_local_services(env.services, asyncio.get_running_loop())
    result = HousemateClient(None).call("rooms")
    assert result["ok"] is False and "acall" in result["error"]


def test_client_reports_hub_down(tmp_path):
    unregister_local_services()
    result = HousemateClient(tmp_path).call("rooms")
    assert result == {"ok": False, "error": HUB_DOWN}
    (tmp_path / "control.token").write_text("stale", encoding="utf-8")
    (tmp_path / "control.json").write_text(json.dumps({"port": _free_port()}), encoding="utf-8")
    assert HousemateClient(tmp_path).call("rooms") == {"ok": False, "error": HUB_DOWN}


def test_client_with_stopped_loop_falls_back_to_hub_down(tmp_path):
    loop = asyncio.new_event_loop()
    services = HubServices(store=None, registry=RoomRegistry(), hub=None, timers=None, announcer=None)
    register_local_services(services, loop)
    try:
        assert HousemateClient(tmp_path).call("rooms") == {"ok": False, "error": HUB_DOWN}
    finally:
        unregister_local_services()
        loop.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_client_rejected_token_is_reported(env):
    (env.data_dir / "control.token").write_text("wrong-token", encoding="utf-8")
    result = await env.call("rooms")
    assert result["ok"] is False and "token" in result["error"]


# --- ops ---------------------------------------------------------------------------------------------


async def test_pairing_devices_show_announce_revoke(env):
    kitchen = await env.paired_sat()
    office = await env.paired_sat("dev-office", "Office laptop", "Office")

    devices = (await env.call("devices"))["devices"]
    assert {d["device_id"]: (d["room"], d["online"], d["paired"]) for d in devices} == {
        "dev-kitchen": ("kitchen", True, True),
        "dev-office": ("office", True, True),
    }

    shown = await env.call("show", {"title": "Pasta", "body": "1. Boil water", "room": "kitchen"})
    assert shown["ok"] is True
    card = await kitchen.recv_control(type="card")
    assert card["title"] == "Pasta"

    result = await env.call("announce", {"text": "Dinner is ready", "rooms": "all", "from_room": "kitchen"})
    assert result["ok"] is True and result["rooms"] == ["office"]
    assert (await office.recv_control(type="caption"))["text"] == "Dinner is ready"
    assert env.tts.texts == ["Dinner is ready"]

    revoked = await env.call("revoke", {"device_id": "dev-office"})
    assert revoked["ok"] is True
    assert await office.wait_closed() == 4401


async def test_pair_with_unknown_code_explains(env):
    result = await env.call("pair_approve", {"code": "123456"})
    assert result["ok"] is False and "123456" in result["error"]


async def test_rooms_op_reports_rooms_devices_timers(env):
    await env.paired_sat()
    env.services.store.ensure_room("Office")
    env.services.timers.add("kitchen", seconds=90, label="tea")
    result = await env.call("rooms")
    by_room = {r["room"]: r for r in result["rooms"]}
    assert set(by_room) == {"kitchen", "office"}
    assert by_room["kitchen"]["online"] is True and by_room["kitchen"]["devices"][0]["name"] == "Kitchen phone"
    assert by_room["kitchen"]["timers"][0]["label"] == "tea"
    assert by_room["office"]["online"] is False and by_room["office"]["state"] == "offline"
    assert result["hub_url"] == "https://192.168.1.20:8742"


async def test_timer_needs_room_when_several_and_none_known_from_session(env):
    for name in ("Kitchen", "Office"):
        env.services.store.ensure_room(name)
    result = await env.call("timer_set", {"seconds": 60})
    assert result["ok"] is False
    assert "which room" in result["error"] and "kitchen" in result["error"] and "office" in result["error"]
    hinted = await env.call("timer_set", {"seconds": 60, "session_room": "Office"})
    assert hinted["ok"] is True and hinted["timer"]["room"] == "office"


async def test_timer_defaults_to_the_only_room(env):
    env.services.store.ensure_room("Kitchen")
    result = await env.call("timer_set", {"seconds": 60})
    assert result["ok"] is True and result["timer"]["room"] == "kitchen"


async def test_unknown_room_lists_known_rooms(env):
    env.services.store.ensure_room("Kitchen")
    result = await env.call("timer_set", {"seconds": 60, "room": "Garage"})
    assert result["ok"] is False and "garage" in result["error"].lower() and "kitchen" in result["error"]


async def test_bad_timer_values_are_errors_not_crashes(env):
    env.services.store.ensure_room("Kitchen")
    for seconds in ("soon", -5, 10**9, None, True):
        result = await env.call("timer_set", {"seconds": seconds, "room": "kitchen"})
        assert result["ok"] is False and result["error"], seconds
    alarm = await env.call("alarm_set", {"time": "25:99", "room": "kitchen"})
    assert alarm["ok"] is False and "HH:MM" in alarm["error"]


async def test_alarm_list_cancel(env):
    env.services.store.ensure_room("Kitchen")
    alarm = await env.call("alarm_set", {"time": "07:30", "label": "wake up", "room": "kitchen"})
    assert alarm["ok"] is True and alarm["timer"]["kind"] == "alarm"
    await env.call("timer_set", {"seconds": 300, "label": "eggs", "room": "kitchen"})
    listed = await env.call("timer_list")
    assert sorted(t["label"] for t in listed["timers"]) == ["eggs", "wake up"]
    cancelled = await env.call("timer_cancel", {"label": "Eggs timer"})
    assert cancelled["ok"] is True and [t["label"] for t in cancelled["cancelled"]] == ["eggs"]
    missing = await env.call("timer_cancel", {"label": "pizza"})
    assert missing["ok"] is False and [t["label"] for t in missing["timers"]] == ["wake up"]
    nothing = await env.call("timer_cancel", {})
    assert nothing["ok"] is False and "which timer" in nothing["error"]
    whole_room = await env.call("timer_cancel", {"session_room": "kitchen"})
    assert whole_room["ok"] is True and env.services.timers.list() == []


async def test_show_to_room_without_device(env):
    env.services.store.ensure_room("Kitchen")
    result = await env.call("show", {"title": "x", "body": "y", "room": "kitchen"})
    assert result["ok"] is False and "online" in result["error"]


async def test_room_admin_ops(env):
    added = await env.call("room_add", {"name": "Living Room"})
    assert added["ok"] is True and added["room"]["slug"] == "living-room"
    grouped = await env.call("room_group", {"room": "living-room", "group": "downstairs"})
    assert grouped["ok"] is True
    assert env.services.store.list_rooms()[0].earshot_group == "downstairs"
    removed = await env.call("room_remove", {"name": "living room"})
    assert removed["ok"] is True and env.services.store.list_rooms() == []
    again = await env.call("room_remove", {"name": "living room"})
    assert again["ok"] is False


async def test_room_with_online_device_is_not_removed(env):
    await env.paired_sat()
    result = await env.call("room_remove", {"name": "Kitchen"})
    assert result["ok"] is False and "Kitchen phone" in result["error"]


async def test_info_op(env):
    result = await env.call("info")
    assert result["ok"] is True and result["fingerprint"] == "AB:CD"
    assert result["urls"] == ["https://192.168.1.20:8742"]


async def test_run_op_never_raises(env):
    class Boom:
        def list(self, room=None):
            raise RuntimeError("disk on fire")

    env.services.timers = Boom()
    result = await run_op(env.services, "timer_list", {})
    assert result["ok"] is False and "disk on fire" in result["error"]
    assert (await run_op(env.services, "timer_set", ["not", "a", "dict"]))["ok"] is False


async def test_timings_op_returns_recent_turns(env):
    assert (await env.call("timings")) == {"ok": True, "turns": []}  # a hub without timing records
    turns = [{"message_id": "m1", "room": "kitchen", "endpoint_at": 1000.0, "stages": {"endpoint": 0.0, "dispatch": 0.2}}]
    env.services.timings = lambda: turns
    result = await asyncio.to_thread(env.client().timings)
    assert result == {"ok": True, "turns": turns}


def test_local_services_are_kept_per_profile(tmp_path):
    """Review L2: under gateway.multiplex_profiles each profile's hub registers in the same process;
    a tool finds the hub of its own profile (by its plugin data dir), never the last one registered."""
    loop = asyncio.new_event_loop()
    a = HubServices(store=None, registry=RoomRegistry(), hub=None, timers=None, announcer=None)
    b = HubServices(store=None, registry=RoomRegistry(), hub=None, timers=None, announcer=None)
    try:
        register_local_services(a, loop, key=tmp_path / "a")
        register_local_services(b, loop, key=tmp_path / "b")
        assert local_services(tmp_path / "a")[0] is a
        assert local_services(tmp_path / "b")[0] is b
        assert local_services(tmp_path / "c") is None
        assert local_services() is None  # two hubs: ambiguous, the control API decides
        unregister_local_services(a)
        assert local_services()[0] is b
        assert local_services(tmp_path / "a") is None
    finally:
        unregister_local_services()
        loop.close()
