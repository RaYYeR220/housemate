"""The Desktop Rooms page's backend: RPC methods, control ops, the snapshot and the change feed."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from housemate import desktop_api, tools
from housemate.core.arbiter import WakeArbiter
from housemate.core.client import HousemateClient
from housemate.core.desk import CONVERSATION_MODES, DeskFeed, DeskLog, qr_svg
from housemate.core.hub import CLOSE_DENIED

from fake_satellite import FakeSatellite
from test_control import env  # noqa: F401  (the hub + control API fixture)

ROOT = Path(__file__).resolve().parents[1]


# --- RPC methods (fake client) ----------------------------------------------------------------


class FakeClient:
    def __init__(self, answer=None, boom=False):
        self.calls = []
        self.answer = answer if answer is not None else {"ok": True}
        self.boom = boom

    def call(self, op, args=None, **kw):
        self.calls.append((op, dict(args or {})))
        if self.boom:
            raise RuntimeError("socket gone")
        return self.answer


class RpcCtx:
    def __init__(self):
        self.methods = {}

    def register_rpc(self, name, handler):
        assert name not in self.methods
        self.methods[name] = handler


def test_register_desktop_registers_every_rooms_method():
    ctx = RpcCtx()
    names = desktop_api.register_desktop(ctx)
    assert names == list(desktop_api.RPC_METHODS)
    assert set(ctx.methods) == {
        "rooms.state", "rooms.announce", "rooms.timer_cancel", "rooms.pair_approve",
        "rooms.pair_deny", "rooms.assign", "rooms.mode",
    }


def test_register_desktop_is_a_no_op_without_plugin_rpc():
    assert desktop_api.register_desktop(SimpleNamespace()) == []


def test_rpc_handlers_forward_to_control_ops_with_only_their_params(monkeypatch):
    fake = FakeClient({"ok": True, "x": 1})
    monkeypatch.setattr(tools, "client", lambda: fake)
    ctx = RpcCtx()
    desktop_api.register_desktop(ctx)
    m = ctx.methods
    assert m["rooms.state"]({}, session_id=None) == {"ok": True, "x": 1}
    m["rooms.announce"]({"text": "Dinner", "rooms": ["kitchen"], "from_room": "office", "evil": 1}, session_id="s")
    m["rooms.timer_cancel"]({"timer_id": "t1", "room": None})
    m["rooms.pair_approve"]({"code": "123456", "room": "Living room"})
    m["rooms.pair_deny"]({"code": "654321"})
    m["rooms.assign"]({"device_id": "dev-1", "room": "office"})
    m["rooms.mode"]({"mode": "per_room"}, session_id=None, future_kwarg=True)
    assert fake.calls == [
        ("desk_state", {"qr": True}),
        ("announce", {"text": "Dinner", "rooms": ["kitchen"]}),
        ("timer_cancel", {"timer_id": "t1"}),
        ("pair_approve", {"code": "123456", "room": "Living room"}),
        ("pair_deny", {"code": "654321"}),
        ("assign", {"device_id": "dev-1", "room": "office"}),
        ("mode", {"mode": "per_room"}),
    ]


def test_rpc_handler_never_raises():
    assert desktop_api.call("rooms.state", None, client=FakeClient(boom=True))["ok"] is False
    assert desktop_api.call("rooms.nope", {}, client=FakeClient())["ok"] is False
    assert desktop_api.call("rooms.announce", "not a dict", client=FakeClient())["ok"] is True


def test_register_tools_registers_rpc_when_available():
    from test_tools import FakeCtx

    class FakeCtxWithRpc(FakeCtx):
        def __init__(self):
            super().__init__()
            self.rpc = {}

        def register_rpc(self, name, handler):
            self.rpc[name] = handler

    ctx = FakeCtxWithRpc()
    tools.register_tools(ctx)
    assert set(ctx.rpc) == set(desktop_api.RPC_METHODS)


def test_rpc_state_reaches_the_hub_when_it_is_down(tmp_path):
    result = desktop_api.call("rooms.state", {}, client=HousemateClient(tmp_path))
    assert result["ok"] is False and "hub not running" in result["error"]


# --- the snapshot, pairing, assignment, mode (real hub + control API) --------------------------


async def test_desk_state_lists_rooms_devices_timers_and_qr(env):  # noqa: F811
    env.services.desk = DeskLog()
    env.services.mode = lambda: "household"
    sat = await env.paired_sat()
    assert (await env.call("timer_set", {"seconds": 300, "label": "pasta", "room": "Kitchen"}))["ok"]
    env.services.desk.caption("kitchen", "user", "Set a pasta timer for five minutes.")
    env.services.desk.caption("kitchen", "assistant", "Your pasta timer is set.")
    state = await env.call("desk_state", {"qr": True})
    assert state["ok"] is True and state["mode"] == "household"
    kitchen = next(r for r in state["rooms"] if r["room"] == "kitchen")
    assert kitchen["online"] is True and kitchen["devices"] == ["dev-kitchen"]
    assert state["devices"] == [{"device_id": "dev-kitchen", "name": "Kitchen phone", "room": "kitchen",
                                 "online": True, "state": "idle"}]
    [timer] = state["timers"]
    assert timer["label"] == "pasta" and timer["room"] == "kitchen" and 290 < timer["due_at"] - state["now"] <= 300
    assert state["exchanges"]["kitchen"]["you"].startswith("Set a pasta")
    assert state["exchanges"]["kitchen"]["say"] == "Your pasta timer is set."
    assert state["hub"]["url"] == "https://192.168.1.20:8742" and state["hub"]["fingerprint"] == "AB:CD"
    assert state["qr"]["size"] > 20 and state["qr"]["path"].startswith("M")
    await sat.close()
    for _ in range(50):
        state = await env.call("desk_state")
        if not state["devices"][0]["online"]:
            break
        await asyncio.sleep(0.02)
    assert state["devices"][0]["state"] == "offline" and "qr" not in state


async def test_pending_pairing_listed_then_denied(env):  # noqa: F811
    sat = FakeSatellite(env.hub.ws_url, "dev-tablet", "Lenovo Tab")
    await sat.connect(send_hello=False)
    env.sats.append(sat)
    await sat.send_control({"type": "hello", "device_id": "dev-tablet", "name": "Lenovo Tab", "ua": "Chrome",
                            "room": "Living room"})
    code = (await sat.recv_control(type="pair.code"))["code"]
    [pending] = (await env.call("desk_state"))["pending"]
    assert pending["code"] == code and pending["name"] == "Lenovo Tab" and pending["wants"] == "Living room"
    assert 590 <= pending["expires_in_s"] <= 600
    assert (await env.call("pair_deny", {"code": code}))["ok"] is True
    err = await sat.recv_control(type="error")
    assert err["code"] == "pairing_denied"
    assert await sat.wait_closed() == CLOSE_DENIED
    assert (await env.call("desk_state"))["pending"] == []
    assert (await env.call("pair_deny", {"code": code}))["ok"] is False
    # a denied device that comes straight back gets no new code
    again = FakeSatellite(env.hub.ws_url, "dev-tablet", "Lenovo Tab")
    await again.connect()
    env.sats.append(again)
    assert (await again.recv_control(type="error"))["code"] == "pairing_denied"
    assert (await env.call("desk_state"))["pending"] == []


async def test_pair_approve_into_a_room_places_the_device(env):  # noqa: F811
    sat = FakeSatellite(env.hub.ws_url, "dev-tablet", "Lenovo Tab")
    await sat.connect()
    env.sats.append(sat)
    code = (await sat.recv_control(type="pair.code"))["code"]
    result = await env.call("pair_approve", {"code": code, "room": "Living room"})
    assert result == {"ok": True, "code": code, "room": "living-room"}
    paired = await sat.recv_control(type="paired")
    assert paired["room"] == "living-room" and paired["room_name"] == "Living room"
    assert env.services.registry.get("dev-tablet").room == "living-room"
    assert env.services.store.get_device("dev-tablet").room == "living-room"


async def test_assign_moves_an_online_device_and_tells_it(env):  # noqa: F811
    sat = await env.paired_sat(room="Kitchen")
    seen = []
    env.hub.on_control(lambda device_id, msg: seen.append((device_id, msg["type"], msg.get("room"))))
    result = await env.call("assign", {"device_id": "dev-kitchen", "room": "Office"})
    assert result == {"ok": True, "device_id": "dev-kitchen", "room": "office"}
    msg = await sat.recv_control(type="state")
    assert msg["room"] == "office" and msg["room_name"] == "Office"
    assert env.services.registry.get("dev-kitchen").room == "office"
    assert ("dev-kitchen", "room.set", "office") in seen
    bad = await env.call("assign", {"device_id": "dev-nobody", "room": "Office"})
    assert bad["ok"] is False and "dev-kitchen" in bad["error"]
    assert (await env.call("assign", {"device_id": "dev-kitchen", "room": "!!!"}))["ok"] is False


async def test_mode_op_switches_and_reports_persistence(env):  # noqa: F811
    current = {"mode": "household"}
    saved = []

    async def set_mode(mode):
        current["mode"] = mode
        saved.append(mode)
        return {"ok": True, "mode": mode, "applies": "next_question", "persisted": True}

    env.services.mode = lambda: current["mode"]
    env.services.set_mode = set_mode
    assert await env.call("mode") == {"ok": True, "mode": "household"}
    assert (await env.call("mode", {"mode": "per-room"}))["mode"] == "per_room"
    assert saved == ["per_room"]
    bad = await env.call("mode", {"mode": "kitchen"})
    assert bad["ok"] is False and "household" in bad["error"]
    assert CONVERSATION_MODES == ("household", "per_room")


# --- the adapter side: mode switch, captions, arbitration, feed ---------------------------------


def test_desk_log_keeps_the_last_exchange_per_room():
    t = [1000.0]
    log = DeskLog(clock=lambda: t[0])
    log.caption("kitchen", "user", "What's two plus two?")
    t[0] += 2
    log.caption("kitchen", "assistant", "Four.")
    log.caption("kitchen", "status", "Skipped.")  # housekeeping never replaces the exchange
    log.caption("office", "announcement", "Dinner is ready.")
    log.caption(None, "user", "nobody")
    ex = log.exchanges()
    assert ex["kitchen"] == {"you": "What's two plus two?", "say": "Four.", "from": None, "at": 1002.0}
    assert ex["office"]["from"] == "announcement" and ex["office"]["say"] == "Dinner is ready."


def test_arbiter_reports_each_round_to_observers():
    clock = [0.0]
    arb = WakeArbiter(window_ms=400, clock=lambda: clock[0])
    log = DeskLog(clock=lambda: 50.0)
    arb.on_decide(log.wake)
    arb.on_decide(lambda w: 1 / 0)  # a broken observer never breaks arbitration
    arb.offer("dev-a", "kitchen", "downstairs", 0.9, 0.2)
    arb.offer("dev-b", "office", "downstairs", 0.92, 0.6)
    clock[0] = 0.5
    assert arb.outcome("dev-b") == "won" and arb.outcome("dev-a") == "lost"
    [wake] = log.wakes()
    assert wake["winner"] == "dev-b" and wake["deferred"] == ["dev-a"] and wake["group"] == "downstairs"
    assert wake["at"] == 50.0 and wake["id"] == 1


async def test_feed_coalesces_pokes_from_any_thread():
    loop = asyncio.get_running_loop()
    sent = []
    built = []

    def build():
        built.append(1)
        return {"ok": True, "n": len(built)}

    feed = DeskFeed(loop, build, sent.append, delay_s=0.05)
    feed.poke()
    feed.poke()
    await asyncio.to_thread(feed.poke)
    await asyncio.sleep(0.15)
    assert sent == [{"rev": 1, "state": {"ok": True, "n": 1}}]
    feed.poke()
    await asyncio.sleep(0.1)
    assert [m["rev"] for m in sent] == [1, 2]
    feed.close()
    feed.poke()
    await asyncio.sleep(0.1)
    assert len(sent) == 2


async def test_feed_survives_a_failing_emit():
    loop = asyncio.get_running_loop()

    def emit(_payload):
        raise RuntimeError("no desktop here")

    feed = DeskFeed(loop, lambda: {"ok": True}, emit, delay_s=0.01)
    feed.poke()
    await asyncio.sleep(0.05)
    feed.poke()
    await asyncio.sleep(0.05)
    assert feed.rev == 2
    feed.close()


async def test_adapter_mode_switch_applies_now_and_persists(tmp_path):
    from adapter_kit import Kit

    adapter = Kit(tmp_path).wire().adapter
    saved = []
    adapter._persist_mode = lambda mode: saved.append(mode)
    result = await adapter._set_mode("per_room")
    assert result == {"ok": True, "mode": "per_room", "previous": "household", "applies": "next_question",
                      "persisted": True}
    assert adapter.router.mode == "per_room" and adapter.cfg.conversation == "per_room"
    assert adapter.router.chat_id_for("Kitchen") == "kitchen"
    assert saved == ["per_room"]
    adapter._persist_mode = lambda mode: "config.yaml is read-only"
    result = await adapter._set_mode("household")
    assert result["persisted"] is False and result["persist_error"] == "config.yaml is read-only"
    assert adapter.router.chat_id_for("Kitchen") == "household"


async def test_adapter_pushes_changes_to_desktop(tmp_path):
    from adapter_kit import Kit

    kit = Kit(tmp_path).wire()
    adapter = kit.adapter
    sent = []
    adapter._feed = DeskFeed(asyncio.get_running_loop(), lambda: {"n": len(sent)}, sent.append, delay_s=0.01)
    kit.add_device("dev-kitchen", "Kitchen", "Kitchen phone")  # registry change -> one push
    await asyncio.sleep(0.05)
    assert len(sent) == 1
    await adapter._caption_room("kitchen", "assistant", "Four.")
    await asyncio.sleep(0.05)
    assert len(sent) == 2 and adapter.desk.exchanges()["kitchen"]["say"] == "Four."
    adapter.desk.wake(SimpleNamespace(device_id="dev-kitchen", room="kitchen", group=None, score=0.9, losers=()))
    await asyncio.sleep(0.05)
    assert len(sent) == 3 and adapter.desk.wakes()[-1]["winner"] == "dev-kitchen"


def test_qr_svg_is_a_path_of_dark_runs():
    qr = qr_svg("https://192.168.1.20:8742")
    assert qr["text"] == "https://192.168.1.20:8742"
    assert 25 <= qr["size"] <= 45
    runs = qr["path"].split("z")
    assert len(runs) > 50 and all(r.startswith("M") for r in runs if r)
    assert qr_svg("https://192.168.1.20:8742") is qr  # cached


# --- the REST twin for Hermes 0.23.0 (dashboard/plugin_api.py) ------------------------------------


def _load_plugin_api():
    pytest.importorskip("fastapi")
    spec = importlib.util.spec_from_file_location("housemate_plugin_api_test", ROOT / "dashboard" / "plugin_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dashboard_manifest_points_at_the_api_and_hides_the_tab():
    manifest = json.loads((ROOT / "dashboard" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "housemate" and manifest["api"] == "plugin_api.py"
    assert manifest["tab"]["hidden"] is True
    assert (ROOT / "dashboard" / manifest["api"]).is_file()


def test_rest_routes_mirror_the_rpc_methods():
    api = _load_plugin_api()
    rpc_ops = {name.split(".", 1)[1]: spec for name, spec in desktop_api.RPC_METHODS.items()}
    assert set(api.ACTIONS) | {"state"} == set(rpc_ops)
    for action, spec in api.ACTIONS.items():
        assert spec == rpc_ops[action]


async def test_rest_routes_reach_the_hub_over_the_control_api(env):  # noqa: F811
    api = _load_plugin_api()
    api._data_dir = lambda: env.data_dir
    state = await api.state()
    assert state["ok"] is True and state["qr"]["path"]

    class Req:
        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

    result = await api.action("pair_deny", Req({"code": "123456", "junk": True}))
    assert result["ok"] is False and "123456" in result["error"]
    assert (await api.action("rm_rf", Req({})))["ok"] is False
    api._data_dir = lambda: env.data_dir / "missing"
    assert "hub not running" in (await api.state())["error"]
