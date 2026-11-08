import asyncio
import io
import json
import sqlite3
import string
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import housemate
from housemate import cli, commands, tools
from housemate.core.client import HUB_DOWN
from housemate.core.control import HubServices, register_local_services, unregister_local_services
from housemate.core.rooms import DeviceConn, RoomRegistry
from housemate.core.store import HousemateStore
from housemate.core.timers import TimerService

ROOT = Path(__file__).resolve().parents[1]
TOOL_NAMES = [
    "housemate_rooms",
    "housemate_announce",
    "housemate_timer_set",
    "housemate_alarm_set",
    "housemate_timer_list",
    "housemate_timer_cancel",
    "housemate_show",
]
MINIMAL_ARGS = {
    "housemate_rooms": {},
    "housemate_announce": {"text": "Dinner is ready", "rooms": ["all"]},
    "housemate_timer_set": {"seconds": 600, "label": "pasta"},
    "housemate_alarm_set": {"time": "07:30"},
    "housemate_timer_list": {},
    "housemate_timer_cancel": {"label": "pasta"},
    "housemate_show": {"title": "Pasta", "body": "Boil water"},
}


class FakeCtx:
    """Records every registration the plugin makes, like Hermes's PluginContext would."""

    def __init__(self, manifest=None):
        self.manifest = manifest if manifest is not None else SimpleNamespace(name="housemate")
        self.tools: dict[str, dict] = {}
        self.commands: dict[str, dict] = {}
        self.cli: dict[str, dict] = {}
        self.blueprints: dict[str, dict] = {}
        self.skills: dict[str, dict] = {}
        self.platforms: list[dict] = []

    def register_tool(self, name, toolset, schema, handler, check_fn=None, requires_env=None,
                      is_async=False, description="", emoji="", override=False):
        self.tools[name] = dict(toolset=toolset, schema=schema, handler=handler, check_fn=check_fn,
                                is_async=is_async, description=description)

    def register_command(self, name, handler, description="", args_hint="", argument_mode=None):
        self.commands[name] = dict(handler=handler, description=description, args_hint=args_hint)

    def register_cli_command(self, name, help, setup_fn, handler_fn=None, description=""):
        self.cli[name] = dict(help=help, setup_fn=setup_fn, handler_fn=handler_fn)

    def register_automation_blueprint(self, key, **fields):
        assert key not in self.blueprints, f"blueprint {key} registered twice"
        self.blueprints[key] = fields

    def register_skill(self, name, path, description="", frontmatter=None):
        self.skills[name] = dict(path=Path(path), description=description)

    def register_platform(self, **kwargs):
        self.platforms.append(kwargs)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    unregister_local_services()
    monkeypatch.setattr(tools, "data_dir", lambda: tmp_path)
    monkeypatch.setattr(tools, "session_room", lambda: None)
    yield
    unregister_local_services()


def manifest() -> dict:
    return yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))


# --- registration -------------------------------------------------------------------------------


def test_declared_tools_match_registered():
    ctx = FakeCtx()
    housemate.register(ctx)
    assert sorted(ctx.tools) == sorted(manifest()["provides_tools"]) == sorted(TOOL_NAMES)
    for name, entry in ctx.tools.items():
        assert entry["toolset"] == "housemate"
        assert entry["schema"]["name"] == name
        assert len(entry["schema"]["description"]) > 80
        params = entry["schema"]["parameters"]
        assert params["type"] == "object"
        assert set(params.get("required", [])) <= set(params["properties"])
        assert entry["check_fn"]() is True
    assert [p["name"] for p in ctx.platforms] == ["housemate"]


def test_register_registers_commands_cli_blueprints_and_skill():
    ctx = FakeCtx()
    housemate.register(ctx)
    assert set(ctx.commands) == {"housemate", "announce"}
    assert set(ctx.cli) == {"housemate"}
    assert set(ctx.blueprints) == {"morning", "bedtime", "remind"}
    assert set(ctx.skills) == {"housemate"}
    skill = ctx.skills["housemate"]["path"]
    assert skill.is_file() and skill.name == "SKILL.md"
    assert skill.read_text(encoding="utf-8").startswith("---\nname: housemate\n")


def test_register_tools_alone_is_enough_for_deferred_platforms():
    """Hermes may import only tools.py (provides_tools) and never call register()."""
    ctx = FakeCtx()
    tools.register_tools(ctx)
    assert sorted(ctx.tools) == sorted(TOOL_NAMES)
    assert set(ctx.commands) == {"housemate", "announce"} and set(ctx.cli) == {"housemate"}
    assert ctx.platforms == []


def test_deferred_then_materialized_registers_once():
    shared = SimpleNamespace(name="housemate")
    early = FakeCtx(shared)
    tools.register_tools(early)
    late = FakeCtx(shared)
    housemate.register(late)  # same manifest: the adapter materialised later
    assert late.blueprints == {} and late.tools == {}
    assert [p["name"] for p in late.platforms] == ["housemate"]
    fresh = FakeCtx(SimpleNamespace(name="housemate"))  # a forced re-discovery parses a new manifest
    housemate.register(fresh)
    assert sorted(fresh.tools) == sorted(TOOL_NAMES)


def test_blueprints_follow_hermes_slot_rules():
    ctx = FakeCtx()
    housemate.register(ctx)
    derived = {"minute": "time", "hour": "time", "dow": None}
    for key, bp in ctx.blueprints.items():
        slots = {s["name"]: s for s in bp["slots"]}
        assert len(slots) == len(bp["slots"])
        for s in bp["slots"]:
            assert s["type"] in {"time", "enum", "text", "weekdays"}
            assert s.get("default") not in (None, ""), f"{key}.{s['name']} needs a default"
        for _, name, _, _ in string.Formatter().parse(bp["prompt_template"]):
            if name is not None:
                assert name in slots, f"{key}: prompt uses {{{name}}}"
        for _, name, _, _ in string.Formatter().parse(bp["schedule_template"]):
            if name is not None:
                assert name in slots or (name in derived and (derived[name] is None or derived[name] in slots))
        assert slots["deliver"]["default"] == "housemate" and slots["deliver"]["strict"] is False
        assert bp["deliver_default"] == "housemate"
    assert ctx.blueprints["morning"]["prompt_template"].startswith(
        "Give me a 30-second spoken morning briefing: weather for my location, my first calendar item "
        "if a calendar tool is available, and today's running timers or alarms."
    )


def test_blueprints_pass_hermes_validation_when_available():
    plugins = pytest.importorskip("cron.blueprint_plugins")
    ctx = FakeCtx()
    housemate.register(ctx)
    for key, bp in ctx.blueprints.items():
        plugins.build_plugin_blueprint("housemate", key, **bp)


# --- tool handlers -------------------------------------------------------------------------------


def test_tools_return_json_when_hub_down():
    for name in TOOL_NAMES:
        out = tools.HANDLERS[name](dict(MINIMAL_ARGS[name]), task_id="t1", session_id="s1")
        assert isinstance(out, str)
        assert json.loads(out) == {"ok": False, "error": HUB_DOWN}, name


@pytest.mark.parametrize("args", [None, [], "text", {"seconds": object()}, {"rooms": 5, "text": 3}])
def test_tools_never_raise_on_garbage(args):
    for name in TOOL_NAMES:
        out = tools.HANDLERS[name](args)
        assert json.loads(out)["ok"] is False


class _StubHub:
    url = "https://127.0.0.1:8742"

    def online_devices(self):
        return []

    async def approve_pairing(self, code):
        return False


@pytest.fixture
async def services():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    store, registry = HousemateStore(conn), RoomRegistry()
    for name in ("Kitchen", "Office"):
        store.ensure_room(name)
    timers = TimerService(store, time.time)
    svc = HubServices(store=store, registry=registry, hub=_StubHub(), timers=timers, announcer=None)
    register_local_services(svc, asyncio.get_running_loop())
    yield svc
    timers.close()
    conn.close()


async def test_timer_set_tool_defaults_to_speaking_room(services, monkeypatch):
    monkeypatch.setattr(tools, "session_room", lambda: "Office")
    out = json.loads(await asyncio.to_thread(tools.HANDLERS["housemate_timer_set"], {"seconds": 90, "label": "tea"}))
    assert out["ok"] is True and out["timer"]["room"] == "office"
    assert [(t.room, t.label) for t in services.timers.list()] == [("office", "tea")]


async def test_timer_set_tool_asks_for_room_when_unknown(services):
    out = json.loads(await asyncio.to_thread(tools.HANDLERS["housemate_timer_set"], {"seconds": 90}))
    assert out["ok"] is False and "which room" in out["error"]


async def test_timer_list_and_cancel_tools(services):
    services.timers.add("kitchen", seconds=300, label="eggs")
    listed = json.loads(await asyncio.to_thread(tools.HANDLERS["housemate_timer_list"], {}))
    assert [t["label"] for t in listed["timers"]] == ["eggs"]
    done = json.loads(await asyncio.to_thread(tools.HANDLERS["housemate_timer_cancel"], {"label": "eggs"}))
    assert done["ok"] is True and services.timers.list() == []


async def test_announce_tool_accepts_all_as_list(services, monkeypatch):
    seen = {}

    class Ann:
        async def announce(self, text, rooms, from_room=None):
            seen.update(text=text, rooms=rooms, from_room=from_room)
            return SimpleNamespace(to_dict=lambda: {"ok": True, "rooms": ["office"]})

    services.announcer = Ann()
    monkeypatch.setattr(tools, "session_room", lambda: "Kitchen")
    out = json.loads(await asyncio.to_thread(tools.HANDLERS["housemate_announce"], {"text": "hi", "rooms": ["all"]}))
    assert out["ok"] is True
    assert seen == {"text": "hi", "rooms": "all", "from_room": "Kitchen"}


# --- slash commands and CLI ------------------------------------------------------------------------


async def test_housemate_command_shows_status(services):
    services.registry.attach("dev-k", "kitchen", DeviceConn("dev-k", "Kitchen phone"))
    services.timers.add("kitchen", seconds=250, label="pasta")
    ctx = FakeCtx()
    housemate.register(ctx)
    text = await ctx.commands["housemate"]["handler"]("")
    assert "Kitchen" in text and "Kitchen phone" in text and "pasta" in text and "Office" in text


async def test_housemate_command_when_hub_down():
    ctx = FakeCtx()
    housemate.register(ctx)
    text = await ctx.commands["housemate"]["handler"]("")
    assert HUB_DOWN in text


async def test_announce_command_needs_text():
    ctx = FakeCtx()
    housemate.register(ctx)
    assert "Usage" in await ctx.commands["announce"]["handler"]("   ")


async def test_announce_command_skips_origin_room(services):
    seen = {}

    class Ann:
        async def announce(self, text, rooms, from_room=None):
            seen.update(text=text, rooms=rooms, from_room=from_room)
            return SimpleNamespace(to_dict=lambda: {"ok": True, "rooms": ["office"], "offline": []})

    services.announcer = Ann()
    ctx = FakeCtx()
    housemate.register(ctx)
    source = SimpleNamespace(user_name="Kitchen", chat_id="household")
    text = await ctx.commands["announce"]["handler"]("Dinner!", platform="housemate", source=source)
    assert seen == {"text": "Dinner!", "rooms": "all", "from_room": "Kitchen"}
    assert "office" in text


def _cli(*argv):
    import argparse

    parser = argparse.ArgumentParser(prog="hermes housemate")
    cli.setup_parser(parser)
    args = parser.parse_args(list(argv))
    out = io.StringIO()
    code = cli.main(args, out=out)
    return code, out.getvalue()


def test_cli_parses_every_subcommand():
    import argparse

    parser = argparse.ArgumentParser(prog="hermes housemate")
    cli.setup_parser(parser)
    for argv in (["status"], ["pair", "123456"], ["devices"], ["revoke", "dev-1"], ["rooms", "add", "Living", "Room"],
                 ["rooms", "remove", "kitchen"], ["rooms", "group", "kitchen", "downstairs"], ["cert"], ["bench"], []):
        parser.parse_args(argv)


def test_cli_status_when_hub_down_explains():
    code, out = _cli("status")
    assert code == 1 and HUB_DOWN in out


def test_cli_bench_not_implemented():
    code, out = _cli("bench")
    assert "not implemented yet" in out


async def test_cli_pair_and_rooms_through_client(services):
    code, out = await asyncio.to_thread(_cli, "rooms", "add", "Living", "Room")
    assert code == 0 and "living-room" in out
    code, out = await asyncio.to_thread(_cli, "pair", "12-34-56")
    assert code == 1 and "123456" in out


def test_cli_rooms_offline_uses_store(monkeypatch, tmp_path):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    store = HousemateStore(conn)
    monkeypatch.setattr(cli, "open_store", lambda: store)
    code, out = _cli("rooms", "add", "Kitchen")
    assert code == 0 and "kitchen" in out
    code, out = _cli("rooms", "group", "kitchen", "downstairs")
    assert code == 0 and store.get_room("kitchen").earshot_group == "downstairs"
    code, out = _cli("devices")
    assert code == 0 and "hub not running" in out
    code, out = _cli("rooms", "remove", "kitchen")
    assert code == 0 and store.list_rooms() == []


def test_cli_cert_prints_url_fingerprint_and_qr(monkeypatch):
    monkeypatch.setattr(cli, "_hub_info", lambda client: {"ok": True, "urls": ["https://192.168.1.20:8742"],
                                                          "hub_url": "https://192.168.1.20:8742",
                                                          "fingerprint": "AB:CD:EF"})
    code, out = _cli("cert")
    assert code == 0
    assert "https://192.168.1.20:8742" in out and "AB:CD:EF" in out
    assert len(out.splitlines()) > 15  # the QR code


def test_qr_falls_back_to_plain_ascii():
    text = cli.render_qr("https://192.168.1.20:8742", ascii_only=True)
    assert text and set(text) <= {"#", " ", "\n"}
