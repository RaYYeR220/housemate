"""Contract checks against a real Hermes install (skipped when Hermes is not importable)."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import hermes_shim

pytestmark = [
    pytest.mark.hermes,
    pytest.mark.skipif(not hermes_shim.HERMES_AVAILABLE, reason="needs a Hermes >= 0.23 install"),
]

ROOT = Path(__file__).resolve().parents[1]


class Ctx:
    def __init__(self):
        self.manifest = SimpleNamespace(name="housemate")
        self.platforms, self.hooks, self.tools = [], {}, {}

    def register_platform(self, **kwargs):
        self.platforms.append(kwargs)

    def register_hook(self, name, cb):
        self.hooks[name] = cb

    def register_tool(self, name, **kwargs):
        self.tools[name] = kwargs

    def register_command(self, *a, **k):
        pass

    def register_cli_command(self, *a, **k):
        pass

    def register_automation_blueprint(self, *a, **k):
        pass

    def register_skill(self, *a, **k):
        pass

    def register_tts_provider(self, provider):
        self.tts = provider


def test_adapter_subclasses_the_real_base_and_renders_approvals():
    import adapter_kit  # noqa: F401  (registers the platform name)
    from gateway.platforms.base import BasePlatformAdapter

    from housemate.adapter import HousemateAdapter

    assert issubclass(HousemateAdapter, BasePlatformAdapter)
    assert HousemateAdapter.supports_exec_approval_buttons() is True
    assert callable(getattr(HousemateAdapter, "retire_clarify_card", None))


def test_registration_matches_real_platform_entry_and_manifest():
    import housemate
    from gateway.platform_registry import PlatformEntry

    ctx = Ctx()
    housemate.register(ctx)
    (kwargs,) = ctx.platforms
    entry = PlatformEntry(**kwargs)  # every keyword we pass is a real field
    assert entry.name == "housemate" and entry.allowed_users_env == "HOUSEMATE_ALLOWED_DEVICES"
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    assert sorted(ctx.hooks) == sorted(manifest["provides_hooks"]) == ["pre_llm_call"]
    assert ctx.hooks["pre_llm_call"](platform="telegram", user_message="hi") is None


def test_streaming_piper_registers_as_a_real_tts_provider():
    import housemate
    from agent.tts_provider import TTSProvider

    ctx = Ctx()
    housemate.register(ctx)
    assert isinstance(ctx.tts, TTSProvider) and ctx.tts.name == "housemate-piper" and ctx.tts.streams_pcm


# Hermes's approver check for taps and spoken answers: on the published plugin API after 0.23
# (``hermes_cli/data/plugin_api.json``: ``gateway.approval_actor:<name>``), absent from 0.23.0, whose
# validator lists only what it ships. The adapter imports them lazily and works without them.
NEWER_PUBLIC = {"gateway.approval_actor:approval_refusal", "gateway.approval_actor:clarify_refusal",
                "gateway.approval_actor:refusal_text"}


def test_only_public_hermes_surfaces():
    import importlib.util
    import json

    import hermes_cli
    from hermes_cli.plugin_validate_private_imports import private_attribute_findings, private_import_findings

    findings = private_import_findings(ROOT)
    if importlib.util.find_spec("gateway.approval_actor") is None:  # a Hermes from before the approver check
        newer = {name.replace(":", ".") for name in NEWER_PUBLIC}
        findings = [f for f in findings if f.split(" ", 1)[-1] not in newer]
    assert findings == []
    assert private_attribute_findings(ROOT) == []
    api = Path(hermes_cli.__file__).resolve().parent / "data" / "plugin_api.json"
    if api.is_file():  # a Hermes that publishes its plugin API lists them
        data = json.loads(api.read_text(encoding="utf-8-sig"))
        assert NEWER_PUBLIC <= {entry for group in data["groups"] for entry in group["entries"]}


async def test_voice_event_built_with_real_classes(tmp_path):
    from adapter_kit import Kit, utterance
    from gateway.platforms.event import MessageEvent, MessageType

    kit = Kit(tmp_path).wire()
    kit.add_device("dev-kitchen", "Kitchen")
    kit.adapter._on_utterance("dev-kitchen", utterance(), "wake")
    await kit.settle()
    (event,) = kit.events
    assert isinstance(event, MessageEvent) and event.message_type is MessageType.VOICE
    assert event.metadata == {}  # 0.23 refuses events with custom metadata keys
    assert event.channel_context == "[room: Kitchen]"
    assert kit.adapter._event_session_key(event).endswith(":dm:household")
