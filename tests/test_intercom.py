"""Satellite-facing additions: the intercom tap (a voice message to another room), the approval
card's extra fields and the facts the hub sends with ``paired``."""

import sqlite3
from pathlib import Path

import pytest

from adapter_kit import Kit, utterance

from gateway.platforms.base import ExecApprovalPrompt  # noqa: E402  (shimmed when Hermes is absent)

from housemate.adapter import wake_phrase
from housemate.core.hub import Hub
from housemate.core.rooms import RoomRegistry
from housemate.core.store import HousemateStore

from fake_satellite import FakeSatellite


@pytest.fixture
async def kit(tmp_path):
    k = Kit(tmp_path).wire()
    k.add_device("dev-kitchen", "Kitchen", "Kitchen phone")
    k.add_device("dev-office", "Office", "Office laptop")
    return k


async def test_intercom_tap_plays_the_next_utterance_in_the_other_room(kit):
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "intercom", "id": "office", "value": "Office"})
    await kit.settle()
    assert kit.listeners["dev-kitchen"].called("open_followup")
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "listening"

    clip = utterance(0.8)
    kit.adapter._on_utterance("dev-kitchen", clip, "followup")
    await kit.settle()

    assert kit.events == []  # a voice message never goes to Hermes
    office = kit.hub.controls("dev-office")
    kinds = [m["type"] for m in office]
    assert {"type": "chime", "kind": "announce"} in office
    assert {"type": "caption", "role": "intercom", "text": "Kitchen"} in office
    begin = next(m for m in office if m["type"] == "play.begin")
    assert begin["sample_rate"] == 16000 and begin["stream_id"].startswith("ann-")
    assert kinds.index("play.begin") < kinds.index("play.end")
    assert kit.hub.pcm("dev-office") == clip.pcm
    assert {"type": "caption", "role": "status", "text": "Sent to the Office."} in kit.hub.controls("dev-kitchen")
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "idle"
    assert kit.hub.pcm("dev-kitchen") == b""


async def test_intercom_to_an_unknown_or_own_room_is_refused(kit):
    for target in ("garage", "kitchen"):
        await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "intercom", "id": target})
    await kit.settle()
    notes = [m["text"] for m in kit.hub.controls("dev-kitchen", "caption")]
    assert notes == ["There is no other room called garage.", "There is no other room called kitchen."]
    assert not kit.listeners["dev-kitchen"].called("open_followup")
    kit.adapter._on_utterance("dev-kitchen", utterance(), "wake")
    await kit.settle()
    assert len(kit.events) == 1  # an ordinary question


async def test_a_wake_word_after_the_intercom_tap_is_an_ordinary_question(kit):
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "intercom", "id": "office"})
    kit.adapter._on_wake("dev-kitchen")
    kit.adapter._on_utterance("dev-kitchen", utterance(), "wake")
    await kit.settle()
    assert len(kit.events) == 1
    assert kit.hub.pcm("dev-office") == b""


async def test_intercom_to_a_room_with_nobody_online(kit):
    kit.drop("dev-office")
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "intercom", "id": "office"})
    kit.adapter._on_utterance("dev-kitchen", utterance(), "followup")
    await kit.settle()
    assert {"type": "caption", "role": "status", "text": "No speaker is on in the Office."} in kit.hub.controls("dev-kitchen")
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "idle"


async def test_approval_card_carries_title_detail_and_command(kit):
    kit.adapter._on_utterance("dev-kitchen", utterance(), "wake")
    await kit.settle()
    prompt = ExecApprovalPrompt(
        chat_id="household", session_key="agent:main:housemate:dm:household", text="needs your OK",
        actions=[("Allow Once", "once", "primary"), ("Deny", "deny", "danger")],
        command="rm -rf  ~/junkbox", description="recursive delete", smart_denied=False, request_id="req-9",
    )
    await kit.adapter._send_exec_approval_prompt(prompt)
    await kit.settle()
    card = kit.hub.controls("dev-kitchen", "prompt")[-1]
    assert card["title"] == "Run a command?" and card["detail"] == "recursive delete"
    assert card["command"] == "rm -rf  ~/junkbox" and card["choices"] == ["Approve", "Deny"]  # as it will run


def test_satellite_info_follows_the_mode(kit):
    info = kit.adapter._satellite_info()
    assert info["wake"] == "Hey Jarvis" and info["conversation"] == "household" and info["hub_name"]
    kit.adapter.router.mode = "per_room"
    assert kit.adapter._satellite_info()["conversation"] == "per_room"


def test_wake_phrase_from_model_names():
    assert wake_phrase("hey_jarvis") == "Hey Jarvis"
    assert wake_phrase("hey_jarvis_v0.1.onnx") == "Hey Jarvis"
    assert wake_phrase(r"C:\models\hey_hermes.onnx") == "Hey Hermes"
    assert wake_phrase("") == ""


async def test_paired_message_carries_hub_info():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    store = HousemateStore(conn)
    store.ensure_room("Kitchen")
    hub = Hub(store, RoomRegistry(), host="127.0.0.1", port=0, cert=None, key=None,
              web_root=Path(__file__).resolve().parents[1] / "web", tls=False)
    hub.info = lambda: {"wake": "Hey Jarvis", "conversation": "household", "hub_name": "office-pc"}
    await hub.start()
    sat = FakeSatellite(hub.ws_url, "dev-new1", "New phone")
    try:
        await sat.connect()
        code = (await sat.recv_control(type="pair.code"))["code"]
        assert await hub.approve_pairing(code)
        paired = await sat.recv_control(type="paired")
        assert paired["wake"] == "Hey Jarvis" and paired["hub_name"] == "office-pc"
        assert paired["conversation"] == "household" and paired["device_key"]
        assert paired["rooms"] == [{"slug": "kitchen", "name": "Kitchen"}]
    finally:
        await sat.close()
        await hub.stop()
