import pytest

from adapter_kit import Kit, utterance

from gateway.platforms.base import ExecApprovalPrompt  # noqa: E402  (shimmed when Hermes is absent)

from housemate.core.answers import SIDE_CHAT, apply_answer, split_marker

SESSION = "agent:main:housemate:dm:household"


@pytest.fixture
async def kit(tmp_path):
    k = Kit(tmp_path).wire()
    k.add_device("dev-kitchen", "Kitchen", "Kitchen phone")
    k.add_device("dev-office", "Office", "Office laptop")
    return k


def approval(request_id="req-1"):
    return ExecApprovalPrompt(
        chat_id="household", session_key=SESSION,
        text="Hermes wants to run a command that needs your OK",
        actions=[("Allow Once", "once", "primary"), ("Deny", "deny", "danger")],
        command="rm -rf ./junk", description="recursive delete", smart_denied=False, request_id=request_id,
    )


async def start_turn(kit, device="dev-kitchen"):
    kit.adapter._on_utterance(device, utterance(), "wake")
    await kit.settle()
    return kit.events[-1]


async def ask_approval(kit, request_id="req-1"):
    await start_turn(kit)
    result = await kit.adapter._send_exec_approval_prompt(approval(request_id))
    await kit.settle()
    return result


async def answer_by_voice(kit, transcript: str, device="dev-kitchen"):
    """The device speaks; the clip goes to the side chat; Hermes transcribes it; our hook runs."""
    before = len(kit.events)
    kit.adapter._on_utterance(device, utterance(0.6), "followup")
    await kit.settle()
    side = kit.events[before]
    assert side.source.chat_id == SIDE_CHAT and side.source.user_id == device
    token, _ = split_marker(side.channel_context)
    out = apply_answer(kit.adapter.book, {"platform": "housemate", "user_message": f'"{transcript}"\n\n{side.channel_context}'})
    await kit.settle()
    return out


# --- the card ----------------------------------------------------------------------------------------


async def test_approval_prompt_reaches_only_origin_room(kit):
    result = await ask_approval(kit)
    assert result.success and result.message_id == "req-1"
    prompts = kit.hub.controls("dev-kitchen", "prompt")
    assert prompts and prompts[-1]["kind"] == "approval" and prompts[-1]["id"] == "req-1"
    assert prompts[-1]["choices"] == ["Approve", "Deny"]
    assert "recursive delete" in prompts[-1]["text"]
    assert kit.hub.controls("dev-office", "prompt") == []
    # and it is spoken on the asking device only
    assert any(m["stream_id"].startswith("ask-") for m in kit.hub.controls("dev-kitchen", "play.begin"))
    assert kit.hub.controls("dev-office", "play.begin") == []


async def test_answer_window_opens_after_the_question_was_heard(kit):
    await ask_approval(kit)
    listener = kit.listeners["dev-kitchen"]
    assert listener.called("open_followup") == []
    sid = [m for m in kit.hub.controls("dev-kitchen", "play.begin") if m["stream_id"].startswith("ask-")][-1]["stream_id"]
    await kit.adapter._on_control("dev-kitchen", {"type": "playback.done", "stream_id": sid})
    assert listener.called("open_followup") == [(10.0,)]


async def test_tap_approve_resolves_card_with_request_id(kit):
    await ask_approval(kit)
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "prompt", "id": "req-1", "value": "Approve"})
    await kit.settle()
    assert kit.resolved == [(SESSION, "once", "req-1")]
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "req-1"
    # a second tap finds no card and resolves nothing
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "prompt", "id": "req-1", "value": "Approve"})
    assert kit.resolved == [(SESSION, "once", "req-1")]


async def test_tap_deny(kit):
    await ask_approval(kit)
    await kit.adapter._on_control("dev-office", {"type": "ui.action", "kind": "deny", "id": "req-1"})
    assert kit.resolved == [(SESSION, "deny", "req-1")]


# --- spoken answers ------------------------------------------------------------------------------


async def test_spoken_approve_goes_through_side_chat(kit):
    await ask_approval(kit)
    out = await answer_by_voice(kit, "Approve.")
    assert out["action"] == "block"
    assert kit.resolved == [(SESSION, "once", "req-1")]
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "req-1"
    # the main household turn was never touched by the answer
    assert kit.adapter.router.active_turn("household") is not None
    assert [e.source.chat_id for e in kit.events].count("household") == 1


async def test_spoken_deny(kit):
    await ask_approval(kit)
    await answer_by_voice(kit, "No, don't do it.")
    assert kit.resolved == [(SESSION, "deny", "req-1")]


async def test_unanswered_approval_never_resolves_approve(kit):
    await ask_approval(kit)
    for said in ("maybe", "what was that", "hmm"):
        await answer_by_voice(kit, said)
    assert kit.resolved == []  # not understood -> re-asked, never approved
    # A fourth try is no longer routed as an answer (re-prompts are capped).
    assert kit.adapter.book.for_device("dev-kitchen") is None
    # Hermes's own timeout denies and edits the card: we only clear it.
    await kit.adapter.edit_message("household", "req-1", "⌛ Approval timed out after 5 minutes - the command was NOT run.")
    await kit.settle()
    assert kit.resolved == []
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "req-1"
    assert "req-1" not in kit.adapter._prompts


async def test_other_room_cannot_answer_by_voice(kit):
    await ask_approval(kit)
    kit.adapter._on_utterance("dev-office", utterance(), "wake")
    await kit.settle()
    # the office speech is an ordinary (queued) question, not an answer
    assert all(e.source.chat_id == "household" for e in kit.events)
    assert kit.resolved == []


# --- clarify -----------------------------------------------------------------------------------------


async def test_clarify_spoken_choice_resolves(kit):
    await start_turn(kit)
    result = await kit.adapter.send_clarify("household", "Which dinner?", ["Pasta", "Pizza"], "clar-1", SESSION)
    await kit.settle()
    assert result.success
    prompt = kit.hub.controls("dev-kitchen", "prompt")[-1]
    assert prompt["kind"] == "clarify" and prompt["choices"] == ["Pasta", "Pizza"]
    assert "1: Pasta" in kit.tts.texts[-1]
    await answer_by_voice(kit, "The second one.")
    assert kit.clarified == [("clar-1", "Pizza")]


async def test_clarify_tap_and_retire(kit):
    await start_turn(kit)
    await kit.adapter.send_clarify("household", "Which dinner?", ["Pasta", "Pizza"], "clar-2", SESSION)
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "prompt", "id": "clar-2", "value": "Pasta"})
    assert kit.clarified == [("clar-2", "Pasta")]
    await kit.adapter.send_clarify("household", "Anything else?", None, "clar-3", SESSION)
    await kit.adapter.retire_clarify_card("clar-3", "Timed out")
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "clar-3"


async def test_open_clarify_takes_the_whole_answer(kit):
    await start_turn(kit)
    await kit.adapter.send_clarify("household", "What should I call the timer?", None, "clar-4", SESSION)
    await kit.settle()
    await answer_by_voice(kit, "Lasagne.")
    assert kit.clarified == [("clar-4", "Lasagne.")]


async def test_clarify_speaks_without_the_typed_skip_hint(kit):
    await start_turn(kit)
    question = 'Would you like tea or coffee?\nReply "skip" to skip this question.'
    await kit.adapter.send_clarify("household", question, ["Tea (Recommended)", "Coffee"], "clar-5", SESSION)
    await kit.settle()
    prompt = kit.hub.controls("dev-kitchen", "prompt")[-1]
    assert prompt["text"] == "Would you like tea or coffee?"
    assert prompt["choices"] == ["Tea (Recommended)", "Coffee"]  # the card keeps Hermes's labels
    assert kit.tts.texts[-1] == "Would you like tea or coffee? 1: Tea; 2: Coffee. Or say skip."


async def test_clarify_skip_by_voice(kit):
    from housemate.adapter import clarify_skip_value

    await start_turn(kit)
    await kit.adapter.send_clarify("household", 'Which dinner?\nReply "skip" to skip this question.',
                                   ["Pasta", "Pizza"], "clar-6", SESSION)
    await kit.settle()
    await answer_by_voice(kit, "Skip it.")
    assert kit.clarified == [("clar-6", clarify_skip_value())]
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "clar-6"


# --- ringing timers ------------------------------------------------------------------------------------


async def test_spoken_stop_while_ringing(kit):
    kit.timers.ringing.add("kitchen")
    await kit.adapter._on_timer_ring(type("T", (), {"room": "kitchen"})())
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "ringing"
    await answer_by_voice(kit, "Okay, stop.")
    assert kit.timers.stopped == ["kitchen"]
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "idle"
    assert kit.adapter.book.for_device("dev-kitchen") is None


async def test_ring_window_ignores_other_speech(kit):
    kit.timers.ringing.add("kitchen")
    await kit.adapter._on_timer_ring(type("T", (), {"room": "kitchen"})())
    await answer_by_voice(kit, "what a lovely day")
    assert kit.timers.stopped == []
    assert kit.adapter.book.for_device("dev-kitchen").kind == "stop"  # still listening for "stop"


async def test_wake_word_stops_a_ring(kit):
    kit.timers.ringing.add("kitchen")
    await kit.adapter._on_timer_ring(type("T", (), {"room": "kitchen"})())
    kit.adapter._on_wake("dev-kitchen")
    assert kit.timers.stopped == ["kitchen"]
    assert kit.adapter.book.for_device("dev-kitchen") is None  # the next utterance is a normal question


async def test_tap_stop_timer(kit):
    kit.timers.ringing.add("kitchen")
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "stop_timer", "id": ""})
    assert kit.timers.stopped == ["kitchen"]


async def test_approval_with_no_device_online_is_not_armed(kit):
    await start_turn(kit)
    kit.drop("dev-kitchen")
    kit.drop("dev-office")
    result = await kit.adapter._send_exec_approval_prompt(approval("req-9"))
    assert result.success is False
    assert "req-9" not in kit.adapter._prompts
    assert kit.adapter.book.entries() == []
    assert kit.tts.texts == []  # nothing spoken to nobody


async def test_approval_reachable_by_voice_while_a_timer_rings(kit):
    await ask_approval(kit)
    kit.timers.ringing.add("kitchen")
    await kit.adapter._on_timer_ring(type("T", (), {"room": "kitchen"})())
    await answer_by_voice(kit, "Approve.")
    assert kit.resolved == [(SESSION, "once", "req-1")]
    assert kit.timers.stopped == []


async def test_resolver_error_in_the_hook_still_clears_the_card(kit):
    await ask_approval(kit)

    def broken(*args):
        raise RuntimeError("approval store went away")

    kit.adapter._resolve_approval = broken
    out = await answer_by_voice(kit, "Approve.")
    assert out["action"] == "block"
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "req-1"
