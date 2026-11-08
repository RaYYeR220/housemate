import time

import pytest

from adapter_kit import Kit, utterance

from gateway.platforms.base import ExecApprovalPrompt  # noqa: E402  (shimmed when Hermes is absent)

from housemate.adapter import ECHO_TAIL_S
from housemate.core.answers import SIDE_CHAT, apply_answer, split_marker

SESSION = "agent:main:housemate:dm:household"


@pytest.fixture
async def kit(tmp_path):
    k = Kit(tmp_path).wire()
    k.add_device("dev-kitchen", "Kitchen", "Kitchen phone")
    k.add_device("dev-office", "Office", "Office laptop")
    return k


def approval(request_id="req-1", *, smart_denied=False):
    return ExecApprovalPrompt(
        chat_id="household", session_key=SESSION,
        text="Hermes wants to run a command that needs your OK",
        actions=[("Allow Once", "once", "primary"), ("Deny", "deny", "danger")],
        command="rm -rf ./junk", description="recursive delete", smart_denied=smart_denied, request_id=request_id,
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


async def quiet(kit):
    """Every device reports that what it played has been heard (the answer window opens)."""
    for device_id, playing in list(kit.adapter._playing.items()):
        await kit.adapter._on_control(device_id, {"type": "playback.done", "stream_id": playing.stream_id})
    await kit.settle()


def fresh_utterance(seconds: float = 0.6):
    """Speech that began after the echo tail of the last sound we played."""
    utt = utterance(seconds)
    utt.started_at = time.time() + ECHO_TAIL_S + 0.05
    utt.ended_at = utt.started_at + seconds
    return utt


async def answer_by_voice(kit, transcript: str, device="dev-kitchen"):
    """The device speaks; the clip goes to the side chat; Hermes transcribes it; our hook runs."""
    await quiet(kit)
    before = len(kit.events)
    kit.adapter._on_utterance(device, fresh_utterance(), "followup")
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
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "deny", "id": "req-1"})
    assert kit.resolved == [(SESSION, "deny", "req-1")]


# --- who may tap (review M1) ----------------------------------------------------------------------


async def test_tap_from_another_room_is_ignored(kit):
    await ask_approval(kit)
    await kit.adapter._on_control("dev-office", {"type": "ui.action", "kind": "approve", "id": "req-1"})
    await kit.settle()
    assert kit.resolved == []
    assert "req-1" in kit.adapter._prompts


async def test_tap_from_a_device_outside_the_allowlist_is_ignored(kit, monkeypatch):
    monkeypatch.setenv("HOUSEMATE_ALLOWED_DEVICES", "dev-kitchen")
    kit.add_device("dev-kid", "Kitchen", "Kid tablet")
    await ask_approval(kit)
    assert kit.hub.controls("dev-kid", "prompt") == []  # the card only goes to allowed devices
    await kit.adapter._on_control("dev-kid", {"type": "ui.action", "kind": "approve", "id": "req-1"})
    await kit.adapter._on_control("dev-kid", {"type": "ui.action", "kind": "prompt", "id": "req-1", "value": "Approve"})
    await kit.settle()
    assert kit.resolved == []
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "deny", "id": "req-1"})
    assert kit.resolved == [(SESSION, "deny", "req-1")]


async def test_tap_from_a_device_that_left_the_room_is_ignored(kit):
    kit.add_device("dev-tablet", "Kitchen", "Kitchen tablet")
    await ask_approval(kit)
    assert kit.hub.controls("dev-tablet", "prompt")[-1]["id"] == "req-1"
    kit.registry.attach("dev-tablet", "office", kit.registry.get("dev-tablet"))  # room.set to the office
    await kit.adapter._on_control("dev-tablet", {"type": "ui.action", "kind": "approve", "id": "req-1"})
    assert kit.resolved == []


async def test_clarify_tap_from_another_room_is_ignored(kit):
    await start_turn(kit)
    await kit.adapter.send_clarify("household", "Which dinner?", ["Pasta", "Pizza"], "clar-9", SESSION)
    await kit.settle()
    await kit.adapter._on_control("dev-office", {"type": "ui.action", "kind": "prompt", "id": "clar-9", "value": "Pasta"})
    assert kit.clarified == []


# --- the answer must be the user's, not our own audio (review H1) --------------------------------


async def test_hedged_spoken_answers_never_approve(kit):
    await ask_approval(kit)
    for said in ("Yes, but what will it delete first?", "Yeah, wait, what does that do?"):
        await answer_by_voice(kit, said)
    assert kit.resolved == []


async def test_answer_heard_while_another_room_plays_is_not_an_answer(kit):
    """A ring or announcement in a neighbouring room (no shared earshot group) is captured by the
    asking device: never sent to speech-to-text as an answer."""
    await ask_approval(kit)
    await quiet(kit)
    kit.adapter._observe_control("dev-office", {"type": "play.begin", "stream_id": "say-ring1", "sample_rate": 24000})
    before = len(kit.events)
    kit.adapter._on_utterance("dev-kitchen", fresh_utterance(), "followup")
    await kit.settle()
    assert len(kit.events) == before
    assert kit.resolved == []
    assert "req-1" in kit.adapter._prompts  # still open: tap, or answer again once it is quiet


async def test_answer_right_after_our_own_audio_is_not_an_answer(kit):
    await ask_approval(kit)
    await quiet(kit)  # the question was heard just now
    before = len(kit.events)
    kit.adapter._on_utterance("dev-kitchen", utterance(0.6), "followup")  # began within the echo tail
    await kit.settle()
    assert len(kit.events) == before
    assert kit.resolved == []
    # a clean answer afterwards still works
    await answer_by_voice(kit, "Approve.")
    assert kit.resolved == [(SESSION, "once", "req-1")]


async def test_a_chime_counts_as_our_own_audio(kit):
    await ask_approval(kit)
    await quiet(kit)
    kit.adapter._observe_control("dev-office", {"type": "chime", "kind": "timer"})
    before = len(kit.events)
    kit.adapter._on_utterance("dev-kitchen", utterance(0.6), "followup")  # started during the chime
    await kit.settle()
    assert len(kit.events) == before


async def test_timer_labelled_approve_cannot_approve(kit):
    """Review H1 exploit: housemate_timer_set(label="approve") rings "Your approve timer is done."
    Even transcribed after the echo tail, the ring text is not an answer."""
    await ask_approval(kit)
    out = await answer_by_voice(kit, "Your approve timer is done.")
    assert out["action"] == "block"
    await answer_by_voice(kit, "Your go ahead timer is done.")
    assert kit.resolved == []


async def test_words_we_announced_are_not_an_answer(kit):
    """housemate_announce("Approve.") into another room: if its audio slips past the echo guard, the
    transcript is what we said ourselves a moment ago, so it is not an answer."""
    await ask_approval(kit)
    await kit.adapter.announcer.say("office", "Approve.")
    await kit.settle()
    await answer_by_voice(kit, "Approve.")
    assert kit.resolved == []


async def test_smart_denied_approval_is_tap_only(kit):
    await start_turn(kit)
    result = await kit.adapter._send_exec_approval_prompt(approval("req-s", smart_denied=True))
    await kit.settle()
    assert result.success
    assert kit.adapter.book.for_device("dev-kitchen") is None  # nothing spoken can answer it
    spoken = kit.tts.texts[-1].lower()
    assert "flagged" in spoken and "tap" in spoken
    assert "flagged" in kit.hub.controls("dev-kitchen", "prompt")[-1]["text"].lower()
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "approve", "id": "req-s"})
    assert kit.resolved == [(SESSION, "once", "req-s")]


# --- one prompt at a time per room (review H2) ------------------------------------------------------


async def test_spoken_answer_resolves_the_prompt_that_was_asked(kit):
    """Two approvals pending on one device: "approve" answers the one on screen and spoken (A),
    never the newest (B). B is shown and asked once A is settled."""
    await ask_approval(kit, "req-A")
    await kit.adapter._send_exec_approval_prompt(approval("req-B"))
    await kit.settle()
    assert [m["id"] for m in kit.hub.controls("dev-kitchen", "prompt")] == ["req-A"]
    assert sum("recursive delete" in t for t in kit.tts.texts) == 1
    await answer_by_voice(kit, "Approve.")
    assert kit.resolved == [(SESSION, "once", "req-A")]
    await kit.settle()
    assert [m["id"] for m in kit.hub.controls("dev-kitchen", "prompt")] == ["req-A", "req-B"]
    await answer_by_voice(kit, "Deny.")
    assert kit.resolved == [(SESSION, "once", "req-A"), (SESSION, "deny", "req-B")]


async def test_three_pending_approvals_are_answered_in_order(kit):
    await ask_approval(kit, "req-A")
    for rid in ("req-B", "req-C"):
        await kit.adapter._send_exec_approval_prompt(approval(rid))
    await kit.settle()
    await answer_by_voice(kit, "Approve.")
    await answer_by_voice(kit, "Deny.")
    await answer_by_voice(kit, "Approve.")
    assert [r[2] for r in kit.resolved] == ["req-A", "req-B", "req-C"]
    assert [r[1] for r in kit.resolved] == ["once", "deny", "once"]


async def test_a_queued_prompt_cannot_be_tapped_before_it_is_shown(kit):
    await ask_approval(kit, "req-A")
    await kit.adapter._send_exec_approval_prompt(approval("req-B"))
    await kit.settle()
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "approve", "id": "req-B"})
    assert kit.resolved == []


async def test_timeout_of_the_first_prompt_brings_up_the_next(kit):
    await ask_approval(kit, "req-A")
    await kit.adapter._send_exec_approval_prompt(approval("req-B"))
    await kit.settle()
    await kit.adapter.edit_message("household", "req-A", "Approval timed out after 5 minutes - the command was NOT run.")
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "prompt")[-1]["id"] == "req-B"
    await answer_by_voice(kit, "Approve.")
    assert kit.resolved == [(SESSION, "once", "req-B")]


async def test_hello_replays_only_the_prompt_on_screen(kit):
    await ask_approval(kit, "req-A")
    await kit.adapter._send_exec_approval_prompt(approval("req-B"))
    await kit.settle()
    kit.hub.calls.clear()
    await kit.adapter._on_hello("dev-kitchen")
    assert [m["id"] for m in kit.hub.controls("dev-kitchen", "prompt")] == ["req-A"]


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


# --- the household queue waits for the prompt (review M2) ----------------------------------------


async def test_open_approval_holds_the_household_turn_past_its_timeout(kit):
    await ask_approval(kit)
    kit.adapter._on_utterance("dev-office", utterance(), "wake")  # queued behind the kitchen turn
    await kit.settle()
    base = kit.adapter.router._clock
    kit.adapter.router._clock = lambda: base() + 10_000
    kit.adapter._watch_step()
    await kit.settle()
    assert [e.source.chat_id for e in kit.events].count("household") == 1  # the office waits
    assert kit.adapter.router.active_turn("household").device_id == "dev-kitchen"
    assert "req-1" in kit.adapter._prompts


def test_default_turn_timeout_outlasts_hermes_approval_timeout(monkeypatch):
    from gateway.config import PlatformConfig

    from housemate import adapter as adapter_mod

    monkeypatch.setattr(adapter_mod, "_hermes_approval_timeout", lambda: 600.0)
    adapter = adapter_mod.HousemateAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter.router.turn_timeout_s > 600
    monkeypatch.setattr(adapter_mod, "_hermes_approval_timeout", lambda: 300.0)
    adapter = adapter_mod.HousemateAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter.router.turn_timeout_s > 300
    # an explicit setting is kept (the open-prompt hold still protects approvals)
    adapter = adapter_mod.HousemateAdapter(PlatformConfig(enabled=True, extra={"turn_timeout_s": 120}))
    assert adapter.router.turn_timeout_s == 120


# --- prompts settled outside Housemate (review L7) ------------------------------------------------


async def test_turn_end_retires_its_open_prompts(kit):
    """An approval answered from Desktop or cancelled by /stop leaves no stale card, no armed answer,
    and no prompt blocking the next one."""
    first = await start_turn(kit)
    await kit.adapter._send_exec_approval_prompt(approval("req-1"))
    await kit.adapter._send_exec_approval_prompt(approval("req-2"))
    await kit.settle()
    await kit.adapter.on_processing_complete(first, "success")
    await kit.settle()
    assert kit.adapter._prompts == {}
    assert kit.adapter.book.for_device("dev-kitchen") is None
    assert kit.hub.controls("dev-kitchen", "prompt.clear")[-1]["id"] == "req-1"
    assert [m["id"] for m in kit.hub.controls("dev-kitchen", "prompt")] == ["req-1"]  # req-2 never came up


async def test_turn_end_keeps_prompts_of_a_newer_turn(kit):
    first = await start_turn(kit)
    await kit.adapter.send("household", "Done.", reply_to=first.message_id)
    await kit.settle()
    second = await start_turn(kit)
    await kit.adapter._send_exec_approval_prompt(approval("req-3"))
    await kit.settle()
    await kit.adapter.on_processing_complete(first, "success")  # late signal for the first turn
    assert "req-3" in kit.adapter._prompts
    assert second.message_id != first.message_id
