"""A gateway restart that cuts a household turn short.

Hermes fences a person's turn that was running when the gateway died (a kill, a crash, a closed
window) and pauses that conversation until the person sends ``/discard`` in the chat (GW-10). It
says so once, in a text notice, and admits later messages silently. Nobody can type ``/discard`` to
a kitchen speaker, so the adapter turns the notice into a spoken one-word recovery: "continue"
(said or tapped) sends ``/discard`` from that device, Hermes drops the lost turn (never re-runs it,
keeps it in history) and answers the saved question.
"""

import time

import pytest

from adapter_kit import Kit, utterance

from gateway.platforms.event import MessageType  # noqa: E402  (shimmed when Hermes is absent)

from housemate.adapter import ECHO_TAIL_S
from housemate.core.answers import SIDE_CHAT, apply_answer, split_marker
from housemate.core.intents import is_continue

# Hermes 0.23 gateway/session_ingress.py pause_notice(reason="unknown_execution"), verbatim.
NOTICE = ("⏸️ This conversation is paused: a previous turn did not finish when Hermes restarted, "
          "and only you know whether to send it again. Your message is saved. Send /discard to drop "
          "the lost turn and continue (it is not re-run), or /reset to start a fresh conversation.")
OTHER_PAUSE = ("⏸️ This conversation is paused: a queued message could not be re-authorized (policy_changed). "
               "Your message is saved but will not be answered here. Send /reset to start a fresh "
               "conversation, or ask the operator to resume this one.")
DISCARDED = "🗑️ Discarded 1 lost turn(s); they were not re-run. Your saved messages will be answered now."


@pytest.fixture
async def kit(tmp_path):
    k = Kit(tmp_path).wire()
    k.add_device("dev-kitchen", "Kitchen", "Kitchen phone")
    k.add_device("dev-office", "Office", "Office laptop")
    return k


async def ask(kit, device="dev-kitchen", kind="wake"):
    kit.adapter._on_utterance(device, utterance(), kind)
    await kit.settle()
    return kit.events[-1]


async def paused(kit, device="dev-kitchen"):
    """A question reaches Hermes after a restart that lost a turn: Hermes answers with the notice."""
    first = await ask(kit, device)
    await kit.adapter.send("household", NOTICE, reply_to=first.message_id)
    await kit.adapter.on_processing_complete(first, "success")
    await kit.settle()
    return first


async def quiet(kit):
    for device_id, playing in list(kit.adapter._playing.items()):
        await kit.adapter._on_control(device_id, {"type": "playback.done", "stream_id": playing.stream_id})
    await kit.settle()


def fresh_utterance(seconds: float = 0.6):
    utt = utterance(seconds)
    utt.started_at = time.time() + ECHO_TAIL_S + 0.05
    utt.ended_at = utt.started_at + seconds
    return utt


async def say(kit, transcript, device="dev-kitchen", kind="followup"):
    """The device speaks while the chat is paused: the clip goes to the side chat, Hermes
    transcribes it and our hook reads it."""
    await quiet(kit)
    before = len(kit.events)
    kit.adapter._on_utterance(device, fresh_utterance(), kind)
    await kit.settle()
    side = kit.events[before]
    assert side.source.chat_id == SIDE_CHAT and side.source.user_id == device
    out = apply_answer(kit.adapter.book, {"platform": "housemate", "user_message": f'"{transcript}"\n\n{side.channel_context}'})
    await kit.settle()
    return out


def discards(kit, chat="household"):
    return [e for e in kit.events if e.message_type == MessageType.TEXT and e.text == "/discard"
            and e.source.chat_id == chat]


def prompts(kit, device):
    return [m for m in kit.hub.controls(device, "prompt") if m["kind"] == "resume"]


# --- the notice ------------------------------------------------------------------------------------


async def test_the_pause_notice_becomes_a_card_and_a_spoken_question(kit):
    first = await paused(kit)
    cards = prompts(kit, "dev-kitchen")
    assert len(cards) == 1 and cards[0]["choices"] == ["Continue"]
    assert "/discard" not in str(cards[0]) and "/reset" not in str(cards[0])
    assert all("/discard" not in m["text"] for m in kit.hub.controls("dev-kitchen", "caption"))
    assert prompts(kit, "dev-office") == []
    # spoken on the asking device, in plain words
    assert any(m["stream_id"].startswith("ask-") for m in kit.hub.controls("dev-kitchen", "play.begin"))
    assert "continue" in kit.tts.texts[-1].lower() and "/" not in kit.tts.texts[-1]
    # nothing is discarded behind the person's back, and the question stays the running turn
    assert discards(kit) == []
    assert kit.adapter.router.active_turn("household").message_id == first.message_id


async def test_hermes_reads_out_plain_words_instead_of_the_notice(kit):
    spoken = kit.adapter.prepare_tts_text(NOTICE)
    assert "continue" in spoken.lower()
    assert "/" not in spoken and "discard" not in spoken.lower()
    assert kit.adapter.prepare_tts_text("Four.").strip() == "Four."


async def test_no_second_spoken_question_when_hermes_already_said_it(kit):
    first = await ask(kit)
    kit.adapter.router.active_turn("household").audio = True  # Hermes's own voice played the notice
    await kit.adapter.send("household", NOTICE, reply_to=first.message_id)
    await kit.settle()
    assert prompts(kit, "dev-kitchen")
    assert not any(m["stream_id"].startswith("ask-") for m in kit.hub.controls("dev-kitchen", "play.begin"))
    # the real answer, once it comes, is still spoken
    assert kit.adapter.router.active_turn("household").audio is False


async def test_the_answer_window_opens_after_the_notice_was_heard(kit):
    await paused(kit)
    await quiet(kit)
    assert (10.0,) in kit.listeners["dev-kitchen"].called("open_followup")


async def test_the_paused_turn_never_times_out(kit):
    await paused(kit)
    assert kit.adapter._turn_held("household")


# --- continue ----------------------------------------------------------------------------------------


async def test_saying_continue_sends_discard_from_that_device(kit):
    first = await paused(kit)
    out = await say(kit, "Continue.")
    assert out["action"] == "block"
    sent = discards(kit)
    assert len(sent) == 1
    assert sent[0].source.user_id == "dev-kitchen" and sent[0].source.chat_id == "household"
    assert kit.hub.controls("dev-kitchen", "prompt.clear")
    assert kit.adapter.router.active_turn("household").message_id == first.message_id

    # Hermes drops the lost turn and answers the saved question: spoken in the kitchen
    await kit.adapter.send("household", DISCARDED, reply_to=sent[0].message_id)
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "caption")[-1]["role"] == "status"
    assert kit.adapter.router.active_turn("household").message_id == first.message_id
    await kit.adapter.send("household", "Four.", reply_to=first.message_id)
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "caption")[-1] == {"type": "caption", "role": "assistant", "text": "Four."}
    assert kit.tts.texts[-1] == "Four."
    assert not kit.adapter.router.household_busy("household")


async def test_tapping_continue_sends_discard(kit):
    await paused(kit)
    card = prompts(kit, "dev-kitchen")[-1]
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "prompt", "id": card["id"],
                                                  "value": "Continue"})
    await kit.settle()
    assert len(discards(kit)) == 1 and discards(kit)[0].source.user_id == "dev-kitchen"


async def test_a_tap_from_another_room_does_not_resume(kit):
    await paused(kit)
    card = prompts(kit, "dev-kitchen")[-1]
    await kit.adapter._on_control("dev-office", {"type": "ui.action", "kind": "prompt", "id": card["id"],
                                                 "value": "Continue"})
    await kit.settle()
    assert discards(kit) == []


async def test_continue_with_the_wake_word_works_too(kit):
    await paused(kit)
    await say(kit, "continue", kind="wake")
    assert len(discards(kit)) == 1


@pytest.mark.parametrize("said", ["Continue.", "carry on", "OK, continue", "go on please", "Resume.", "keep going",
                                  "Yes.", "okay"])
def test_continue_words(said):
    assert is_continue(said)


@pytest.mark.parametrize("said", ["", "continue?", "don't continue", "continue the story about the moon",
                                  "what's two plus two", "stop", "no", "yes but what happened"])
def test_not_continue(said):
    assert not is_continue(said)


# --- anything else said while paused -------------------------------------------------------------------


async def test_a_question_asked_while_paused_is_kept_for_after_continue(kit):
    first = await paused(kit)
    out = await say(kit, "What's the capital of France?", device="dev-office", kind="wake")
    assert out["action"] == "block"
    assert discards(kit) == []
    assert kit.adapter.router.queued("household") == 1
    # the card follows the person who spoke, with a reminder
    assert prompts(kit, "dev-office")
    assert kit.hub.controls("dev-kitchen", "prompt.clear")
    assert "continue" in kit.tts.texts[-1].lower()

    await say(kit, "Continue.", device="dev-office")
    assert len(discards(kit)) == 1 and discards(kit)[0].source.user_id == "dev-office"
    before = len(kit.events)
    await kit.adapter.send("household", "Four.", reply_to=first.message_id)
    await kit.settle()
    nxt = kit.events[before]
    assert nxt.message_type == MessageType.VOICE and nxt.source.user_id == "dev-office"
    assert nxt.source.chat_id == "household"


async def test_an_empty_clip_is_not_kept_as_a_question(kit):
    await paused(kit)
    await say(kit, "")
    assert discards(kit) == [] and kit.adapter.router.queued("household") == 0


async def test_our_own_audio_never_resumes(kit):
    await paused(kit)
    before = len(kit.events)
    utt = utterance(0.6)  # speech that began while the question was still playing
    kit.adapter._on_utterance("dev-kitchen", utt, "followup")
    await kit.settle()
    assert [e for e in kit.events[before:] if e.source.chat_id == SIDE_CHAT] == []
    assert discards(kit) == []


async def test_a_ringing_timer_can_still_be_stopped_while_paused(tmp_path):
    from adapter_kit import FakeTimers

    timers = FakeTimers()
    k = Kit(tmp_path, timers=timers).wire()
    k.add_device("dev-kitchen", "Kitchen")
    await paused(k)
    timers.ringing.add("kitchen")
    k.adapter._arm_stop("kitchen", "dev-kitchen")
    await say(k, "Stop.")
    assert timers.stopped == ["kitchen"]
    assert discards(k) == []


async def test_a_device_that_may_not_talk_cannot_lift_the_pause(kit, monkeypatch):
    monkeypatch.setenv("HOUSEMATE_ALLOWED_DEVICES", "dev-kitchen")
    await paused(kit)
    await quiet(kit)
    before = len(kit.events)
    kit.adapter._on_utterance("dev-office", fresh_utterance(), "wake")
    await kit.settle()
    assert kit.events[before:] == []  # not even sent for transcription
    assert discards(kit) == [] and kit.adapter._hold_for("household") is not None


async def test_the_pause_comes_back_if_hermes_cannot_take_the_command(kit):
    first = await paused(kit)

    async def broken(event):
        kit.events.append(event)
        raise RuntimeError("gateway gone")

    kit.adapter.handle_message = broken
    card = prompts(kit, "dev-kitchen")[-1]
    await kit.adapter._on_control("dev-kitchen", {"type": "ui.action", "kind": "prompt", "id": card["id"],
                                                  "value": "Continue"})
    await kit.settle()
    hold = kit.adapter._hold_for("household")
    assert hold is not None and hold.turn_id == first.message_id
    assert len(prompts(kit, "dev-kitchen")) == 2  # shown again


# --- resolved somewhere else -----------------------------------------------------------------------------


async def test_an_answer_to_the_saved_question_clears_the_pause(kit):
    """Someone pressed Discard on Desktop or ran ``hermes sessions discard``: Hermes just answers."""
    first = await paused(kit)
    await kit.adapter.send("household", "Four.", reply_to=first.message_id)
    await kit.settle()
    card = prompts(kit, "dev-kitchen")[-1]
    assert card["id"] in [m["id"] for m in kit.hub.controls("dev-kitchen", "prompt.clear")]
    assert not kit.adapter._turn_held("household")
    assert not kit.adapter.router.household_busy("household")
    await ask(kit, "dev-office")
    assert kit.events[-1].message_type == MessageType.VOICE and kit.events[-1].source.user_id == "dev-office"


# --- other pauses ------------------------------------------------------------------------------------------


async def test_the_answer_side_chat_is_unpaused_on_its_own(kit):
    """The side chat only carries spoken answers that never reach the model: nothing to lose."""
    await kit.adapter.send(SIDE_CHAT, NOTICE)
    await kit.settle()
    assert len(discards(kit, SIDE_CHAT)) == 1


async def test_a_pause_nobody_can_lift_by_voice_is_said_plainly(kit):
    first = await ask(kit)
    await kit.adapter.send("household", OTHER_PAUSE, reply_to=first.message_id)
    await kit.settle()
    caption = kit.hub.controls("dev-kitchen", "caption")[-1]["text"]
    assert "/reset" not in caption and "paused" in caption.lower()
    assert prompts(kit, "dev-kitchen") == [] and discards(kit) == []
    assert "/" not in kit.adapter.prepare_tts_text(OTHER_PAUSE)
