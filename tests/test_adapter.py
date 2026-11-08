import asyncio
import wave

import pytest

from adapter_kit import RATE, Kit, utterance

from gateway.platforms.base import AudioFormat  # noqa: E402  (shimmed when Hermes is absent)
from gateway.platforms.event import MessageType  # noqa: E402


@pytest.fixture
async def kit(tmp_path):
    k = Kit(tmp_path).wire()
    k.add_device("dev-kitchen", "Kitchen", "Kitchen phone")
    k.add_device("dev-office", "Office", "Office laptop")
    return k


async def speak(kit, device_id="dev-kitchen", kind="wake"):
    kit.adapter._on_utterance(device_id, utterance(), kind)
    await kit.settle()
    return kit.events[-1] if kit.events else None


# --- inbound ---------------------------------------------------------------------------------------


async def test_wake_utterance_emits_voice_event_with_room_metadata(kit):
    event = await speak(kit)
    assert event.message_type == MessageType.VOICE
    assert event.media_types == ["audio/wav"]
    with wave.open(event.media_urls[0], "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2)
    assert event.source.chat_id == "household"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "dev-kitchen"
    assert event.source.user_name == "Kitchen"
    assert event.channel_context == "[room: Kitchen]"
    assert "Kitchen" in event.channel_prompt
    # Hermes 0.23 refuses events with custom metadata: the room must never travel there.
    assert not event.metadata
    assert kit.adapter.router.active_turn("household").message_id == event.message_id


async def test_per_room_mode_uses_room_chat(tmp_path):
    k = Kit(tmp_path, extra={"conversation": "per_room"}).wire()
    k.add_device("dev-kitchen", "Kitchen")
    event = await speak(k)
    assert event.source.chat_id == "kitchen"


async def test_utterance_without_room_is_ignored(kit):
    kit.store.upsert_device("dev-new", "New phone")
    kit.hub.online.add("dev-new")
    kit.adapter._on_utterance("dev-new", utterance(), "ptt")
    await kit.settle()
    assert kit.events == []


async def test_household_queue_holds_second_room_until_turn_end(kit):
    """Review focus 1 at the adapter: the office question waits for the kitchen turn to end."""
    first = await speak(kit, "dev-kitchen")
    kit.adapter._on_utterance("dev-office", utterance(), "wake")
    await kit.settle()
    assert len(kit.events) == 1  # nothing handed to Hermes mid-turn
    assert kit.adapter.router.queued("household") == 1
    assert kit.hub.controls("dev-office", "state")[-1]["state"] == "thinking"

    await kit.adapter.send("household", "Four.", reply_to=first.message_id)
    await kit.settle()
    assert len(kit.events) == 2
    second = kit.events[1]
    assert second.source.user_id == "dev-office" and second.channel_context == "[room: Office]"
    assert kit.adapter.router.active_turn("household").message_id == second.message_id


async def test_queued_utterance_waits_for_hermes_session_to_go_idle(kit):
    first = await speak(kit, "dev-kitchen")
    kit.adapter._on_utterance("dev-office", utterance(), "wake")
    await kit.settle()
    key = kit.adapter._event_session_key(first)
    kit.adapter._active_sessions[key] = asyncio.Event()  # Hermes still finishing the kitchen turn
    await kit.adapter.send("household", "Four.", reply_to=first.message_id)
    await asyncio.sleep(0.2)
    assert len(kit.events) == 1
    del kit.adapter._active_sessions[key]
    await kit.settle()
    assert len(kit.events) == 2


async def test_processing_complete_ends_turn_when_no_reply_text(kit):
    first = await speak(kit, "dev-kitchen")
    kit.adapter._on_utterance("dev-office", utterance(), "wake")
    await kit.adapter.on_processing_complete(first, "success")
    await kit.settle()
    assert len(kit.events) == 2


# --- outbound text -----------------------------------------------------------------------------


async def test_send_caption_goes_to_origin_room_only(kit):
    event = await speak(kit, "dev-kitchen")
    await kit.adapter.send("household", "Four.", reply_to=event.message_id)
    await kit.settle()
    captions = kit.hub.controls("dev-kitchen", "caption")
    assert captions and captions[-1] == {"type": "caption", "role": "assistant", "text": "Four."}
    assert kit.hub.controls("dev-office", "caption") == []


async def test_stt_echo_becomes_the_you_said_caption(kit):
    await speak(kit, "dev-kitchen")
    await kit.adapter.send("household", '\U0001F399️ "What is 2 plus 2?"')
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "caption")[-1] == {"type": "caption", "role": "user", "text": "What is 2 plus 2?"}
    assert kit.adapter.router.household_busy("household")  # the echo is not the end of the turn


async def test_gateway_notices_and_side_chat_are_swallowed(kit):
    await kit.adapter.send("household", "\U0001F4EC No home channel is set for Housemate. Type /sethome ...")
    await kit.adapter.send("housemate-answers", "(housemate: approve)")
    await kit.settle()
    assert kit.hub.calls == []


async def test_cron_delivery_to_a_room_is_spoken_there(kit):
    result = await kit.adapter.send("kitchen", "Good morning. It is sunny.")
    await kit.settle()
    assert result.success
    assert kit.hub.controls("dev-kitchen", "caption")[-1]["role"] == "announcement"
    assert any(m["stream_id"].startswith("say-") for m in kit.hub.controls("dev-kitchen", "play.begin"))
    assert kit.tts.texts == ["Good morning. It is sunny."]
    assert kit.hub.controls("dev-office", "caption") == []


async def test_reply_without_audio_is_spoken_by_the_hub(kit):
    event = await speak(kit, "dev-kitchen")
    await kit.adapter.send("household", "Paris.", reply_to=event.message_id)
    await kit.settle()
    assert kit.tts.texts == ["Paris."]
    assert any(m["stream_id"].startswith("rep-") for m in kit.hub.controls("dev-kitchen", "play.begin"))


async def test_send_typing_sets_thinking_state(kit):
    await speak(kit, "dev-kitchen")
    kit.registry.set_device_state("dev-kitchen", "idle")
    await kit.adapter.send_typing("household")
    await kit.settle()
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "thinking"


async def test_spoken_replies_on_by_default_except_side_chat(kit):
    assert kit.adapter._should_auto_tts_for_chat("household") is True
    assert kit.adapter._should_auto_tts_for_chat("housemate-answers") is False


async def test_paired_devices_are_the_allowlist(kit):
    kit.store.upsert_device("dev-stranger", "Not paired")
    kit.adapter._refresh_allowlist()
    assert kit.adapter.config.extra["allow_from"] == ["dev-kitchen", "dev-office"]
    kit.store.revoke_device("dev-office")
    kit.drop("dev-office")  # the hub reports the revoked device's socket closing
    assert kit.adapter.config.extra["allow_from"] == ["dev-kitchen"]


async def test_error_right_after_a_turn_goes_to_the_room_that_asked(kit):
    event = await speak(kit, "dev-office")
    await kit.adapter.on_processing_complete(event, "failure")  # Hermes gave up on the turn
    await kit.adapter.send("household", "Sorry, I encountered an error.")
    await kit.settle()
    assert kit.hub.controls("dev-office", "caption")[-1] == {"type": "caption", "role": "assistant", "text": "Sorry, I encountered an error."}
    assert kit.hub.controls("dev-kitchen", "caption") == []


# --- streaming TTS ---------------------------------------------------------------------------------


async def reply_stream(kit, chunks):
    event = await speak(kit, "dev-kitchen")
    fmt = AudioFormat(sample_rate=RATE, channels=1, sample_width=2)
    assert kit.adapter.supports_streaming_tts("household", fmt)
    handle = await kit.adapter.begin_streaming_tts("household", fmt)
    assert handle is not None
    for chunk in chunks:
        await kit.adapter.write_streaming_tts(handle, chunk)
    return event, handle


async def test_streaming_tts_frames_reach_origin_device(kit):
    chunks = [b"\x01\x02" * 2400, b"\x03\x04" * 2400, b"\x05\x06" * 2400]
    event, handle = await reply_stream(kit, chunks)
    await kit.adapter.finish_streaming_tts(handle)
    await kit.settle()
    seq = kit.hub.to("dev-kitchen")
    kinds = [(k, p["type"] if k == "control" else "pcm") for k, p in seq if k == "pcm" or p["type"].startswith("play.")]
    assert kinds[0] == ("control", "play.begin") and kinds[-1] == ("control", "play.end")
    assert all(k == ("pcm", "pcm") for k in kinds[1:-1]) and len(kinds) >= 3
    begin = kit.hub.controls("dev-kitchen", "play.begin")[-1]
    assert begin["sample_rate"] == RATE and begin["stream_id"].startswith("rep-")
    assert kit.hub.pcm("dev-kitchen") == b"".join(chunks)
    assert kit.hub.to("dev-office") == []
    assert handle.audible


async def test_odd_sized_chunks_are_realigned(kit):
    data = bytes(range(256)) * 30
    sizes, chunks, pos = [7, 1, 1222, 3, 4901, 52], [], 0
    for size in sizes:
        chunks.append(data[pos : pos + size])
        pos += size
    _, handle = await reply_stream(kit, chunks)
    await kit.adapter.finish_streaming_tts(handle)
    await kit.settle()
    sent = kit.hub.pcm("dev-kitchen")
    assert len(sent) % 2 == 0 and sent == data[: len(sent)] and len(sent) == pos - (pos % 2)


async def test_followup_opened_after_playback_done(kit):
    _, handle = await reply_stream(kit, [b"\x00\x01" * 1200])
    await kit.adapter.finish_streaming_tts(handle)
    await kit.settle()
    listener = kit.listeners["dev-kitchen"]
    assert listener.called("set_playing") == [(True,)]
    assert listener.called("open_followup") == []  # not before the device has played it
    stream_id = kit.hub.controls("dev-kitchen", "play.begin")[-1]["stream_id"]
    await kit.adapter._on_control("dev-kitchen", {"type": "playback.done", "stream_id": stream_id})
    await kit.settle()
    assert listener.called("set_playing")[-1] == (False,)
    assert listener.called("open_followup") == [(6.0,)]
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "listening"


async def test_no_followup_after_an_announcement(kit):
    await kit.adapter.send("kitchen", "Dinner is ready.")
    await kit.settle()
    sid = kit.hub.controls("dev-kitchen", "play.begin")[-1]["stream_id"]
    await kit.adapter._on_control("dev-kitchen", {"type": "playback.done", "stream_id": sid})
    assert kit.listeners["dev-kitchen"].called("open_followup") == []


async def test_stream_to_disconnected_device_aborts_cleanly(kit):
    """Review focus 2: the phone drops mid-reply; nothing raises, the caption waits for it."""
    event, handle = await reply_stream(kit, [b"\x00\x01" * 2400])
    kit.drop("dev-kitchen")
    for _ in range(3):
        await kit.adapter.write_streaming_tts(handle, b"\x02\x03" * 2400)
    await kit.adapter.finish_streaming_tts(handle)
    await kit.adapter.abort_streaming_tts(handle, "late abort is fine too")
    await kit.adapter.send("household", "The answer is four.", reply_to=event.message_id)
    await kit.settle()
    assert not kit.adapter.router.household_busy("household")
    before = len(kit.hub.controls("dev-kitchen", "caption"))

    kit.connect("dev-kitchen", "kitchen", "Kitchen phone")
    await kit.adapter._on_control("dev-kitchen", {"type": "hello", "device_id": "dev-kitchen", "name": "Kitchen phone", "ua": "t"})
    await kit.settle()
    captions = kit.hub.controls("dev-kitchen", "caption")[before:]
    assert {"type": "caption", "role": "assistant", "text": "The answer is four."} in captions


async def test_stream_declined_when_room_offline(kit):
    event = await speak(kit, "dev-kitchen")
    kit.drop("dev-kitchen")
    fmt = AudioFormat(sample_rate=RATE, channels=1, sample_width=2)
    assert kit.adapter.supports_streaming_tts("household", fmt) is False
    assert await kit.adapter.begin_streaming_tts("household", fmt) is None


async def test_barge_in_on_wake_aborts_playout(kit):
    _, handle = await reply_stream(kit, [b"\x00\x01" * RATE * 5])  # 5 s of speech
    await asyncio.sleep(0.05)
    kit.adapter._on_wake("dev-kitchen")
    await kit.settle()
    aborts = kit.hub.controls("dev-kitchen", "play.abort")
    assert aborts and aborts[-1]["stream_id"].startswith("rep-")
    assert len(kit.hub.pcm("dev-kitchen")) < 2 * RATE * 5
    assert kit.hub.controls("dev-kitchen", "state")[-1]["state"] == "listening"
    await kit.adapter.write_streaming_tts(handle, b"\x00\x00" * 100)  # late chunks are dropped quietly


async def test_send_voice_plays_whole_file(kit, tmp_path):
    event = await speak(kit, "dev-kitchen")
    path = tmp_path / "reply.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
        w.writeframes(b"\x07\x00" * 4800)
    result = await kit.adapter.send_voice("household", str(path), reply_to=event.message_id)
    await kit.settle()
    assert result.success
    assert kit.hub.pcm("dev-kitchen") == b"\x07\x00" * 4800
    assert kit.adapter.router.turn_for(event.message_id).audio


async def test_prewarm_sends_a_vetoed_clip_to_the_side_chat(kit):
    from housemate.core.answers import apply_answer

    await kit.adapter._prewarm()
    (event,) = kit.events
    assert event.source.chat_id == "housemate-answers" and event.source.user_id == "dev-kitchen"
    out = apply_answer(kit.adapter.book, {"platform": "housemate", "user_message": f'"you"\n\n{event.channel_context}'})
    assert out["action"] == "block"
    assert kit.adapter.book.for_device("dev-kitchen") is None  # never mistaken for a device's answer


async def test_prewarm_skipped_without_paired_devices(tmp_path):
    k = Kit(tmp_path).wire()
    await k.adapter._prewarm()
    assert k.events == []
