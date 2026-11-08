from types import SimpleNamespace

import pytest

from housemate.core.config import HousemateConfig
from housemate.core.router import HOUSEHOLD_CHAT, SessionRouter


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def utterance(room: str, message_id: str) -> SimpleNamespace:
    return SimpleNamespace(room=room, message_id=message_id)


# --- config ------------------------------------------------------------------------------------


def test_config_defaults():
    cfg = HousemateConfig.from_extra({})
    assert cfg.conversation == "household"
    assert (cfg.port, cfg.control_port) == (8742, 8743)
    assert cfg.wake_threshold == 0.5 and cfg.followup_s == 6.0
    assert cfg.warnings == ()


def test_config_reads_extra_and_warns_on_bad_values():
    cfg = HousemateConfig.from_extra(
        {"conversation": "per-room", "port": "9000", "wake_threshold": 2, "rooms": "Kitchen, Living Room", "tls": "no"}
    )
    assert cfg.conversation == "per_room"
    assert cfg.port == 9000
    assert cfg.wake_threshold == 0.5  # out of range -> default
    assert cfg.rooms == ("Kitchen", "Living Room")
    assert cfg.tls is False
    assert any("wake_threshold" in w for w in cfg.warnings)


# --- routing -------------------------------------------------------------------------------------


def test_household_mode_shares_chat_id():
    router = SessionRouter("household")
    assert router.chat_id_for("Kitchen") == router.chat_id_for("office") == HOUSEHOLD_CHAT


def test_per_room_mode_uses_room_slug():
    router = SessionRouter("per_room")
    assert router.chat_id_for("Living Room") == "living-room"
    assert router.chat_id_for("kitchen") == "kitchen"


def test_reply_room_follows_origin_turn():
    router = SessionRouter("household")
    chat = router.chat_id_for("kitchen")
    router.begin_turn(chat, "kitchen", "m-kitchen", device_id="dev-k")
    router.on_turn_end(chat, "m-kitchen")
    router.begin_turn(chat, "office", "m-office", device_id="dev-o")
    assert router.reply_room(chat, "m-kitchen") == "kitchen"
    assert router.reply_device(chat, "m-kitchen") == "dev-k"
    assert router.reply_room(chat, None) == "office"
    assert router.reply_device(chat, None) == "dev-o"


def test_household_queue_holds_second_room_until_turn_end():
    """Review focus 1: a second room speaking mid-turn is queued, never dispatched over it."""
    router = SessionRouter("household")
    chat = router.chat_id_for("kitchen")
    router.begin_turn(chat, "kitchen", "m1", device_id="dev-k")
    assert router.household_busy(chat)

    office = utterance("office", "m2")
    assert router.chat_id_for("office") == chat
    router.enqueue(office, chat)
    assert router.household_busy(chat)  # still the kitchen turn
    assert router.active_turn(chat).message_id == "m1"
    assert router.queued(chat) == 1

    # A turn-end signal for some other message does not release the queue.
    assert router.on_turn_end(chat, "not-this-turn") is None
    assert router.household_busy(chat)

    nxt = router.on_turn_end(chat, "m1")
    assert nxt is office
    assert router.queued(chat) == 0
    # The released utterance holds the slot until its own turn begins: nothing can slip in between.
    assert router.household_busy(chat)
    router.begin_turn(chat, "office", "m2", device_id="dev-o")
    assert router.active_turn(chat).room == "office"
    assert router.on_turn_end(chat, "m2") is None
    assert not router.household_busy(chat)


def test_per_room_turns_do_not_block_each_other():
    router = SessionRouter("per_room")
    router.begin_turn("kitchen", "kitchen", "m1")
    assert router.household_busy("kitchen")
    assert not router.household_busy("office")


def test_queue_is_fifo_and_bounded():
    router = SessionRouter("household", max_queue=2)
    router.begin_turn(HOUSEHOLD_CHAT, "kitchen", "m1")
    a, b, c = utterance("office", "a"), utterance("hall", "b"), utterance("office", "c")
    for item in (a, b, c):
        router.enqueue(item, HOUSEHOLD_CHAT)
    assert router.queued(HOUSEHOLD_CHAT) == 2  # the oldest was dropped
    assert router.on_turn_end(HOUSEHOLD_CHAT, "m1") is b
    router.begin_turn(HOUSEHOLD_CHAT, "hall", "b")
    assert router.on_turn_end(HOUSEHOLD_CHAT, "b") is c


def test_stale_queued_utterances_are_dropped():
    clock = Clock()
    router = SessionRouter("household", clock=clock, queue_ttl_s=60)
    router.begin_turn(HOUSEHOLD_CHAT, "kitchen", "m1")
    router.enqueue(utterance("office", "old"), HOUSEHOLD_CHAT)
    clock.t += 61
    assert router.on_turn_end(HOUSEHOLD_CHAT, "m1") is None
    assert not router.household_busy(HOUSEHOLD_CHAT)


def test_turn_safety_timeout_releases_queue():
    clock = Clock()
    router = SessionRouter("household", clock=clock, turn_timeout_s=180)
    router.begin_turn(HOUSEHOLD_CHAT, "kitchen", "m1")
    router.enqueue(utterance("office", "m2"), HOUSEHOLD_CHAT)
    clock.t += 179
    assert router.expire() == []
    clock.t += 2
    released = router.expire()
    assert [(chat, item.message_id) for chat, item in released] == [(HOUSEHOLD_CHAT, "m2")]
    assert router.active_turn(HOUSEHOLD_CHAT) is None or router.active_turn(HOUSEHOLD_CHAT).message_id != "m1"


def test_turn_lookup_by_message_id():
    router = SessionRouter("household")
    router.begin_turn(HOUSEHOLD_CHAT, "kitchen", "m1", device_id="dev-k")
    turn = router.turn_for("m1")
    assert turn is not None and turn.room == "kitchen" and turn.device_id == "dev-k"
    assert router.turn_for("nope") is None


def test_reply_room_unknown_chat_is_none():
    router = SessionRouter("household")
    assert router.reply_room("household", None) is None


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        SessionRouter("everyone")


def test_dropped_and_expired_are_reported():
    clock = Clock()
    router = SessionRouter("household", clock=clock, max_queue=1, turn_timeout_s=60)
    router.begin_turn(HOUSEHOLD_CHAT, "kitchen", "m1", device_id="dev-k")
    a, b = utterance("office", "a"), utterance("hall", "b")
    router.enqueue(a, HOUSEHOLD_CHAT)
    router.enqueue(b, HOUSEHOLD_CHAT)  # overflow drops a
    clock.t += 61
    released = router.expire()
    assert [item for _, item in released] == [b]
    dropped, expired = router.take_dropped()
    assert dropped == [a]
    assert [t.message_id for t in expired] == ["m1"]
    assert router.take_dropped() == ([], [])
