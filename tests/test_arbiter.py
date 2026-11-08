import pytest

from housemate.core.arbiter import WakeArbiter, WakeWinner


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


def test_arbiter_picks_single_winner_and_mutes_group(clock):
    arb = WakeArbiter(window_ms=400, clock=clock)
    arb.offer("phone", "kitchen", "downstairs", 0.8, 0.2)
    clock.advance(0.15)
    arb.offer("laptop", "office", "downstairs", 0.7, 0.9)
    clock.advance(0.1)
    assert arb.poll() is None  # window still open
    assert arb.outcome("phone") is None and arb.outcome("laptop") is None

    clock.advance(0.2)  # 450 ms after the first offer
    win = arb.poll()
    assert isinstance(win, WakeWinner)
    assert (win.device_id, win.room, win.group) == ("phone", "kitchen", "downstairs")
    assert arb.losers() == ["laptop"]
    assert arb.poll() is None  # exactly one winner for the round
    assert arb.outcome("phone") == "won"
    assert arb.outcome("laptop") == "lost"

    # a lagging device that heard the same wake (offer stamped inside the decided window)
    # loses straight away and does not open a round of its own
    arb.offer("tablet", "office", "downstairs", 0.99, 0.99, at=clock.t - 0.3)
    assert arb.outcome("tablet") == "lost"
    clock.advance(1.0)
    assert arb.poll() is None


def test_close_scores_go_to_the_louder_device(clock):
    arb = WakeArbiter(window_ms=400, clock=clock)
    arb.offer("far", "kitchen", None, 0.999, 0.05)
    arb.offer("near", "kitchen", None, 0.991, 0.40)  # same room, no group: one earshot
    clock.advance(0.5)
    win = arb.poll()
    assert win.device_id == "near"
    assert arb.losers() == ["far"]


def test_arbiter_separate_groups_both_win(clock):
    arb = WakeArbiter(window_ms=400, clock=clock)
    arb.offer("phone", "kitchen", None, 0.8, 0.2)
    arb.offer("laptop", "office", None, 0.7, 0.9)
    clock.advance(0.5)
    winners = {arb.poll().device_id, arb.poll().device_id}
    assert winners == {"phone", "laptop"}
    assert arb.losers() == []
    assert arb.poll() is None
    assert arb.outcome("phone") == "won" and arb.outcome("laptop") == "won"


def test_offer_after_window_is_new_round(clock):
    arb = WakeArbiter(window_ms=400, clock=clock)
    arb.offer("phone", "kitchen", "downstairs", 0.9, 0.5)
    clock.advance(0.5)
    arb.offer("laptop", "office", "downstairs", 0.6, 0.1)
    first = arb.poll()
    assert first.device_id == "phone" and arb.losers() == []
    assert arb.poll() is None  # the second round is still open
    assert arb.outcome("laptop") is None
    clock.advance(0.45)
    second = arb.poll()
    assert second.device_id == "laptop"
    assert arb.outcome("laptop") == "won"


def test_group_playing_covers_the_whole_earshot_group(clock):
    arb = WakeArbiter(clock=clock)
    assert not arb.group_playing("kitchen", "downstairs")
    arb.set_playing("phone", "kitchen", "downstairs", True)
    assert arb.group_playing("kitchen", "downstairs")
    assert arb.group_playing("office", "downstairs")
    assert not arb.group_playing("bedroom", None)
    arb.set_playing("phone", "kitchen", "downstairs", False)
    assert not arb.group_playing("office", "downstairs")
    arb.set_playing("tab", "bedroom", None, True)
    assert arb.group_playing("bedroom", None)
    arb.forget("tab")
    assert not arb.group_playing("bedroom", None)
