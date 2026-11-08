import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from housemate.core.store import HousemateStore
from housemate.core.timers import RingController, Timer, TimerService, next_occurrence, ring_text

T0 = 1_700_000_000.0
CET = timezone(timedelta(hours=1))


class FakeClock:
    def __init__(self, t: float = T0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Recorder:
    """Collects every callback the timer service fires."""

    def __init__(self) -> None:
        self.rings: list[Timer] = []
        self.chimes: list[str] = []
        self.spoken: list[tuple[str, str]] = []
        self.pushed: list[tuple[str, list[dict]]] = []

    async def ring(self, timer: Timer) -> None:
        self.rings.append(timer)

    async def chime(self, room: str) -> None:
        self.chimes.append(room)

    async def speak(self, room: str, text: str) -> None:
        self.spoken.append((room, text))

    async def push(self, room: str, items: list[dict]) -> None:
        self.pushed.append((room, items))


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    yield c
    c.close()


@pytest.fixture
def store(conn):
    return HousemateStore(conn)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rec():
    return Recorder()


def make_service(store, clock, rec, **kw) -> TimerService:
    return TimerService(
        store,
        clock,
        rec.ring,
        play_chime=rec.chime,
        speak=rec.speak,
        broadcast_timers=rec.push,
        **kw,
    )


async def step(svc: TimerService, clock: FakeClock, seconds: float = 0.0) -> None:
    clock.advance(seconds)
    await svc.tick()
    await svc.drain()


def state_of(store, timer_id: str) -> str:
    return store.get_timer(timer_id).state


# --- the plan's five ---------------------------------------------------------------


async def test_timer_rings_once_at_due(store, clock, rec):
    svc = make_service(store, clock, rec)
    t = svc.add("Kitchen", seconds=300, label="pasta")
    assert t.room == "kitchen" and t.state == "pending" and t.remaining_s == 300

    await step(svc, clock, 299)
    assert rec.rings == []

    await step(svc, clock, 1)
    assert [r.id for r in rec.rings] == [t.id]
    rung = rec.rings[0]
    assert rung.state == "ringing" and rung.missed is False and rung.fired_at == T0 + 300
    assert rec.spoken == [("kitchen", "Your pasta timer is done.")]
    assert rec.chimes == ["kitchen"]
    assert state_of(store, t.id) == "ringing"

    for _ in range(5):
        await step(svc, clock, 1)
    assert len(rec.rings) == 1


async def test_list_shows_remaining(store, clock, rec):
    svc = make_service(store, clock, rec)
    kitchen = svc.add("kitchen", seconds=300, label="pasta")
    office = svc.add("office", seconds=60, label="call")

    clock.advance(10)
    items = svc.list()
    assert [(t.id, t.remaining_s) for t in items] == [(office.id, 50), (kitchen.id, 290)]
    assert all(isinstance(t.remaining_s, int) for t in items)
    assert [t.id for t in svc.list("Kitchen")] == [kitchen.id]

    clock.advance(0.4)  # partial seconds round up: never says 0 before it rings
    assert svc.list("office")[0].remaining_s == 50


async def test_cancel_by_label(store, clock, rec):
    svc = make_service(store, clock, rec)
    pasta = svc.add("kitchen", seconds=60, label="Pasta")
    eggs = svc.add("kitchen", seconds=60, label="eggs")
    office_pasta = svc.add("office", seconds=60, label="pasta")

    cancelled = svc.cancel(room="Kitchen", label="the PASTA timer")
    assert [t.id for t in cancelled] == [pasta.id]
    assert cancelled[0].state == "cancelled"
    assert state_of(store, pasta.id) == "cancelled"
    assert [t.id for t in svc.list("kitchen")] == [eggs.id]

    await step(svc, clock, 60)
    assert sorted(r.id for r in rec.rings) == sorted([eggs.id, office_pasta.id])


async def test_timer_overdue_after_restart_rings_once(conn, clock):
    first = Recorder()
    svc1 = make_service(HousemateStore(conn), clock, first)
    t = svc1.add("kitchen", seconds=60, label="pasta")
    await svc1.drain()
    del svc1  # the hub goes down before the timer is due

    clock.advance(600)  # back up ten minutes later
    second = Recorder()
    svc2 = make_service(HousemateStore(conn), clock, second)
    restored = svc2.restore(clock())
    assert [(r.id, r.missed) for r in restored] == [(t.id, True)]
    await svc2.drain()
    assert [(r.id, r.missed) for r in second.rings] == [(t.id, True)]
    assert second.spoken == [("kitchen", "Your pasta timer finished while I was offline.")]

    # Ticking and restoring again in the same process never re-rings it.
    for _ in range(3):
        await step(svc2, clock, 1)
    assert svc2.restore(clock()) == []
    await svc2.drain()
    assert len(second.rings) == 1

    # The hub dies again mid-ring: the next process must not ring it a second time.
    third = Recorder()
    svc3 = make_service(HousemateStore(conn), clock, third)
    assert svc3.restore(clock()) == []
    await step(svc3, clock, 1)
    assert third.rings == [] and third.spoken == []
    assert HousemateStore(conn).get_timer(t.id).state == "done"
    assert first.rings == []


async def test_restore_retires_long_overdue_timers_without_ringing(conn, clock):
    """Review L8: a test timer from hours ago must not ring "finished while I was offline" at the
    next start (3 a.m., mid-demo). Recently missed ones still ring once."""
    svc1 = make_service(HousemateStore(conn), clock, Recorder())
    stale = svc1.add("kitchen", seconds=60, label="test")
    recent = svc1.add("kitchen", seconds=3 * 3600 - 1800, label="pasta")
    await svc1.drain()
    del svc1
    clock.advance(3 * 3600)
    rec = Recorder()
    svc2 = make_service(HousemateStore(conn), clock, rec)
    assert [r.id for r in svc2.restore(clock())] == [recent.id]
    await svc2.drain()
    assert [r.id for r in rec.rings] == [recent.id]
    assert all("test" not in text for _, text in rec.spoken)
    assert HousemateStore(conn).get_timer(stale.id).state == "done"
    assert [t.id for t in svc2.missed_while_offline] == [stale.id]


def test_restore_before_the_loop_runs_rings_on_first_tick(store, clock, rec):
    svc = make_service(store, clock, rec)
    t = svc.add("kitchen", seconds=5, label="tea")  # no event loop yet
    clock.advance(10)
    assert [x.id for x in svc.restore(clock())] == [t.id]
    assert rec.rings == [] and rec.spoken == []

    async def first_tick():
        await svc.tick()
        await svc.drain()

    asyncio.run(first_tick())
    assert [r.id for r in rec.rings] == [t.id]
    assert rec.spoken == [("kitchen", "Your tea timer finished while I was offline.")]
    assert rec.pushed[-1][1][0]["state"] == "ringing"


async def test_alarm_next_occurrence_today_or_tomorrow(store, clock, rec):
    now = datetime(2026, 11, 8, 7, 0, tzinfo=CET).timestamp()
    assert next_occurrence("07:30", now, CET) == datetime(2026, 11, 8, 7, 30, tzinfo=CET)
    assert next_occurrence("06:30", now, CET) == datetime(2026, 11, 9, 6, 30, tzinfo=CET)
    assert next_occurrence("7:00", now, CET) == datetime(2026, 11, 9, 7, 0, tzinfo=CET)
    assert next_occurrence("7:05 pm", now, CET) == datetime(2026, 11, 8, 19, 5, tzinfo=CET)
    assert next_occurrence("12:15am", now, CET) == datetime(2026, 11, 9, 0, 15, tzinfo=CET)
    for bad in ("25:00", "7", "07:60", "13:00 pm", ""):
        with pytest.raises(ValueError):
            next_occurrence(bad, now, CET)

    clock.t = now
    svc = make_service(store, clock, rec, tz=CET)
    alarm = svc.add_alarm("bedroom", "06:30", label="wake up")
    assert alarm.kind == "alarm" and alarm.duration_s is None
    assert alarm.due_at == datetime(2026, 11, 9, 6, 30, tzinfo=CET).timestamp()

    await step(svc, clock, alarm.due_at - clock())
    assert rec.spoken == [("bedroom", "It's 6:30. Your wake up alarm.")]


# --- ringing policy -------------------------------------------------------------------


async def test_ring_respeaks_every_20_seconds(store, clock, rec):
    svc = make_service(store, clock, rec)
    svc.add("kitchen", seconds=10, label="tea")
    await step(svc, clock, 10)
    assert len(rec.spoken) == 1 and len(rec.chimes) == 1

    await step(svc, clock, 19)
    assert len(rec.spoken) == 1

    await step(svc, clock, 1)
    assert rec.spoken == [("kitchen", "Your tea timer is done.")] * 2
    assert rec.chimes == ["kitchen", "kitchen"]


async def test_ring_auto_stops_after_three_minutes(store, clock, rec):
    svc = make_service(store, clock, rec)
    t = svc.add("kitchen", seconds=5, label="tea")
    await step(svc, clock, 5)
    for _ in range(179):
        await step(svc, clock, 1)
    assert svc.is_ringing("kitchen")
    assert len(rec.spoken) == 9  # at 0, 20, ..., 160 s

    await step(svc, clock, 1)  # 180 s after it started ringing
    assert not svc.is_ringing("kitchen")
    assert state_of(store, t.id) == "done"
    assert rec.pushed[-1] == ("kitchen", [])

    await step(svc, clock, 40)
    assert len(rec.spoken) == 9 and len(rec.rings) == 1


async def test_stop_ringing_silences_room_and_persists_stopped(store, clock, rec):
    svc = make_service(store, clock, rec)
    t = svc.add("kitchen", seconds=5, label="tea")
    other = svc.add("office", seconds=5, label="call")
    await step(svc, clock, 5)
    assert svc.is_ringing("kitchen") and svc.is_ringing("office")

    assert svc.stop_ringing("Kitchen") is True
    await svc.drain()
    assert state_of(store, t.id) == "stopped"
    assert not svc.is_ringing("kitchen") and svc.is_ringing("office")
    assert ("kitchen", []) in rec.pushed

    await step(svc, clock, 20)
    assert [room for room, _ in rec.spoken] == ["kitchen", "office", "office"]
    assert svc.stop_ringing("kitchen") is False
    assert state_of(store, other.id) == "ringing"


async def test_stop_ringing_cancels_audio_in_flight(store, clock):
    started = asyncio.Event()
    cancelled = []

    async def slow_speak(room, text):
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(room)
            raise

    svc = TimerService(store, clock, None, speak=slow_speak)
    svc.add("kitchen", seconds=1)
    clock.advance(1)
    await svc.tick()
    await asyncio.wait_for(started.wait(), 1)
    assert svc.stop_ringing("kitchen")
    await asyncio.wait_for(svc.drain(), 1)
    assert cancelled == ["kitchen"]


async def test_two_timers_in_one_room_share_the_ring(store, clock, rec):
    svc = make_service(store, clock, rec)
    svc.add("kitchen", seconds=30, label="pasta")
    svc.add("kitchen", seconds=30, label="sauce")
    await step(svc, clock, 30)
    assert rec.spoken == [("kitchen", "Your pasta timer is done. Your sauce timer is done.")]
    assert svc.stop_ringing("kitchen")
    assert svc.list("kitchen") == []


def test_ring_controller_alone(clock):
    calls = []

    async def speak(room, text):
        calls.append((clock(), room, text))

    ringer = RingController(clock, speak=speak, every_s=20, max_s=60)
    timer = Timer(id="t1", room="kitchen", kind="timer", label="tea", due_at=T0, created_at=T0 - 5,
                  duration_s=5.0, state="ringing", fired_at=T0)

    async def scenario():
        ringer.start(timer)
        await ringer.drain()
        expired = []
        for _ in range(60):
            clock.advance(1)
            expired += ringer.tick()
            await ringer.drain()
        return expired

    expired = asyncio.run(scenario())
    assert [c[0] - T0 for c in calls] == [0, 20, 40]
    assert [t.id for t in expired] == ["t1"]
    assert ringer.ringing() == []


# --- the rest of the surface -------------------------------------------------------------


async def test_cancel_by_id_and_all_in_room(store, clock, rec):
    svc = make_service(store, clock, rec)
    a = svc.add("kitchen", seconds=60, label="a")
    b = svc.add("kitchen", seconds=90, label="b")
    c = svc.add("office", seconds=60, label="c")

    assert [t.id for t in svc.cancel(timer_id=a.id)] == [a.id]
    assert svc.cancel(timer_id=a.id) == []  # already gone
    assert [t.id for t in svc.cancel(room="kitchen")] == [b.id]
    assert [t.id for t in svc.list()] == [c.id]
    with pytest.raises(ValueError):
        svc.cancel()


async def test_cancel_while_ringing_stops_the_ring(store, clock, rec):
    svc = make_service(store, clock, rec)
    t = svc.add("kitchen", seconds=5, label="tea")
    await step(svc, clock, 5)
    assert [x.id for x in svc.cancel(label="tea")] == [t.id]
    assert not svc.is_ringing("kitchen")
    assert state_of(store, t.id) == "cancelled"


async def test_changes_are_broadcast_to_the_room(store, clock, rec):
    svc = make_service(store, clock, rec)
    t = svc.add("kitchen", seconds=60, label="pasta")
    await svc.drain()
    room, items = rec.pushed[-1]
    assert room == "kitchen"
    assert [(i["id"], i["state"], i["remaining_s"], i["label"]) for i in items] == [(t.id, "pending", 60, "pasta")]

    await step(svc, clock, 60)
    assert rec.pushed[-1][1][0]["state"] == "ringing"

    svc.stop_ringing("kitchen")
    await svc.drain()
    assert rec.pushed[-1] == ("kitchen", [])


async def test_add_validates_input(store, clock, rec):
    svc = make_service(store, clock, rec, tz=CET)
    with pytest.raises(ValueError):
        svc.add("kitchen", seconds=0)
    with pytest.raises(ValueError):
        svc.add("kitchen", seconds=-5)
    with pytest.raises(ValueError):
        svc.add("kitchen", seconds=True)
    with pytest.raises(ValueError):
        svc.add("kitchen")
    with pytest.raises(ValueError):
        svc.add("kitchen", seconds=60, at=datetime(2030, 1, 1, tzinfo=CET))
    with pytest.raises(ValueError):
        svc.add("kitchen", at=datetime.fromtimestamp(T0 - 1, CET))
    with pytest.raises(ValueError):
        svc.add("kitchen", seconds=60, kind="reminder")
    with pytest.raises(ValueError):
        svc.add("!!!", seconds=60)
    alarm = svc.add("kitchen", at=datetime.fromtimestamp(T0 + 3600, CET), kind="alarm")
    assert alarm.due_at == T0 + 3600 and alarm.remaining_s == 3600


async def test_late_tick_marks_timer_missed(store, clock, rec):
    """The PC slept through the due time: it still rings once, flagged as missed."""
    svc = make_service(store, clock, rec)
    svc.add("kitchen", seconds=60, label="pasta")
    await step(svc, clock, 60 + 3600)
    assert [r.missed for r in rec.rings] == [True]


async def test_run_loop_ticks_with_injected_sleep(store, clock, rec):
    async def fake_sleep(seconds):
        clock.advance(seconds)
        await asyncio.sleep(0)

    svc = make_service(store, clock, rec, sleep=fake_sleep)
    svc.add("kitchen", seconds=5, label="tea")
    task = asyncio.create_task(svc.run())
    for _ in range(200):
        await asyncio.sleep(0)
        if rec.rings:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await svc.drain()
    assert len(rec.rings) == 1
    assert rec.rings[0].fired_at - T0 < 6


def test_ring_text_variants():
    def timer(**kw):
        base = dict(id="t", room="kitchen", kind="timer", label="", due_at=T0, created_at=T0 - 300,
                    duration_s=300.0, state="ringing", fired_at=T0)
        base.update(kw)
        return Timer(**base)

    assert ring_text(timer()) == "Your 5 minute timer is done."
    assert ring_text(timer(duration_s=5400.0)) == "Your 1 hour 30 minute timer is done."
    assert ring_text(timer(label="pasta timer")) == "Your pasta timer is done."
    due = datetime(2026, 11, 8, 7, 30, tzinfo=CET).timestamp()
    assert ring_text(timer(kind="alarm", due_at=due, duration_s=None), CET) == "It's 7:30. Your alarm."
    assert (
        ring_text(timer(kind="alarm", due_at=due, duration_s=None, missed=True), CET)
        == "Your 7:30 alarm went off while I was offline."
    )
