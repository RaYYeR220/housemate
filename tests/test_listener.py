import asyncio
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

from housemate.core.arbiter import WakeArbiter
from housemate.core.audio import FRAME_BYTES
from housemate.core.endpointer import Endpointer
from housemate.core.listener import DeviceListener, InlineExecutor

SAMPLES = FRAME_BYTES // 2


def fr(speech: bool = False, wake: float = 0.0, level: int = 0) -> bytes:
    """A scripted frame: sample 0 carries the wake score, sample 1 the VAD verdict."""
    body = [level] * SAMPLES
    body[0] = int(round(wake * 1000))
    body[1] = 1 if speech else 0
    return struct.pack(f"<{SAMPLES}h", *body)


class FrameWake:
    """Scores like openWakeWord: one score per 80 ms (4 frames), the max tagged in that chunk."""

    def __init__(self) -> None:
        self.frames = 0
        self._cur = 0.0

    def feed(self, frame: bytes) -> float | None:
        self._cur = max(self._cur, struct.unpack_from("<h", frame, 0)[0] / 1000)
        self.frames += 1
        if self.frames % 4:
            return None
        out, self._cur = self._cur, 0.0
        return out


class FrameVad:
    def speech_prob(self, frame: bytes) -> float:
        return float(struct.unpack_from("<h", frame, 2)[0])


class FakeClock:
    def __init__(self) -> None:
        self.t = 5000.0

    def __call__(self) -> float:
        return self.t


class Rig:
    """Listeners sharing one arbiter and one fake clock that advances 20 ms per frame step."""

    def __init__(self) -> None:
        self.clock = FakeClock()
        self.arbiter = WakeArbiter(window_ms=400, clock=self.clock)
        self.got: list[tuple[str, bytes, str]] = []
        self.wakes: list[str] = []

    def listener(self, device_id: str, room: str, group: str | None = None) -> DeviceListener:
        lst = DeviceListener(
            device_id,
            FrameWake(),
            Endpointer(FrameVad()),
            self.arbiter,
            0.5,
            room=room,
            group=group,
            executor=InlineExecutor(),
            dispatch=lambda fn: fn(),
            clock=self.clock,
        )
        lst.on_utterance(lambda dev, utt, kind: self.got.append((dev, utt, kind)))
        lst.on_wake(self.wakes.append)
        return lst

    def feed(self, plan: dict[DeviceListener, list[bytes]]) -> None:
        """Feed several listeners in lock-step, one frame each per 20 ms."""
        n = max(len(v) for v in plan.values())
        for i in range(n):
            for lst, frames in plan.items():
                if i < len(frames):
                    lst.feed(frames[i])
            self.clock.t += 0.02

    def kinds(self) -> list[tuple[str, str]]:
        return [(dev, kind) for dev, _, kind in self.got]


def silence(n: int, level: int = 0) -> list[bytes]:
    return [fr(level=level) for _ in range(n)]


def speech(n: int, level: int = 0) -> list[bytes]:
    return [fr(speech=True, level=level) for _ in range(n)]


def wake_phrase(score: float, level: int = 0, chunks: int = 1) -> list[bytes]:
    """~0.4 s of speech whose last ``chunks`` 80 ms chunks score ``score``."""
    return speech(16, level) + [fr(speech=True, wake=score, level=level) for _ in range(4 * chunks)]


def test_wake_then_command_emits_one_wake_utterance():
    rig = Rig()
    phone = rig.listener("phone", "kitchen")
    rig.feed({phone: silence(20) + wake_phrase(0.99) + speech(40) + silence(40)})
    assert rig.kinds() == [("phone", "wake")]
    assert rig.wakes == ["phone"]
    utt = rig.got[0][1]
    assert not utt.truncated
    assert len(utt.pcm) >= 40 * FRAME_BYTES


def test_speech_without_wake_or_window_is_ignored():
    rig = Rig()
    phone = rig.listener("phone", "kitchen")
    rig.feed({phone: silence(10) + speech(50) + silence(40)})
    assert rig.got == []


def test_wake_without_command_times_out():
    rig = Rig()
    phone = rig.listener("phone", "kitchen")
    rig.feed({phone: silence(20) + wake_phrase(0.99) + silence(400)})
    assert rig.got == []
    rig.feed({phone: speech(30) + silence(40)})  # later talk is not a command
    assert rig.got == []


def test_followup_window_accepts_speech_without_wake():
    rig = Rig()
    phone = rig.listener("phone", "kitchen")
    phone.open_followup(6.0)
    rig.feed({phone: silence(100) + speech(30) + silence(40)})  # starts 2 s into the window
    assert rig.kinds() == [("phone", "followup")]
    # the window closes after one utterance
    rig.feed({phone: speech(30) + silence(40)})
    assert len(rig.got) == 1
    # and a window nobody uses expires
    phone.open_followup(1.0)
    rig.feed({phone: silence(60) + speech(30) + silence(40)})
    assert len(rig.got) == 1


def test_playback_raises_threshold_and_blocks_followup():
    rig = Rig()
    phone = rig.listener("phone", "kitchen", group="downstairs")
    laptop = rig.listener("laptop", "office", group="downstairs")
    laptop.set_playing(True)  # a device in the same earshot group is talking

    phone.open_followup(6.0)
    rig.feed({phone: speech(30) + silence(40)})
    assert rig.got == []  # no follow-up capture while the group plays

    rig.feed({phone: silence(20) + wake_phrase(0.6) + speech(40) + silence(40)})
    assert rig.got == [] and rig.wakes == []  # 0.6 < 0.5 + 0.15

    rig.feed({phone: silence(20) + wake_phrase(0.7) + speech(40) + silence(40)})
    assert rig.kinds() == [("phone", "wake")]  # a clear wake still barges in

    laptop.set_playing(False)
    phone.open_followup(6.0)
    rig.feed({phone: speech(30) + silence(40)})
    assert rig.kinds()[-1] == ("phone", "followup")

    phone.set_playing(True)  # the device's own playback blocks it too
    phone.open_followup(6.0)
    rig.feed({phone: speech(30) + silence(40)})
    assert len(rig.got) == 2


def test_ptt_bypasses_wake():
    rig = Rig()
    phone = rig.listener("phone", "kitchen")
    phone.set_ptt(True)
    talk = speech(30, level=7)
    rig.feed({phone: talk})
    phone.set_ptt(False)
    assert rig.kinds() == [("phone", "ptt")]
    assert rig.got[0][1].pcm == b"".join(talk)
    # a tap too short to hold speech is dropped
    phone.set_ptt(True)
    rig.feed({phone: speech(5)})
    phone.set_ptt(False)
    assert len(rig.got) == 1


def test_muted_device_emits_nothing():
    rig = Rig()
    phone = rig.listener("phone", "kitchen")
    phone.set_muted(True)
    rig.feed({phone: silence(20) + wake_phrase(0.99) + speech(40) + silence(40)})
    phone.open_followup(6.0)
    rig.feed({phone: speech(30) + silence(40)})
    phone.set_ptt(True)
    rig.feed({phone: speech(30)})
    phone.set_ptt(False)
    assert rig.got == [] and rig.wakes == []
    phone.set_muted(False)
    rig.feed({phone: silence(20) + wake_phrase(0.99) + speech(40) + silence(40)})
    assert rig.kinds() == [("phone", "wake")]


def test_two_devices_in_earshot_only_the_winner_answers():
    rig = Rig()
    phone = rig.listener("phone", "kitchen", group="downstairs")
    laptop = rig.listener("laptop", "office", group="downstairs")
    # the score stays high for 800 ms, longer than the 400 ms window: the loser must not
    # re-offer the same wake and win a fresh round of its own
    near = silence(20) + wake_phrase(0.99, level=3000, chunks=10) + speech(40, level=3000) + silence(40)
    far = silence(20) + wake_phrase(0.98, level=300, chunks=10) + speech(40, level=300) + silence(40)
    rig.feed({phone: near, laptop: far})
    assert rig.kinds() == [("phone", "wake")]
    assert rig.wakes == ["phone"]
    assert laptop.stats["lost"] == 1 and phone.stats["won"] == 1
    # the loser re-arms normally afterwards
    rig.feed({laptop: silence(60) + wake_phrase(0.99) + speech(40) + silence(40)})
    assert rig.kinds() == [("phone", "wake"), ("laptop", "wake")]


async def test_inference_runs_off_loop_and_callback_returns_to_loop():
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int, int]] = []
    done = asyncio.Event()
    pool = ThreadPoolExecutor(max_workers=2)
    worker_threads: set[int] = set()

    class ThreadWake(FrameWake):
        def feed(self, frame):
            worker_threads.add(threading.get_ident())
            return super().feed(frame)

    lst = DeviceListener(
        "phone", ThreadWake(), Endpointer(FrameVad()), WakeArbiter(), 0.5, room="kitchen", executor=pool
    )

    def cb(dev, utt, kind):
        seen.append((kind, threading.get_ident(), len(utt.pcm)))
        done.set()

    lst.on_utterance(cb)
    lst.set_ptt(True)
    for f in speech(30):
        lst.feed(f)
    lst.set_ptt(False)
    await asyncio.wait_for(done.wait(), 5)
    pool.shutdown(wait=True)
    assert seen == [("ptt", loop_thread, 30 * FRAME_BYTES)]
    assert worker_threads and loop_thread not in worker_threads
