import asyncio

import pytest

from housemate.core.playout import PcmRealigner, Playout

RATE = 24_000
BPS = RATE * 2  # bytes per second, PCM16 mono


class FakeTime:
    """A clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.t = 100.0

    def clock(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        await asyncio.sleep(0)


class Sink:
    def __init__(self, time: FakeTime, fail_after: int | None = None) -> None:
        self.time = time
        self.frames: list[tuple[float, bytes]] = []
        self.fail_after = fail_after

    async def __call__(self, pcm: bytes) -> bool:
        if self.fail_after is not None and len(self.frames) >= self.fail_after:
            return False
        self.frames.append((self.time.t, bytes(pcm)))
        return True

    @property
    def data(self) -> bytes:
        return b"".join(f for _, f in self.frames)


def tone(seconds: float) -> bytes:
    n = int(seconds * RATE)
    return bytes((i * 7) & 0xFF for i in range(n * 2))


# --- realignment ----------------------------------------------------------------------------------


def test_realigner_carries_the_odd_byte():
    r = PcmRealigner()
    out = [r.feed(c) for c in (b"\x01\x02\x03", b"\x04\x05", b"\x06", b"\x07\x08\x09")]
    assert all(len(o) % 2 == 0 for o in out)
    assert b"".join(out) == b"\x01\x02\x03\x04\x05\x06\x07\x08"
    assert r.pending == 1
    assert r.flush() == b""  # a dangling half sample is dropped, never sent
    assert r.pending == 0


@pytest.mark.parametrize("sizes", [[1, 1, 1, 1], [52, 4901, 1223, 3], [7, 9, 11]])
def test_realigner_never_splits_a_sample(sizes):
    r = PcmRealigner()
    data = bytes(range(256)) * 40
    pos, out = 0, []
    for size in sizes:
        out.append(r.feed(data[pos : pos + size]))
        pos += size
    joined = b"".join(out)
    assert len(joined) % 2 == 0
    assert joined == data[: len(joined)]
    assert len(joined) + r.pending == pos


# --- pacing ---------------------------------------------------------------------------------------


async def test_playout_paces_faster_than_real_time_input():
    time = FakeTime()
    sink = Sink(time)
    play = Playout(sink, sample_rate=RATE, lead_s=0.5, clock=time.clock, sleep=time.sleep)
    pcm = tone(3.0)
    play.write(pcm)  # TTS hands over 3 s of audio at once
    play.finish()
    start = time.t
    assert await play.run() == "done"
    assert sink.data == pcm
    first = sink.frames[0][0]
    sent = 0
    for at, frame in sink.frames:
        sent += len(frame)
        # never more than the lead (plus one frame) ahead of the device's playhead
        assert sent / BPS <= (at - first) + 0.5 + play.max_frame_s + 1e-6
    # it took roughly real time minus the lead to hand over 3 s of audio
    assert 2.3 <= time.t - start <= 3.0
    assert play.sent_bytes == len(pcm)


async def test_playout_sends_whole_samples_only():
    time = FakeTime()
    sink = Sink(time)
    play = Playout(sink, sample_rate=RATE, clock=time.clock, sleep=time.sleep)
    play.write(tone(0.5))
    play.finish()
    await play.run()
    assert all(len(frame) % 2 == 0 for _, frame in sink.frames)


async def test_playout_streams_while_writes_arrive():
    time = FakeTime()
    sink = Sink(time)
    play = Playout(sink, sample_rate=RATE, clock=time.clock, sleep=time.sleep)
    runner = asyncio.create_task(play.run())
    pcm = tone(1.0)
    for i in range(0, len(pcm), 4800):
        play.write(pcm[i : i + 4800])
        await asyncio.sleep(0)
    play.finish()
    assert await asyncio.wait_for(runner, 2) == "done"
    assert sink.data == pcm


async def test_playout_reports_failure_when_device_drops():
    time = FakeTime()
    sink = Sink(time, fail_after=1)
    play = Playout(sink, sample_rate=RATE, clock=time.clock, sleep=time.sleep)
    play.write(tone(2.0))
    play.finish()
    assert await play.run() == "failed"
    play.write(b"\x00\x00")  # late writes after a failure are ignored, never raise
    assert len(sink.frames) == 1


async def test_playout_abort_stops_quickly():
    time = FakeTime()
    sink = Sink(time)
    play = Playout(sink, sample_rate=RATE, clock=time.clock, sleep=time.sleep)
    runner = asyncio.create_task(play.run())
    play.write(tone(5.0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    play.abort()
    assert await asyncio.wait_for(runner, 2) == "aborted"
    assert len(sink.data) < len(tone(5.0))


async def test_playout_waits_for_more_data_without_spinning():
    time = FakeTime()
    sink = Sink(time)
    play = Playout(sink, sample_rate=RATE, clock=time.clock, sleep=time.sleep)
    runner = asyncio.create_task(play.run())
    await asyncio.sleep(0.01)  # nothing written yet: the pump just waits
    assert sink.frames == []
    play.write(tone(0.2))
    play.finish()
    assert await asyncio.wait_for(runner, 2) == "done"
    assert len(sink.data) == len(tone(0.2))
