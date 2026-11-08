import struct

import pytest

from housemate.core.audio import FRAME_BYTES, SAMPLE_RATE, pcm_to_wav
from housemate.core.endpointer import Endpointer, Utterance

SAMPLES = FRAME_BYTES // 2


def frame(tag: int) -> bytes:
    """A 20 ms frame whose samples all equal ``tag`` (so frames can be told apart in the pcm)."""
    return struct.pack(f"<{SAMPLES}h", *([tag] * SAMPLES))


class ScriptedVad:
    """Returns the scripted probabilities in order (0.0 once the script runs out)."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.calls = 0

    def speech_prob(self, frame: bytes) -> float:
        p = self.probs[self.calls] if self.calls < len(self.probs) else 0.0
        self.calls += 1
        return p


def run(ep: Endpointer, n: int, tag_start: int = 1):
    out = []
    for i in range(n):
        utt = ep.push(frame(tag_start + i))
        if utt is not None:
            out.append((i, utt))
    return out


def test_endpoint_after_700ms_silence():
    vad = ScriptedVad([0.0] * 10 + [0.9] * 50 + [0.1] * 40)
    ep = Endpointer(vad, clock=lambda: 100.0)
    got = run(ep, 100)
    assert len(got) == 1
    idx, utt = got[0]
    assert idx == 10 + 50 + 35 - 1  # the 35th silent frame (700 ms) closes it, not earlier
    assert isinstance(utt, Utterance)
    assert not utt.truncated
    # preroll (only 10 frames existed) + 50 speech frames + 300 ms tail
    assert len(utt.pcm) == (10 + 50 + 15) * FRAME_BYTES
    assert frame(11) * 50 != utt.pcm  # sanity: tags differ per frame
    assert utt.pcm[10 * FRAME_BYTES : 11 * FRAME_BYTES] == frame(11)
    assert utt.started_at == pytest.approx(100.0 - 95 * 0.02)
    assert utt.ended_at == pytest.approx(utt.started_at + 75 * 0.02)


def test_too_short_speech_discarded():
    vad = ScriptedVad([0.0] * 10 + [0.9] * 12 + [0.0] * 60)  # 240 ms < 250 ms
    ep = Endpointer(vad)
    assert run(ep, 82) == []
    assert not ep.active


def test_short_speech_then_real_speech_still_ends():
    vad = ScriptedVad([0.9] * 5 + [0.0] * 40 + [0.9] * 20 + [0.0] * 40)
    ep = Endpointer(vad)
    got = run(ep, 105)
    assert len(got) == 1
    assert got[0][0] == 5 + 40 + 20 + 35 - 1


def test_max_utterance_truncates_at_15s():
    frames_15s = 15 * 1000 // 20
    vad = ScriptedVad([0.9] * 1000 + [0.0] * 40 + [0.9] * 20 + [0.0] * 40)  # 20 s of talk, then a new phrase
    ep = Endpointer(vad)
    got = run(ep, 1100)
    assert len(got) == 2
    idx, utt = got[0]
    assert idx == frames_15s - 1
    assert utt.truncated
    assert len(utt.pcm) == 15 * SAMPLE_RATE * 2
    # the rest of the long talk is not a second utterance; the next phrase after silence is
    idx2, utt2 = got[1]
    assert idx2 == 1000 + 40 + 20 + 35 - 1
    assert not utt2.truncated


def test_preroll_included():
    vad = ScriptedVad([0.0] * 30 + [0.9] * 20 + [0.0] * 40)
    ep = Endpointer(vad)
    got = run(ep, 90, tag_start=1)
    assert len(got) == 1
    utt = got[0][1]
    preroll = b"".join(frame(t) for t in range(16, 31))  # the 15 frames (300 ms) before speech
    assert utt.pcm.startswith(preroll)
    assert utt.pcm[15 * FRAME_BYTES : 16 * FRAME_BYTES] == frame(31)  # first speech frame


def test_restart_drops_segment_but_keeps_preroll():
    vad = ScriptedVad([0.0] * 20 + [0.9] * 30 + [0.9] * 20 + [0.0] * 40)
    ep = Endpointer(vad)
    for i in range(50):
        assert ep.push(frame(1 + i)) is None
    assert ep.active
    ep.restart()
    assert not ep.active
    got = []
    for i in range(50, 110):
        utt = ep.push(frame(1 + i))
        if utt is not None:
            got.append(utt)
    assert len(got) == 1
    # new segment: 15 preroll frames (the last of the old talk) + 20 speech + 15 tail
    assert len(got[0].pcm) == (15 + 20 + 15) * FRAME_BYTES
    assert got[0].pcm.startswith(b"".join(frame(t) for t in range(36, 51)))


def test_restart_without_preroll_starts_after_the_restart_point():
    # 20 frames of the wake word, a restart at its end, a short pause, then the command
    vad = ScriptedVad([0.9] * 20 + [0.1] * 5 + [0.9] * 30 + [0.1] * 40)
    ep = Endpointer(vad)
    assert run(ep, 20, tag_start=1) == []
    ep.restart(keep_preroll=False, skip_ms=40)
    got = run(ep, 75, tag_start=21)
    assert len(got) == 1
    utt = got[0][1]
    # nothing from before the restart, and the 2 skipped frames (40 ms) are gone too:
    # preroll = the 3 remaining pause frames, then 30 speech frames + 300 ms tail
    assert utt.pcm.startswith(frame(23))
    assert len(utt.pcm) == (3 + 30 + 15) * FRAME_BYTES


def test_restart_without_preroll_while_speech_runs_on():
    # the command follows the wake word with no pause: the clip starts right after the skip
    vad = ScriptedVad([0.9] * 20 + [0.9] * 30 + [0.1] * 40)
    ep = Endpointer(vad)
    run(ep, 20, tag_start=1)
    ep.restart(keep_preroll=False, skip_ms=40)
    got = run(ep, 70, tag_start=21)
    assert len(got) == 1
    assert got[0][1].pcm.startswith(frame(23))
    assert len(got[0][1].pcm) == (28 + 15) * FRAME_BYTES


def test_plain_restart_cancels_a_pending_skip():
    vad = ScriptedVad([0.9] * 20 + [0.0] * 20 + [0.9] * 20 + [0.0] * 40)
    ep = Endpointer(vad)
    run(ep, 20, tag_start=1)
    ep.restart(keep_preroll=False, skip_ms=400)
    ep.restart()  # e.g. a follow-up window opened: back to the normal preroll
    got = run(ep, 80, tag_start=21)
    assert len(got) == 1
    assert got[0][1].pcm.startswith(b"".join(frame(t) for t in range(26, 41)))


def test_pcm_to_wav_header():
    pcm = frame(5) * 3
    wav = pcm_to_wav(pcm)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert struct.unpack("<I", wav[24:28])[0] == 16000
    assert wav.endswith(pcm)
    assert len(wav) == 44 + len(pcm)
