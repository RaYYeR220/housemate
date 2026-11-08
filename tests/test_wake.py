"""Real-model tests: download the pinned models once into the repo's gitignored ``.models/``."""

import json
from pathlib import Path

import pytest

from housemate.core.arbiter import WakeArbiter
from housemate.core.audio import CHUNK_SAMPLES, FRAME_BYTES, iter_frames, read_wav
from housemate.core.listener import DeviceListener, InlineExecutor
from housemate.core.models import ModelDownloadError, load_audio_models

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"
CACHE = Path(__file__).resolve().parents[1] / ".models"
THRESHOLD = 0.5


@pytest.fixture(scope="module")
def audio_models():
    try:
        return load_audio_models(CACHE, wake_model="hey_jarvis")
    except ModelDownloadError as exc:  # offline machine: nothing to test against
        pytest.skip(f"models not available: {exc}")


def padded(pcm: bytes) -> bytes:
    """The spike's padding: 1.5 s before / 1 s after of quiet noise, whole 80 ms chunks."""
    import numpy as np

    rng = np.random.RandomState(7)
    pre = (rng.randn(24000) * 30).astype(np.int16)
    post = (rng.randn(16000) * 30).astype(np.int16)
    x = np.concatenate([pre, np.frombuffer(pcm, dtype="<i2"), post])
    return x[: len(x) // CHUNK_SAMPLES * CHUNK_SAMPLES].astype("<i2").tobytes()


def scores(det, pcm: bytes) -> list[float]:
    det.reset()
    return [s for f in iter_frames(pcm) if (s := det.feed(f)) is not None]


def test_matches_spike_reference_scores(audio_models):
    ref = json.loads((FIXTURES / "wake_ref_scores.json").read_text())["files"]
    det = audio_models.new_wake()
    assert len(ref) >= 2
    for name, want in ref.items():
        got = scores(det, padded(read_wav(FIXTURES / name)))
        assert len(got) == len(want), name
        worst = max(abs(a - b) for a, b in zip(got, want))
        assert worst < 1e-4, f"{name}: max |delta| {worst}"


def test_detects_fixture_wakes(audio_models):
    det = audio_models.new_wake()
    wakes = sorted(FIXTURES.glob("wake_*.wav"))
    negs = sorted(FIXTURES.glob("neg_*.wav"))
    assert len(wakes) >= 10 and len(negs) >= 5
    hits = [w.name for w in wakes if max(scores(det, padded(read_wav(w)))) >= THRESHOLD]
    false = [n.name for n in negs if max(scores(det, padded(read_wav(n)))) >= THRESHOLD]
    assert len(hits) >= 9, f"only {len(hits)}/{len(wakes)} wakes detected"
    assert len(false) <= 1, f"false wakes: {false}"


def test_sessions_are_shared_between_devices(audio_models):
    a, b = audio_models.new_wake(), audio_models.new_wake()
    assert a._emb is b._emb and a._mel is b._mel and a._head is b._head
    va, vb = audio_models.new_vad(), audio_models.new_vad()
    if audio_models.vad is not None:
        assert va._s is vb._s


def test_full_pipeline_turns_a_spoken_command_into_one_utterance(audio_models):
    """Real wake model + Silero VAD + endpointer: "Hey Jarvis, what's two plus two?"."""
    t = [100.0]
    got = []
    lst = DeviceListener(
        "phone",
        audio_models.new_wake(),
        audio_models.new_endpointer(),
        WakeArbiter(clock=lambda: t[0]),
        THRESHOLD,
        room="kitchen",
        executor=InlineExecutor(),
        dispatch=lambda fn: fn(),
        clock=lambda: t[0],
    )
    lst.on_utterance(lambda dev, utt, kind: got.append((dev, utt, kind)))
    pcm = padded(read_wav(FIXTURES / "wake_jarvis-math_enusaria.wav")) + bytes(FRAME_BYTES * 75)
    for f in iter_frames(pcm):
        lst.feed(f)
        t[0] += 0.02
    assert [(d, k) for d, _, k in got] == [("phone", "wake")]
    utt = got[0][1]
    assert 0.8 < utt.duration_s < 3.0  # the command after the wake word, not the padding
