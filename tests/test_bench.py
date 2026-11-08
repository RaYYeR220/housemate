import json

import pytest

from housemate.bench import corpus
from housemate.bench.run import FalseAcceptCounter, breakdown, match_turn, percentile, speech_end_offset


def test_false_accept_counter_needs_rearm_and_refractory():
    c = FalseAcceptCounter(threshold=0.5, refractory_s=1.0)
    # one long high stretch is one false accept
    for i, s in enumerate([0.1, 0.9, 0.95, 0.9, 0.2]):
        c.feed(s, i * 0.08)
    assert c.count == 1
    # a second crossing 0.4 s later is inside the refractory second
    c.feed(0.9, 0.72)
    assert c.count == 1
    c.feed(0.1, 1.6)
    c.feed(0.8, 1.68)
    assert c.count == 2 and c.events == [0.08, 1.68]


def test_percentile_nearest_rank():
    vals = [3.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(vals, 50) == 5.0
    assert percentile(vals, 95) == 10.0
    assert percentile([], 50) is None


def test_match_turn_picks_the_turn_that_ended_after_this_speech():
    turns = [
        {"message_id": "old", "endpoint_at": 100.0, "stages": {}},
        {"message_id": "mine", "endpoint_at": 200.35, "stages": {}},
        {"message_id": "later", "endpoint_at": 260.0, "stages": {}},
    ]
    assert match_turn(turns, speech_end=200.0)["message_id"] == "mine"
    assert match_turn(turns, speech_end=150.0) is None


def test_breakdown_is_relative_to_the_end_of_speech():
    turn = {"endpoint_at": 100.3, "stages": {"endpoint": 0.0, "captured": 0.4, "dispatch": 0.41, "transcript": 1.2,
                                             "llm_start": 1.3, "tts_begin": 3.9, "first_pcm_out": 3.91}}
    got = breakdown(turn, speech_end=100.0, first_pcm_at=104.25)
    assert got["captured"] == pytest.approx(0.7)
    assert got["transcript"] == pytest.approx(1.5)
    assert got["first_pcm_out"] == pytest.approx(4.21)
    assert got["satellite_first_pcm"] == pytest.approx(4.25)


def test_speech_end_offset_finds_the_last_speech_frame():
    class Vad:
        def __init__(self):
            self.n = 0

        def speech_prob(self, frame):
            self.n += 1
            return 0.9 if 10 <= self.n <= 40 else 0.0

    pcm = b"\x00\x00" * 320 * 60  # 60 frames of 20 ms
    assert speech_end_offset(pcm, Vad()) == pytest.approx(0.8)  # frame 40 ends at 800 ms


def test_negative_corpus_is_pinned_public_domain():
    total = sum(f.seconds for f in corpus.NEGATIVES)
    assert 9 * 3600 < total < 9.5 * 3600
    assert all(len(f.sha1) == 40 and f.url.startswith("https://archive.org/download/") for f in corpus.NEGATIVES)
    assert "public domain" in corpus.NEGATIVE_LICENCE.lower()


def test_pick_negatives_respects_the_hour_limit():
    picked = corpus.pick_negatives(max_hours=1.0)
    assert sum(f.seconds for f in picked) >= 3600
    assert sum(f.seconds for f in picked[:-1]) < 3600
    assert corpus.pick_negatives(None) == list(corpus.NEGATIVES)


def test_results_metadata_never_contains_keys(monkeypatch):
    from housemate.bench import run

    cfg = {"model": {"provider": "custom", "base_url": "https://api.example.com/v1", "default": "m", "api_key": "sk-secret"},
           "stt": {"provider": "local", "local": {"model": "small", "device": "cuda"}},
           "tts": {"provider": "cloud", "cloud": {"model": "tts-kokoro", "voice": "af_sky", "api_key": "sk-x",
                                                  "base_url": "https://api.example.com/v1"}},
           "platforms": {"housemate": {"end_silence_ms": 500}}}
    settings = run.settings_from_config(cfg)
    text = json.dumps(settings)
    assert "sk-" not in text
    assert settings["llm"] == {"provider": "custom", "model": "m", "host": "api.example.com"}
    assert settings["tts"]["model"] == "tts-kokoro" and settings["tts"]["host"] == "api.example.com"
    assert settings["stt"]["model"] == "small" and settings["housemate"]["end_silence_ms"] == 500
    piper = run.settings_from_config({"tts": {"provider": "housemate-piper", "piper": {"voice": "en_GB-alba-medium"}}})
    assert piper["tts"] == {"provider": "housemate-piper", "voice": "en_GB-alba-medium"}
