import json
import threading
import wave
from pathlib import Path

import pytest

from housemate.core.tts_piper import DEFAULT_VOICE, PROVIDER_NAME, PiperEngine


class Chunk:
    def __init__(self, pcm: bytes, rate: int = 22050) -> None:
        self.audio_int16_bytes = pcm
        self.sample_rate, self.sample_width, self.sample_channels = rate, 2, 1


class FakeVoice:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.active = 0
        self.overlap = False

    def synthesize(self, text: str):
        self.active += 1
        self.overlap |= self.active > 1
        self.texts.append(text)
        try:
            for sentence in [s for s in text.split(".") if s.strip()]:
                yield Chunk(sentence.strip().encode()[:2].ljust(2, b"\x00") * 100)
        finally:
            self.active -= 1


def voice_files(d: Path, name: str, rate: int = 22050) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.onnx").write_bytes(b"onnx")
    (d / f"{name}.onnx.json").write_text(json.dumps({"audio": {"sample_rate": rate}}), encoding="utf-8")
    return d / f"{name}.onnx"


def engine(tmp_path, config=None, **kw):
    loads: list[Path] = []
    voice = FakeVoice()

    def loader(path):
        loads.append(Path(path))
        return voice

    eng = PiperEngine(voices_dir=lambda: tmp_path / "voices", config=lambda: config or {}, loader=loader, **kw)
    return eng, voice, loads


def test_voice_comes_from_config_then_hermes_piper_then_default(tmp_path):
    eng, _, _ = engine(tmp_path)
    assert eng.voice_name() == DEFAULT_VOICE
    eng, _, _ = engine(tmp_path, {"tts": {"piper": {"voice": "en_GB-alba-medium"}}})
    assert eng.voice_name() == "en_GB-alba-medium"
    eng, _, _ = engine(tmp_path, {"tts": {"piper": {"voice": "en_GB-alba-medium"},
                                          PROVIDER_NAME: {"voice": "en_US-amy-medium"}}})
    assert eng.voice_name() == "en_US-amy-medium"
    assert eng.voice_name("en_US-ryan-high") == "en_US-ryan-high"


def test_stream_yields_pcm_per_sentence_and_loads_once(tmp_path):
    voice_files(tmp_path / "voices", DEFAULT_VOICE)
    eng, voice, loads = engine(tmp_path)
    assert eng.sample_rate() == 22050
    chunks = list(eng.stream_pcm("Four. Nine."))
    assert len(chunks) == 2 and all(len(c) % 2 == 0 and c for c in chunks)
    list(eng.stream_pcm("Again."))
    assert loads == [tmp_path / "voices" / f"{DEFAULT_VOICE}.onnx"]  # the voice stays in memory


def test_rate_unknown_until_the_voice_is_there(tmp_path):
    eng, _, _ = engine(tmp_path)
    assert eng.sample_rate() is None  # never stream at a guessed rate


def test_missing_voice_is_downloaded_once(tmp_path):
    fetched: list[str] = []

    def download(name, folder):
        fetched.append(name)
        voice_files(folder, name, rate=16000)

    eng, _, _ = engine(tmp_path, downloader=download)
    list(eng.stream_pcm("Hello."))
    list(eng.stream_pcm("Hello."))
    assert fetched == [DEFAULT_VOICE] and eng.sample_rate() == 16000


def test_absolute_onnx_path_is_used_as_is(tmp_path):
    path = voice_files(tmp_path / "elsewhere", "my-voice", rate=24000)
    eng, _, loads = engine(tmp_path, {"tts": {PROVIDER_NAME: {"voice": str(path)}}})
    list(eng.stream_pcm("Hi."))
    assert loads == [path] and eng.sample_rate() == 24000


def test_synthesis_is_serialised_across_threads(tmp_path):
    voice_files(tmp_path / "voices", DEFAULT_VOICE)
    eng, voice, _ = engine(tmp_path)
    threads = [threading.Thread(target=lambda: list(eng.stream_pcm("One. Two. Three."))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(voice.texts) == 4 and not voice.overlap


def test_write_wav(tmp_path):
    voice_files(tmp_path / "voices", DEFAULT_VOICE)
    eng, _, _ = engine(tmp_path)
    out = eng.write_wav("Four.", tmp_path / "out.wav")
    with wave.open(str(out), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 22050) and w.getnframes() == 100


@pytest.mark.hermes
def test_provider_streams_pcm_for_the_gateway(tmp_path):
    pytest.importorskip("agent.tts_provider")
    from housemate.core.tts_piper import make_provider

    voice_files(tmp_path / "voices", DEFAULT_VOICE)
    eng, _, _ = engine(tmp_path, available=lambda: True)
    provider = make_provider(eng)
    assert provider.name == PROVIDER_NAME and provider.streams_pcm and provider.stream_sample_rate == 22050
    assert provider.is_available()
    assert b"".join(provider.stream("Four.", format="pcm"))
    with pytest.raises(NotImplementedError):
        next(iter(provider.stream("Four.", format="opus")))
    written = provider.synthesize("Four.", str(tmp_path / "reply.mp3"), format="mp3")
    assert written.endswith(".wav") and Path(written).exists()  # WAV is what Piper makes; no ffmpeg round trip


def test_voice_loaded_once_per_process_even_with_two_providers(tmp_path):
    # Hermes may register the plugin's provider again (re-discovery); the voice must stay loaded.
    voice_files(tmp_path / "voices", DEFAULT_VOICE)
    loads = []
    voice = FakeVoice()

    def loader(path):
        loads.append(path)
        return voice

    first = PiperEngine(voices_dir=lambda: tmp_path / "voices", config=dict, loader=loader)
    second = PiperEngine(voices_dir=lambda: tmp_path / "voices", config=dict, loader=loader)
    list(first.stream_pcm("One."))
    list(second.stream_pcm("Two."))
    assert len(loads) == 1
