"""Streamed local speech: Piper voices as a Hermes TTS provider (``tts.provider: housemate-piper``).

Hermes's built-in Piper provider writes whole files. The gateway can only play a whole file after
the turn has ended, so a spoken reply starts ~1.3 s later than with a streaming provider, plus an
MP3 round trip through ffmpeg. This provider streams raw PCM sentence by sentence through Hermes's
plugin streaming path (``streams_pcm``), so a short answer starts playing ~0.1 s after the model
has finished. It uses the same voices, the same cache folder (``<Hermes home>/cache/piper-voices``)
and the ``piper-tts`` package that Hermes's ``piper`` extra installs; without it the provider
reports itself unavailable and Hermes falls back as usual.

Voice: ``tts.housemate-piper.voice``, else ``tts.piper.voice``, else ``en_US-lessac-medium``; a
voice name is downloaded once like Hermes does, or give a path to an ``.onnx`` voice.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

__all__ = ["DEFAULT_VOICE", "PROVIDER_NAME", "PiperEngine", "make_provider"]

log = logging.getLogger(__name__)

PROVIDER_NAME = "housemate-piper"
DEFAULT_VOICE = "en_US-lessac-medium"
DOWNLOAD_TIMEOUT_S = 300


def _hermes_voices_dir() -> Path:
    from hermes_constants import get_hermes_home

    return Path(get_hermes_home()) / "cache" / "piper-voices"


def _hermes_config() -> dict:
    from hermes_cli.config import load_config

    cfg = load_config()
    return cfg if isinstance(cfg, dict) else {}


def _load_piper(path: Path) -> Any:
    from piper import PiperVoice

    return PiperVoice.load(str(path))


def _download_voice(name: str, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "piper.download_voices", name, "--download-dir", str(folder)],
                   check=True, capture_output=True, timeout=DOWNLOAD_TIMEOUT_S, stdin=subprocess.DEVNULL)


def _piper_installed() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("piper") is not None
    except Exception:
        return False


class PiperEngine:
    """Voice lookup, loading (once per voice), and serialised synthesis to PCM or WAV."""

    def __init__(
        self,
        *,
        voices_dir: Callable[[], Path] = _hermes_voices_dir,
        config: Callable[[], dict] = _hermes_config,
        loader: Callable[[Path], Any] = _load_piper,
        downloader: Callable[[str, Path], None] = _download_voice,
        available: Callable[[], bool] = _piper_installed,
    ) -> None:
        self._voices_dir = voices_dir
        self._config = config
        self._loader = loader
        self._downloader = downloader
        self._available = available
        self._voices: dict[Path, Any] = {}
        self._lock = threading.Lock()  # espeak-ng phonemisation is not thread-safe

    def available(self) -> bool:
        return bool(self._available())

    def voice_name(self, voice: Optional[str] = None) -> str:
        if voice:
            return str(voice)
        try:
            tts = self._config().get("tts") or {}
        except Exception:
            tts = {}
        for section in (PROVIDER_NAME, "piper"):
            value = (tts.get(section) or {}).get("voice") if isinstance(tts.get(section), dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return DEFAULT_VOICE

    def _path(self, voice: Optional[str]) -> Path:
        name = self.voice_name(voice)
        candidate = Path(name).expanduser()
        if candidate.suffix.lower() == ".onnx":
            return candidate
        return self._voices_dir() / f"{name}.onnx"

    def sample_rate(self, voice: Optional[str] = None) -> int | None:
        """The voice's rate from its ``.onnx.json``; None until the voice is on disk."""
        meta = Path(str(self._path(voice)) + ".json")
        try:
            rate = json.loads(meta.read_text(encoding="utf-8"))["audio"]["sample_rate"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return int(rate) if isinstance(rate, int) and rate > 0 else None

    def load(self, voice: Optional[str] = None) -> Any:
        path = self._path(voice)
        with self._lock:
            loaded = self._voices.get(path)
            if loaded is not None:
                return loaded
            if not path.exists():
                name = self.voice_name(voice)
                if Path(name).suffix.lower() == ".onnx":
                    raise FileNotFoundError(f"Piper voice {path} not found")
                log.info("housemate: downloading Piper voice %s", name)
                self._downloader(name, path.parent)
            loaded = self._loader(path)
            self._voices[path] = loaded
            return loaded

    def stream_pcm(self, text: str, voice: Optional[str] = None) -> Iterator[bytes]:
        """PCM16 LE mono at ``sample_rate()``, one chunk per sentence."""
        piper_voice = self.load(voice)
        with self._lock:
            chunks = [c.audio_int16_bytes for c in piper_voice.synthesize(text)]
        for pcm in chunks:
            if pcm:
                yield pcm[: len(pcm) - len(pcm) % 2]

    def write_wav(self, text: str, path: Path | str, voice: Optional[str] = None) -> Path:
        path = Path(path)
        rate = None
        frames = bytearray()
        piper_voice = self.load(voice)
        with self._lock:
            for chunk in piper_voice.synthesize(text):
                rate = rate or chunk.sample_rate
                frames += chunk.audio_int16_bytes
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(rate or self.sample_rate(voice) or 22050))
            w.writeframes(bytes(frames))
        return path


def make_provider(engine: PiperEngine | None = None) -> Any:
    """The Hermes ``TTSProvider`` (imported here so the rest of the module needs no Hermes)."""
    from agent.tts_provider import TTSProvider

    eng = engine or PiperEngine()

    class HousematePiperTTS(TTSProvider):
        streams_pcm = True

        @property
        def name(self) -> str:
            return PROVIDER_NAME

        @property
        def display_name(self) -> str:
            return "Piper, streamed (Housemate)"

        @property
        def stream_sample_rate(self) -> int | None:  # read each time a reply resolves its streamer
            return eng.sample_rate()

        def is_available(self) -> bool:
            return eng.available()

        def list_voices(self) -> list[dict[str, Any]]:
            return [{"id": eng.voice_name(), "language": "en"}]

        def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None,
                       model: Optional[str] = None, speed: Optional[float] = None, format: str = "mp3",
                       **extra: Any) -> str:
            # Piper makes WAV; hand that back (the documented "closest format") rather than paying
            # an ffmpeg round trip that Housemate would only decode again.
            wav = output_path if output_path.lower().endswith(".wav") else output_path.rsplit(".", 1)[0] + ".wav"
            return str(eng.write_wav(text, wav, voice))

        def stream(self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
                   format: str = "opus", **extra: Any) -> Iterator[bytes]:
            if format != "pcm":
                raise NotImplementedError("housemate-piper streams raw PCM only")
            yield from eng.stream_pcm(text, voice)

        def warm(self) -> None:
            try:
                eng.load()
            except Exception:
                log.debug("housemate: Piper warm-up failed", exc_info=True)

    return HousematePiperTTS()
