"""The bench corpus. Only small clips live in the repo; long audio is fetched by this script.

- Positives: the wake fixtures committed in ``tests/fixtures`` (``wake_*.wav``), plus clips made
  with Edge neural voices by ``make_positives`` (needs ``edge-tts`` and ffmpeg) into the cache.
- Negatives: LibriVox "Short Story Collection Vol. 062", 20 readings by different volunteers,
  9.2 h of public-domain speech from archive.org, pinned by SHA-1 (archive.org's own checksums).
  Decoded on the fly with ffmpeg; nothing is written but the downloaded MP3s.

Sources and licences: ``bench/corpus.md``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
CLIPS = ROOT / "bench" / "clips"

NEGATIVE_ITEM = "https://archive.org/download/shortstorycollection062_1512_librivox/"
NEGATIVE_LICENCE = "Public domain (LibriVox recordings; archive.org Public Domain Mark 1.0)"


@dataclass(frozen=True)
class CorpusFile:
    name: str
    sha1: str
    seconds: int

    @property
    def url(self) -> str:
        return NEGATIVE_ITEM + self.name


NEGATIVES: tuple[CorpusFile, ...] = tuple(CorpusFile(*row) for row in (
    ("shortstory062_01_wattleblossombride_64kb.mp3", "d4cd546a1e9729d76c5900bd1548f156f6aad0db", 790),
    ("shortstory062_02_smudge_64kb.mp3", "65427483cc2163ada671e171905a59becf9e24c7", 3744),
    ("shortstory062_03_fateandapothecary_64kb.mp3", "2ba60a8a1d2a857c8fa2a1c940d289f0d97563eb", 1431),
    ("shortstory062_04_oldmanofvisions_64kb.mp3", "5aa8e649df34ffb6beaf7e70cc7eddc4bb6fa4cf", 1486),
    ("shortstory062_05_houseofsuddhoo_64kb.mp3", "841e4445267c9111a7780fb0b95e8b2c3b05171c", 1128),
    ("shortstory062_06_songofthemorrow_64kb.mp3", "1dd36a6b4db7240153f1bb95dc1250329c5cd97b", 594),
    ("shortstory062_07_miracleofpurunbhagat_64kb.mp3", "b4a67cf91aada20b1645556cfb383eb6f8d6c2da", 2403),
    ("shortstory062_08_mrsbathurst_64kb.mp3", "e196b856cab1f34c7332a8f4d5911ee22516fae9", 2777),
    ("shortstory062_09_fatoftheland_64kb.mp3", "a68607e14795990dbdbfcc239c1a412e5d789a62", 2817),
    ("shortstory062_10_defenseofthedevil_64kb.mp3", "087cc2db687fcb5c11061bc36a2b135372bcab64", 690),
    ("shortstory062_11_griffinandtheminorcanon_64kb.mp3", "05b22595de18927f7948d464127cd2f47ed8c19d", 2706),
    ("shortstory062_12_lessingsfables_64kb.mp3", "e390c1447d70a0d16fdbdbd2bcdeb44c68a3fac8", 177),
    ("shortstory062_13_threefablesoflessing_64kb.mp3", "b07106a66880123918d4262de0482e8fce982418", 169),
    ("shortstory062_14_waifwoman_64kb.mp3", "36dfd7cfd0ba5662139313b0807ab2e020f91a38", 3035),
    ("shortstory062_15_bishop_64kb.mp3", "8654ea752a9007ffe6d66babf597f3b08b52ff02", 1922),
    ("shortstory062_16_voyageoflife_64kb.mp3", "43e54f972810a733c3eab396d74cc4c793534234", 663),
    ("shortstory062_17_sculptorsfuneral_64kb.mp3", "b1a2f08eff954c76f41bba960fd783ba8251c92f", 2082),
    ("shortstory062_18_grevillefane_64kb.mp3", "729b7ebd7a0724b7c2a65e71b0c4f7a643224528", 3393),
    ("shortstory062_19_olevirginia_64kb.mp3", "f63b61410d2f93b5d4c234001845beb49a0a14e5", 693),
    ("shortstory062_20_daveinlove_64kb.mp3", "5257781ba3c85383a1a3a730ce62b17e27ac1c24", 465),
))

# Edge neural voices for generated positives (English accents from 9 regions).
EDGE_VOICES = (
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural", "en-US-ChristopherNeural", "en-US-EricNeural",
    "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural", "en-GB-RyanNeural", "en-GB-LibbyNeural",
    "en-GB-MaisieNeural", "en-GB-ThomasNeural", "en-AU-NatashaNeural", "en-AU-WilliamNeural", "en-CA-ClaraNeural",
    "en-CA-LiamNeural", "en-IN-PrabhatNeural", "en-IN-NeerjaNeural", "en-IE-EmilyNeural", "en-NZ-MitchellNeural",
    "en-ZA-LeahNeural", "en-ZA-LukeNeural", "en-SG-WayneNeural", "en-PH-RosaNeural",
)
POSITIVE_PHRASES = ("Hey Jarvis.", "Hey Jarvis, what's the weather like tomorrow?")


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pick_negatives(max_hours: float | None) -> list[CorpusFile]:
    """The first files of the collection that add up to at least ``max_hours`` (None = all)."""
    if max_hours is None:
        return list(NEGATIVES)
    picked, total = [], 0.0
    for f in NEGATIVES:
        if total >= max_hours * 3600:
            break
        picked.append(f)
        total += f.seconds
    return picked


def fetch_negatives(cache_dir: Path, files: list[CorpusFile], log: Callable[[str], None] = print) -> list[Path]:
    """Download (once) and verify the negative recordings; returns their local paths."""
    out_dir = Path(cache_dir) / "negatives"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        path = out_dir / f.name
        if not (path.exists() and sha1_file(path) == f.sha1):
            log(f"downloading {f.url}")
            fd, tmp = tempfile.mkstemp(prefix=f".{f.name}.", suffix=".part", dir=out_dir)
            try:
                req = urllib.request.Request(f.url, headers={"User-Agent": "housemate-bench"})
                with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=60) as resp:
                    shutil.copyfileobj(resp, out, 1 << 20)
                got = sha1_file(Path(tmp))
                if got != f.sha1:
                    raise RuntimeError(f"{f.name}: sha1 {got} does not match the pinned {f.sha1}")
                os.replace(tmp, path)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        paths.append(path)
    return paths


def decode_16k(path: Path, *, seconds: float | None = None, block_bytes: int = 1280 * 2 * 50) -> Iterator[bytes]:
    """PCM16 LE mono 16 kHz of any audio file (the first ``seconds`` of it), streamed from ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is needed to decode the negative recordings")
    limit = ["-t", f"{seconds:g}"] if seconds else []
    proc = subprocess.Popen([ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path), *limit, "-f", "s16le", "-ac", "1",
                             "-ar", "16000", "-"], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    try:
        while True:
            block = proc.stdout.read(block_bytes)
            if not block:
                break
            yield block
    finally:
        proc.stdout.close()
        proc.wait()


def read_wav16k(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
            raise ValueError(f"{path}: need a 16 kHz mono PCM16 WAV")
        return w.readframes(w.getnframes())


def positive_files(cache_dir: Path | None, extra_dirs: list[Path] = ()) -> list[Path]:
    """Committed wake fixtures + generated clips in the cache + any extra folders."""
    dirs = [FIXTURES] + ([Path(cache_dir) / "positives"] if cache_dir else []) + [Path(d) for d in extra_dirs]
    files: list[Path] = []
    for d in dirs:
        pattern = "wake_*.wav" if d == FIXTURES else "*.wav"
        files += sorted(d.glob(pattern)) if d.is_dir() else []
    return files


def make_positives(cache_dir: Path, log: Callable[[str], None] = print) -> list[Path]:
    """Generate "Hey Jarvis" clips with Edge neural voices (skips voices that fail)."""
    import asyncio

    try:
        import edge_tts
    except ImportError:
        log("edge-tts is not installed: only the committed wake fixtures are used")
        return []
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log("ffmpeg is not installed: only the committed wake fixtures are used")
        return []
    out_dir = Path(cache_dir) / "positives"
    out_dir.mkdir(parents=True, exist_ok=True)

    async def one(voice: str, i: int, text: str) -> Path | None:
        wav = out_dir / f"wake{i}_{voice}.wav"
        if wav.exists():
            return wav
        mp3 = wav.with_suffix(".mp3")
        try:
            await asyncio.wait_for(edge_tts.Communicate(text, voice).save(str(mp3)), 60)
            subprocess.run([ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", str(mp3), "-ar", "16000", "-ac", "1",
                            "-sample_fmt", "s16", str(wav)], check=True, stdin=subprocess.DEVNULL)
            return wav
        except Exception as exc:
            log(f"skipped {voice} ({type(exc).__name__})")
            return None
        finally:
            mp3.unlink(missing_ok=True)

    async def all_clips() -> list[Path]:
        made = []
        for voice in EDGE_VOICES:
            for i, text in enumerate(POSITIVE_PHRASES):
                path = await one(voice, i, text)
                if path is not None:
                    made.append(path)
        return made

    return asyncio.run(all_clips())
