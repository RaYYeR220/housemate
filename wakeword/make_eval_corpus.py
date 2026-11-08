"""Build the held-out evaluation clips for the wake model with Microsoft Edge neural TTS.

None of these voices (or this TTS engine) is used in training, which is 100 % Piper/LibriTTS-R.
Needs `edge-tts`, `numpy` and an `ffmpeg` binary on PATH. Writes 16 kHz mono PCM16 WAVs + manifest.json.

  python make_eval_corpus.py --out <dir>

  <out>/positives/  "Hey Hermes" alone and followed by a command, every English Edge voice, varied rate/pitch
  <out>/nearmiss/   tricky near-phrases ("hey herman", "hair mess", "hermes" alone, ...), rotating voices
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

COMMANDS = ["what time is it?", "turn on the kitchen lights.", "set a timer for ten minutes.",
            "what's the weather like tomorrow?", "play some music.", "add milk to the shopping list."]
NEAR_MISSES = ["Hey Herman.", "Hey, hermit.", "Hair mess.", "Hey there, miss.", "Hermes.", "Hey, her mess.",
               "Hey Hermione!", "Her messages are on the table.", "I love my Hermes scarf.",
               "Hermes was the messenger of the gods.", "Hey, Herb, is that yours?", "They hear me, sir."]
RATES = ["-15%", "+0%", "+15%"]
PITCHES = ["-6Hz", "+0Hz", "+6Hz"]


def trim(pcm: np.ndarray, pad_ms: int = 150) -> np.ndarray:
    """Trim leading/trailing near-silence (20 ms frames below -50 dBFS), keep pad_ms each side."""
    f = 320
    n = len(pcm) // f
    fr = pcm[: n * f].astype(np.float32).reshape(n, f) / 32768.0
    on = np.where(np.sqrt((fr ** 2).mean(axis=1) + 1e-12) > 10 ** (-50 / 20))[0]
    if len(on) == 0:
        return pcm
    a, b = max(0, on[0] * f - pad_ms * 16), min(len(pcm), (on[-1] + 1) * f + pad_ms * 16)
    return pcm[a:b]


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.astype("<i2").tobytes())


async def synth(text: str, voice: str, rate: str, pitch: str, mp3: Path) -> np.ndarray | None:
    import edge_tts
    for attempt in range(6):
        try:
            await asyncio.wait_for(edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(mp3)), 60)
            break
        except Exception as e:  # NoAudioReceived / transient websocket errors
            print(f"  retry {voice} ({attempt + 1}): {type(e).__name__}")
            await asyncio.sleep(3 + 3 * attempt)
    else:
        return None
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return trim(np.frombuffer(raw, dtype="<i2"))


async def main() -> None:
    import edge_tts
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261108)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    voices = sorted(v["ShortName"] for v in await edge_tts.list_voices() if v["Locale"].startswith("en-"))
    jobs = []
    for v in voices:
        jobs.append(("positives", "short", "Hey Hermes.", v, RATES[rng.randint(3)], PITCHES[rng.randint(3)]))
        cmd = COMMANDS[rng.randint(len(COMMANDS))]
        jobs.append(("positives", "command", f"Hey Hermes, {cmd}", v, RATES[rng.randint(3)], PITCHES[rng.randint(3)]))
    for i, text in enumerate(NEAR_MISSES):
        for k in range(6):
            v = voices[(i * 7 + k * 5) % len(voices)]
            jobs.append(("nearmiss", f"nm{i:02d}", text, v, RATES[rng.randint(3)], "+0Hz"))
    manifest = []
    with tempfile.TemporaryDirectory() as td:
        for kind, slug, text, v, rate, pitch in jobs:
            name = f"{slug}_{v.replace('Neural', '').replace('-', '').lower()}.wav"
            p = a.out / kind / name
            if not p.exists():
                pcm = await synth(text, v, rate, pitch, Path(td) / "x.mp3")
                if pcm is None:
                    print(f"SKIP {kind} {name}: edge-tts gave no audio for {v}")
                    continue
                write_wav(p, pcm)
            manifest.append({"file": f"{kind}/{name}", "kind": kind, "text": text, "voice": v, "rate": rate,
                             "pitch": pitch, "engine": "edge-tts"})
            print(kind, name, text)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "clips")


if __name__ == "__main__":
    asyncio.run(main())
