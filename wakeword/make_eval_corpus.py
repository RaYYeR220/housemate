"""Build Microsoft Edge neural TTS clips for the wake model: the held-out test set, or extra training clips.

The 47 English Edge voices are split into disjoint TRAIN and TEST personas in `edge_voices.py`.

  python make_eval_corpus.py --split test  --out <dir>   # held-out evaluation set (TEST voices only)
  python make_eval_corpus.py --split train --out <dir>   # training clips (TRAIN voices only)

test:   <out>/positives/  per voice "Hey Hermes." / "Hey Hermes, <command>" / "Hey, Hermes." / "Hey Hermes!"
        <out>/nearmiss/   12 tricky near-phrases ("hey herman", "hair mess", "hermes" alone, ...), 6 voices each
train:  <out>/positives/  per voice a grid of punctuation x rate x pitch variants of "hey hermes"
        <out>/negatives/  the near-misses that fired in earlier runs ("hey, her mess", "hey herman", ...)

Needs `edge-tts`, `numpy` and an `ffmpeg` binary on PATH. Writes 16 kHz mono PCM16 WAVs + manifest.json.
Requests run a few at a time; existing files are skipped, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from edge_voices import TEST, TRAIN  # noqa: E402

COMMANDS = ["what time is it?", "turn on the kitchen lights.", "set a timer for ten minutes.",
            "what's the weather like tomorrow?", "play some music.", "add milk to the shopping list."]
NEAR_MISSES = ["Hey Herman.", "Hey, hermit.", "Hair mess.", "Hey there, miss.", "Hermes.", "Hey, her mess.",
               "Hey Hermione!", "Her messages are on the table.", "I love my Hermes scarf.",
               "Hermes was the messenger of the gods.", "Hey, Herb, is that yours?", "They hear me, sir."]
RATES = ["-15%", "+0%", "+15%"]
PITCHES = ["-6Hz", "+0Hz", "+6Hz"]

TRAIN_TEXTS = ["Hey Hermes", "Hey, Hermes.", "hey hermes!", "Hey Hermes.", "Hey Hermes?", "Hey Hermes,"]
TRAIN_RATES = ["-20%", "-10%", "+0%", "+10%", "+20%"]
TRAIN_PITCHES = ["-10Hz", "-5Hz", "+0Hz", "+5Hz", "+10Hz"]
TRAIN_NEGATIVES = ["Hey, her mess.", "Hey Herman.", "Hair mess.", "Hey, hermit.", "Hey, Herb."]


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
            print(f"  retry {voice} ({attempt + 1}): {type(e).__name__}", flush=True)
            await asyncio.sleep(1 + attempt)
    else:
        return None
    proc = await asyncio.to_thread(  # decode off the event loop so requests keep flowing
        subprocess.run, ["ffmpeg", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        check=True, capture_output=True)
    return trim(np.frombuffer(proc.stdout, dtype="<i2"))


def short(v: str) -> str:
    return v.replace("Neural", "").replace("-", "").lower()


def test_jobs(rng: np.random.RandomState) -> list[tuple]:
    jobs = []
    for v in TEST:
        jobs.append(("positives", f"short_{short(v)}", "Hey Hermes.", v, RATES[rng.randint(3)], PITCHES[rng.randint(3)]))
        cmd = COMMANDS[rng.randint(len(COMMANDS))]
        jobs.append(("positives", f"command_{short(v)}", f"Hey Hermes, {cmd}", v, RATES[rng.randint(3)],
                     PITCHES[rng.randint(3)]))
        jobs.append(("positives", f"comma_{short(v)}", "Hey, Hermes.", v, RATES[rng.randint(3)], PITCHES[rng.randint(3)]))
        jobs.append(("positives", f"excl_{short(v)}", "Hey Hermes!", v, RATES[rng.randint(3)], PITCHES[rng.randint(3)]))
    for i, text in enumerate(NEAR_MISSES):
        for k in range(6):
            v = TEST[(i * 7 + k * 5) % len(TEST)]
            jobs.append(("nearmiss", f"nm{i:02d}_{short(v)}", text, v, RATES[rng.randint(3)], "+0Hz"))
    return jobs


def train_jobs(rng: np.random.RandomState, per_voice: int, neg_per_voice: int) -> list[tuple]:
    jobs = []
    grid = [(t, r, p) for t in TRAIN_TEXTS for r in TRAIN_RATES for p in TRAIN_PITCHES]
    for v in TRAIN:
        for j in rng.choice(len(grid), size=min(per_voice, len(grid)), replace=False):
            t, r, p = grid[j]
            jobs.append(("positives", f"e_{short(v)}_{j:03d}", t, v, r, p))
        ngrid = [(t, r) for t in TRAIN_NEGATIVES for r in TRAIN_RATES]
        for j in rng.choice(len(ngrid), size=min(neg_per_voice, len(ngrid)), replace=False):
            t, r = ngrid[j]
            jobs.append(("negatives", f"e_{short(v)}_n{j:02d}", t, v, r, PITCHES[rng.randint(3)]))
    return jobs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261108)
    ap.add_argument("--per-voice", type=int, default=100, help="train: positives per voice")
    ap.add_argument("--neg-per-voice", type=int, default=20, help="train: near-miss negatives per voice")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    rng = np.random.RandomState(a.seed)
    jobs = test_jobs(rng) if a.split == "test" else train_jobs(rng, a.per_voice, a.neg_per_voice)
    order = np.random.RandomState(a.seed + 1).permutation(len(jobs))  # interleave voices: one flaky voice
    jobs = [jobs[i] for i in order]                                    # must not hog every request slot
    sem = asyncio.Semaphore(a.concurrency)
    manifest: list[dict] = []
    done = 0

    async def run(job: tuple, td: Path, idx: int) -> None:
        nonlocal done
        kind, name, text, v, rate, pitch = job
        p = a.out / kind / f"{name}.wav"
        if not p.exists():
            async with sem:
                pcm = await synth(text, v, rate, pitch, td / f"{idx}.mp3")
            if pcm is None:
                print(f"SKIP {kind} {name}: edge-tts gave no audio for {v}", flush=True)
                return
            write_wav(p, pcm)
        manifest.append({"file": f"{kind}/{p.name}", "kind": kind, "text": text, "voice": v, "rate": rate,
                         "pitch": pitch, "engine": "edge-tts", "split": a.split})
        done += 1
        if done % 50 == 0:
            print(f"{done}/{len(jobs)}", flush=True)

    with tempfile.TemporaryDirectory() as td:
        await asyncio.gather(*(run(j, Path(td), i) for i, j in enumerate(jobs)))
    manifest.sort(key=lambda m: m["file"])
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(len(manifest), "of", len(jobs), "clips")


if __name__ == "__main__":
    asyncio.run(main())
