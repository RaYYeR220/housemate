"""Evaluate a wake head with Housemate's own runtime pipeline (core/wake.py), not the trainer's metrics.

Every clip/stream goes through `OnnxWakeDetector.feed()` in 80 ms chunks from a fresh `reset()`, exactly
as a satellite stream would. Needs only onnxruntime + numpy (the plugin's runtime deps).

  python wakeword/eval_wake.py --head wakeword/hey_hermes.onnx --models-dir <dir with mel+embedding .onnx> \
      --positives <dir> [<dir> ...] --nearmiss <dir> [...] \
      --speech <dir of long wavs> --music-noise <dir> [...] --bg-speech <dir> --out <report.json>

Positives: each clip gets 1.5 s of lead-in and 1.0 s of tail, either over a -65 dBFS noise floor ("clean")
or inside continuous background speech at a random 15-20 dB SNR ("speech_bg"). A clip counts as detected
if any chunk ending between clip start and clip end + 1 s scores >= threshold. Detection latency is measured
on the plain "Hey Hermes." clips: first chunk >= threshold minus the end of speech in the clip.

Negatives: long streams scored continuously; a false accept is a crossing >= threshold, after which 2 s are
ignored (the runtime would be listening for a command then). Near-misses are scored like clean positives
and reported as a count. Thresholds are swept post hoc from the stored per-chunk scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import wave
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SR = 16000
CHUNK = 1280
LEAD_S, TAIL_S = 1.5, 1.0
REFRACTORY_S = 2.0
FA_BAR = 0.5  # false accepts per hour
THRESHOLDS = sorted({round(float(t), 3) for t in list(np.arange(0.05, 0.951, 0.05)) + [0.97, 0.98, 0.99, 0.995]})

_det = None


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR):
            raise ValueError(f"{path}: need 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def seed_of(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def active_rms(x: np.ndarray) -> float:
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    r = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    act = r[r > 10 ** (-45 / 20)]
    return float(np.sqrt((act ** 2).mean())) if act.size else float(r.mean())


def speech_end_s(x: np.ndarray) -> float:
    f = x[: len(x) // 320 * 320].reshape(-1, 320)
    on = np.where(np.sqrt((f ** 2).mean(axis=1) + 1e-12) > 10 ** (-45 / 20))[0]
    return (on[-1] + 1) * 320 / SR if on.size else len(x) / SR


def _init(mel: str, emb: str, head: str) -> None:
    global _det
    from core.wake import OnnxWakeDetector
    _det = OnnxWakeDetector(mel, emb, head)


def stream_scores(x: np.ndarray) -> np.ndarray:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    pcm = np.concatenate([pcm, np.zeros((-len(pcm)) % CHUNK, dtype="<i2")])
    _det.reset()
    b = pcm.tobytes()
    step = CHUNK * 2
    return np.array([_det.feed(b[i:i + step]) for i in range(0, len(b), step)], dtype=np.float32)


def job_clip(args: tuple) -> dict:
    """Positive or near-miss clip in one condition."""
    path, condition, bg_files, snr_lo, snr_hi = args
    clip = read_wav(Path(path))
    rng = np.random.RandomState(seed_of(path + condition))
    n_lead, n_tail = int(LEAD_S * SR), int(TAIL_S * SR)
    n = n_lead + len(clip) + n_tail
    fg = np.zeros(n, dtype=np.float32)
    fg[n_lead:n_lead + len(clip)] = clip
    snr = None
    if condition == "clean":
        sig = fg + rng.normal(0, 10 ** (-65 / 20), n).astype(np.float32)
    else:  # speech_bg
        bg_path = bg_files[rng.randint(len(bg_files))]
        bg = read_wav(Path(bg_path))
        off = rng.randint(0, max(1, len(bg) - n))
        bg = bg[off:off + n]
        if len(bg) < n:
            bg = np.pad(bg, (0, n - len(bg)))
        snr = float(rng.uniform(snr_lo, snr_hi))
        bg = bg * (active_rms(clip) / max(active_rms(bg), 1e-6)) / 10 ** (snr / 20)
        sig = fg + bg
    s = stream_scores(sig)
    t_end = (np.arange(len(s)) + 1) * CHUNK / SR
    win = (t_end >= LEAD_S) & (t_end <= LEAD_S + len(clip) / SR + TAIL_S)
    return {"file": path, "condition": condition, "snr_db": snr, "max": float(s[win].max()),
            "pre_max": float(s[t_end < LEAD_S].max()), "speech_end_s": LEAD_S + speech_end_s(clip),
            "win_t": t_end[win].tolist(), "win_s": s[win].tolist()}


def job_stream(path: str) -> dict:
    p = Path(path)
    if p.is_dir():  # many short clips (AudioSet/FMA): one continuous stream, no resets in between
        x = np.concatenate([read_wav(f) for f in sorted(p.glob("*.wav"))])
    else:
        x = read_wav(p)
    s = stream_scores(x)
    return {"file": path, "seconds": len(x) / SR, "scores": s}


def events(scores: np.ndarray, thr: float) -> list[int]:
    out, block_until = [], -1
    ref = int(REFRACTORY_S * SR / CHUNK)
    for i in np.flatnonzero(scores >= thr):
        if i > block_until:
            out.append(int(i))
            block_until = i + ref
    return out


def transcript_at(path: str, t: float) -> str:
    j = Path(path).with_suffix(".json")
    if not j.exists():
        return ""
    for u in json.loads(j.read_text()):
        if u["start"] - 0.5 <= t <= u["end"] + 0.5:
            return u["text"]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--models-dir", required=True, type=Path)
    ap.add_argument("--positives", nargs="+", default=[])
    ap.add_argument("--nearmiss", nargs="+", default=[])
    ap.add_argument("--speech", nargs="+", default=[], help="dirs of long speech WAVs (one stream per file)")
    ap.add_argument("--music-noise", nargs="+", default=[], help="dirs of short clips (one stream per dir)")
    ap.add_argument("--bg-speech", help="dir of speech WAVs to mix under positives (default: first --speech)")
    ap.add_argument("--snr", nargs=2, type=float, default=[15.0, 20.0])
    ap.add_argument("--threshold", type=float, help="operating threshold (default: best recall with FA/h <= 0.5)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    from core.models import MODELS, sha256_file
    mel, emb = a.models_dir / MODELS["melspectrogram"].filename, a.models_dir / MODELS["embedding"].filename
    for spec, path in ((MODELS["melspectrogram"], mel), (MODELS["embedding"], emb)):
        if sha256_file(path) != spec.sha256:
            raise SystemExit(f"{path}: sha256 mismatch with the runtime pin")

    pos = sorted(str(f) for d in a.positives for f in Path(d).glob("*.wav"))
    nm = sorted(str(f) for d in a.nearmiss for f in Path(d).glob("*.wav"))
    speech = sorted(str(f) for d in a.speech for f in Path(d).glob("*.wav"))
    music = [str(Path(d)) for d in a.music_noise]
    bg = sorted(str(f) for f in Path(a.bg_speech or a.speech[0]).glob("*.wav"))
    lo, hi = a.snr

    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(str(mel), str(emb), a.head)) as pool:
        clips = pool.map(job_clip, [(p, c, bg, lo, hi) for p in pos for c in ("clean", "speech_bg")]
                         + [(p, "clean", bg, lo, hi) for p in nm], chunksize=4)
        streams = pool.map(job_stream, speech + music, chunksize=1)
    wall = time.time() - t0
    P = [c for c in clips if c["file"] in set(pos)]
    N = [c for c in clips if c["file"] in set(nm)]
    S_speech = [s for s in streams if s["file"] in set(speech)]
    S_music = [s for s in streams if s["file"] in set(music)]
    h_speech = sum(s["seconds"] for s in S_speech) / 3600
    h_music = sum(s["seconds"] for s in S_music) / 3600

    def is_short(c: dict) -> bool:
        name = Path(c["file"]).name
        return name.startswith("short_") or name.startswith("hermes_hermes_")

    def at(thr: float) -> dict:
        r = {"threshold": thr}
        for cond in ("clean", "speech_bg"):
            cs = [c for c in P if c["condition"] == cond]
            r[f"recall_{cond}"] = sum(c["max"] >= thr for c in cs) / max(1, len(cs))
        r["recall_all"] = sum(c["max"] >= thr for c in P) / max(1, len(P))
        fa_s = sum(len(events(s["scores"], thr)) for s in S_speech)
        fa_m = sum(len(events(s["scores"], thr)) for s in S_music)
        r["fa_speech"], r["fa_music_noise"] = fa_s, fa_m
        r["fa_per_hour"] = (fa_s + fa_m) / max(1e-9, h_speech + h_music)
        r["fa_per_hour_speech"] = fa_s / max(1e-9, h_speech)
        r["nearmiss_fires"] = sum(c["max"] >= thr for c in N)
        return r

    sweep = [at(t) for t in THRESHOLDS]
    if a.threshold is not None:
        chosen = a.threshold
    else:
        ok = [r for r in sweep if r["fa_per_hour"] <= FA_BAR]
        best = max((r["recall_all"] for r in ok), default=None)
        chosen = max(r["threshold"] for r in ok if r["recall_all"] == best) if ok else max(THRESHOLDS)

    def detail(thr: float) -> dict:
        d = at(thr)
        lat = []
        for c in P:
            if not is_short(c):
                continue
            hit = [t for t, s in zip(c["win_t"], c["win_s"]) if s >= thr]
            if hit:
                lat.append(1000 * (hit[0] - c["speech_end_s"]))
        d["latency_ms"] = ({"n": len(lat), "p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                            "min": float(min(lat)), "max": float(max(lat))} if lat else None)
        d["misses"] = [{"file": Path(c["file"]).name, "condition": c["condition"], "snr_db": c["snr_db"],
                        "max": round(c["max"], 4)} for c in P if c["max"] < thr]
        d["false_accepts"] = []
        for s in S_speech + S_music:
            for i in events(s["scores"], thr):
                t = (i + 1) * CHUNK / SR
                d["false_accepts"].append({"file": Path(s["file"]).name, "t_s": round(t, 2),
                                           "score": round(float(s["scores"][i]), 4),
                                           "transcript": transcript_at(s["file"], t)})
        d["nearmiss_fired"] = [Path(c["file"]).name for c in N if c["max"] >= thr]
        return d

    from collections import defaultdict
    nm_by_text: dict[str, list[float]] = defaultdict(list)
    man = {}
    for d in a.nearmiss:
        mf = Path(d).parent / "manifest.json"
        if mf.exists():
            man.update({Path(m["file"]).name: m["text"] for m in json.loads(mf.read_text())})
    for c in N:
        nm_by_text[man.get(Path(c["file"]).name, Path(c["file"]).stem)].append(round(c["max"], 4))

    report = {
        "head": str(a.head), "head_sha256": sha256_file(a.head), "runtime": "core/wake.py OnnxWakeDetector (80 ms chunks)",
        "positives": {"clips": len(pos), "conditions": ["clean", f"speech_bg {lo:g}-{hi:g} dB SNR"], "trials": len(P)},
        "nearmiss": {"clips": len(nm), "max_score_by_phrase": dict(sorted(nm_by_text.items()))},
        "negatives": {"speech_hours": round(h_speech, 3), "music_noise_hours": round(h_music, 3),
                      "speech_files": len(S_speech), "music_noise_streams": len(S_music),
                      "max_score_speech": float(max((s["scores"].max() for s in S_speech), default=0)),
                      "max_score_music_noise": float(max((s["scores"].max() for s in S_music), default=0))},
        "refractory_s": REFRACTORY_S, "bar": {"recall": 0.9, "fa_per_hour": FA_BAR},
        "sweep": sweep, "at_0.5": detail(0.5), "chosen_threshold": chosen, "at_chosen": detail(chosen),
        "eval_wall_s": round(wall, 1),
    }
    c = report["at_chosen"]
    report["pass"] = bool(c["recall_all"] >= 0.9 and c["fa_per_hour"] <= FA_BAR)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    np.savez_compressed(a.out.with_suffix(".scores.npz"), **{Path(s["file"]).name: s["scores"] for s in streams})
    for k in ("at_0.5", "at_chosen"):
        r = report[k]
        print(f"{k}: thr={r['threshold']} recall_all={r['recall_all']:.3f} clean={r['recall_clean']:.3f} "
              f"speech_bg={r['recall_speech_bg']:.3f} FA/h={r['fa_per_hour']:.3f} (speech {r['fa_speech']}, "
              f"music/noise {r['fa_music_noise']}) nearmiss_fires={r['nearmiss_fires']}/{len(N)} latency={r['latency_ms']}")
    print(f"negatives: {h_speech:.2f} h speech + {h_music:.2f} h music/noise; pass={report['pass']}; wall {wall:.0f}s")


if __name__ == "__main__":
    main()
