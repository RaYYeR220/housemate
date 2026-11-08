"""Download and convert the training/evaluation audio for the "Hey Hermes" wake model.

Runs inside the training container (see Dockerfile / README.md). Everything lands under --data
(default /data). Each step is idempotent: finished outputs are skipped on re-runs.

  python /recipe/prepare_data.py rirs audioset fma speechbg features librispeech

Steps
  rirs         MIT environmental impulse responses, 16 kHz (HF davidscripka/MIT_environmental_impulse_responses)
  audioset     AudioSet balanced-train parquet shard(s) (HF agkphysics/AudioSet) -> 16 kHz mono WAV.
               The upstream notebook's bal_train09.tar no longer exists; the same data is now parquet.
  fma          Free Music Archive "small" clips, read member-by-member from fma_small.zip over HTTP ranges
               (what the notebook's rudraml/fma loader streams) -> 16 kHz mono WAV.
  speechbg     LibriSpeech dev-clean (OpenSLR 12) cut into 10 s chunks -> background *speech* for augmentation,
               so positives are also seen inside running speech (TV/podcast in the room). dev-clean speakers
               are disjoint from the test-clean/test-other speakers used for evaluation.
  features     openWakeWord pre-computed features: ACAV100M (~2,000 h, 17.3 GB) + false-positive validation set.
  librispeech  LibriSpeech test-clean + test-other (OpenSLR 12), concatenated per chapter -> 16 kHz WAV.
               Evaluation-only negatives (never used for training).

A deterministic hash split keeps ~15 % of the AudioSet/FMA clips out of training (written under eval/) so the
music/noise part of the false-accept evaluation is unseen.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import wave
from pathlib import Path

import numpy as np

SR = 16000


def is_eval(key: str, frac: float = 0.15) -> bool:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(x, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())


def ffmpeg_decode(data: bytes) -> np.ndarray:
    """Any audio container/codec -> float32 mono 16 kHz via ffmpeg."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
                       input=data, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype="<i2").astype(np.float32) / 32768.0


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    subprocess.run(["curl", "-fL", "--retry", "8", "--retry-delay", "5", "-C", "-", "-o", str(tmp), url], check=True)
    tmp.rename(dest)
    return dest


def step_rirs(data: Path) -> None:
    from huggingface_hub import snapshot_download
    out = data / "mit_rirs"
    if out.exists() and len(list(out.glob("*.wav"))) >= 270:
        print("rirs: done")
        return
    snap = Path(snapshot_download("davidscripka/MIT_environmental_impulse_responses", repo_type="dataset",
                                  allow_patterns=["16khz/*.wav"], local_dir=str(data / "_hf" / "mit_rirs")))
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((snap / "16khz").glob("*.wav")):
        (out / f.name).write_bytes(f.read_bytes())
    print("rirs:", len(list(out.glob("*.wav"))), "files")


def step_audioset(data: Path, shards: list[str]) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    out_tr, out_ev = data / "audioset_16k", data / "eval" / "audioset_16k"
    for shard in shards:
        marker = data / "_done" / f"audioset_{shard}"
        if marker.exists():
            print(f"audioset {shard}: done")
            continue
        path = hf_hub_download("agkphysics/AudioSet", f"data/bal_train/{shard}.parquet", repo_type="dataset",
                               local_dir=str(data / "_hf" / "audioset"))
        pf = pq.ParquetFile(path)
        n = 0
        for batch in pf.iter_batches(batch_size=64, columns=["video_id", "audio"]):
            for vid, audio in zip(batch.column("video_id").to_pylist(), batch.column("audio").to_pylist()):
                try:
                    x = ffmpeg_decode(audio["bytes"])
                except subprocess.CalledProcessError:
                    continue
                if x.size < SR:  # < 1 s, useless as background
                    continue
                write_wav((out_ev if is_eval(vid) else out_tr) / f"{vid}.wav", x)
                n += 1
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        print(f"audioset {shard}: {n} clips")


def step_fma(data: Path, n_clips: int) -> None:
    import zipfile
    import fsspec
    out_tr, out_ev = data / "fma", data / "eval" / "fma_16k"
    marker = data / "_done" / f"fma_{n_clips}"
    if marker.exists():
        print("fma: done")
        return
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    with fsspec.open(url, "rb", block_size=2 ** 20) as f, zipfile.ZipFile(f) as z:
        members = sorted(m for m in z.namelist() if m.endswith(".mp3"))
        rng = np.random.RandomState(1234)
        pick = [members[i] for i in rng.choice(len(members), size=min(n_clips, len(members)), replace=False)]
        n = 0
        for m in pick:
            name = Path(m).stem
            dest = (out_ev if is_eval(name) else out_tr) / f"{name}.wav"
            if dest.exists():
                n += 1
                continue
            try:
                x = ffmpeg_decode(z.read(m))
            except subprocess.CalledProcessError:
                continue
            if x.size < SR:
                continue
            write_wav(dest, x)
            n += 1
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print("fma:", n, "clips")


def step_features(data: Path) -> None:
    base = "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
    for name in ("validation_set_features.npy", "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"):
        download(base + name, data / "features" / name)
    print("features: done")


def step_librispeech(data: Path, gap_s: float = 0.3) -> None:
    import soundfile as sf
    out = data / "eval" / "librispeech"
    for part in ("test-clean", "test-other"):
        marker = data / "_done" / f"librispeech_{part}"
        if marker.exists():
            print(f"librispeech {part}: done")
            continue
        tgz = download(f"https://www.openslr.org/resources/12/{part}.tar.gz", data / "_dl" / f"{part}.tar.gz")
        chapters: dict[str, list[tuple[str, bytes]]] = {}
        texts: dict[str, str] = {}
        with tarfile.open(tgz, "r:gz") as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(".flac"):
                    spk, chap = m.name.split("/")[-3:-1]
                    chapters.setdefault(f"{spk}-{chap}", []).append((m.name, tar.extractfile(m).read()))
                elif m.isfile() and m.name.endswith(".trans.txt"):
                    for line in tar.extractfile(m).read().decode().splitlines():
                        uid, _, txt = line.partition(" ")
                        texts[uid] = txt
        gap = np.zeros(int(gap_s * SR), dtype=np.float32)
        for key, items in sorted(chapters.items()):
            parts, index, t = [], [], 0
            for name, b in sorted(items):
                x, sr = sf.read(io.BytesIO(b), dtype="float32")
                assert sr == SR
                uid = Path(name).stem
                index.append({"id": uid, "start": t / SR, "end": (t + len(x)) / SR, "text": texts.get(uid, "")})
                parts += [x, gap]
                t += len(x) + len(gap)
            write_wav(out / f"{part}_{key}.wav", np.concatenate(parts))
            (out / f"{part}_{key}.json").write_text(json.dumps(index))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        print(f"librispeech {part}: {len(chapters)} chapters")


def step_speechbg(data: Path, n_chunks: int, chunk_s: float = 10.0) -> None:
    """LibriSpeech dev-clean -> 10 s chunks of continuous read speech (one chapter at a time, evenly sampled)."""
    import soundfile as sf
    out = data / "speech_bg_16k"
    marker = data / "_done" / f"speechbg_{n_chunks}"
    if marker.exists():
        print("speechbg: done")
        return
    tgz = download("https://www.openslr.org/resources/12/dev-clean.tar.gz", data / "_dl" / "dev-clean.tar.gz")
    chapters: dict[str, list[tuple[str, bytes]]] = {}
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".flac"):
                spk, chap = m.name.split("/")[-3:-1]
                chapters.setdefault(f"{spk}-{chap}", []).append((m.name, tar.extractfile(m).read()))
    chunks: list[tuple[str, np.ndarray]] = []
    n = int(chunk_s * SR)
    for key, items in sorted(chapters.items()):
        x = np.concatenate([sf.read(io.BytesIO(b), dtype="float32")[0] for _, b in sorted(items)])
        chunks += [(f"{key}_{i:03d}", x[i * n:(i + 1) * n]) for i in range(len(x) // n)]
    step = max(1, len(chunks) // n_chunks)
    for name, x in chunks[::step][:n_chunks]:
        write_wav(out / f"devclean_{name}.wav", x)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    print(f"speechbg: {min(n_chunks, len(chunks[::step]))} chunks of {chunk_s:g} s from {len(chapters)} chapters")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="+", choices=["rirs", "audioset", "fma", "speechbg", "features", "librispeech"])
    ap.add_argument("--data", default="/data", type=Path)
    ap.add_argument("--audioset-shards", nargs="+", default=["09"])
    ap.add_argument("--fma-clips", type=int, default=160)
    ap.add_argument("--speechbg-chunks", type=int, default=400)
    a = ap.parse_args()
    for s in a.steps:
        if s == "rirs":
            step_rirs(a.data)
        elif s == "audioset":
            step_audioset(a.data, a.audioset_shards)
        elif s == "fma":
            step_fma(a.data, a.fma_clips)
        elif s == "speechbg":
            step_speechbg(a.data, a.speechbg_chunks)
        elif s == "features":
            step_features(a.data)
        elif s == "librispeech":
            step_librispeech(a.data)


if __name__ == "__main__":
    main()
