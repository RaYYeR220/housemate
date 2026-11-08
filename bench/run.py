"""``hermes housemate bench``: measure Housemate and write ``results/*.json``.

Parts (default: ``wake latency``):

- ``wake``: recall on the positive clips (clean, and under speech 10 dB down), false wakes on
  near-miss phrases, and false accepts per hour on 9.2 h of public-domain speech. Same wake model,
  threshold, re-arm and 1 s refractory rule as the hub's listener.
- ``latency``: a scripted satellite joins a room on the running hub and speaks a wake-word
  question ``--runs`` times (after one warm-up run). Measured: end of speech (the last speech
  frame by Silero VAD, as a satellite would send it) -> first reply audio frame received by the
  satellite; p50/p95 plus the hub's stage times for each turn (control op ``timings``).
- ``tests``: runs the test suite with each ``--python`` and records the counts.

Each JSON records the date, Hermes version, the STT/TTS/LLM settings in use (never keys) and the
hardware. ``scripts/render_numbers.py`` copies the numbers into README.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO
from urllib.parse import urlparse

if __package__ in (None, ""):  # python bench/run.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "housemate.bench"

from . import corpus

ROOT = corpus.ROOT
RESULTS = ROOT / "results"
FRAME_BYTES = 640  # 20 ms of PCM16 @ 16 kHz
FRAME_S = 0.02
CHUNK_BYTES = 1280 * 2  # one wake score per 80 ms
PARTS = ("wake", "latency", "tests")


# --- small pure helpers (unit-tested) ---------------------------------------------------------


class FalseAcceptCounter:
    """The listener's trigger rule: a score at/above the threshold fires once, then the detector
    re-arms only after the score falls below it again, and never within ``refractory_s``."""

    def __init__(self, threshold: float, refractory_s: float = 1.0) -> None:
        self.threshold = threshold
        self.refractory_s = refractory_s
        self.armed = True
        self.until = -math.inf
        self.count = 0
        self.events: list[float] = []

    def feed(self, score: float, t: float) -> bool:
        if score < self.threshold:
            self.armed = True
            return False
        if not self.armed or t < self.until:
            return False
        self.armed = False
        self.until = t + self.refractory_s
        self.count += 1
        self.events.append(round(t, 2))
        return True


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (no interpolation)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    rank = max(1, math.ceil(p / 100 * len(vals)))
    return vals[rank - 1]


def match_turn(turns: list[dict], speech_end: float, *, before: float = 1.0, after: float = 5.0) -> dict | None:
    """The hub's turn record for speech that ended at ``speech_end`` (wall clock)."""
    near = [t for t in turns if speech_end - before <= float(t.get("endpoint_at", 0)) <= speech_end + after]
    return min(near, key=lambda t: abs(float(t["endpoint_at"]) - speech_end)) if near else None


def breakdown(turn: dict, speech_end: float, first_pcm_at: float | None) -> dict[str, float]:
    """Every stage of a turn in seconds after the end of speech, plus the satellite's first audio."""
    base = float(turn["endpoint_at"])
    out = {k: round(base + float(v) - speech_end, 3) for k, v in (turn.get("stages") or {}).items()}
    if first_pcm_at is not None:
        out["satellite_first_pcm"] = round(first_pcm_at - speech_end, 3)
    return out


def speech_end_offset(pcm: bytes, vad: Any, threshold: float = 0.5) -> float:
    """Seconds from the start of ``pcm`` to the end of its last speech frame."""
    last = -1
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        if vad.speech_prob(pcm[i : i + FRAME_BYTES]) >= threshold:
            last = i // FRAME_BYTES
    return (last + 1) * FRAME_S


def _host(url: Any) -> str | None:
    try:
        return urlparse(str(url)).hostname if url else None
    except ValueError:
        return None


def settings_from_config(cfg: dict) -> dict:
    """The speech and model settings a number depends on. Never copies keys or full commands."""
    cfg = cfg if isinstance(cfg, dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    llm = {"provider": model.get("provider"), "model": model.get("default") or model.get("model"),
           "host": _host(model.get("base_url"))}
    stt_cfg = cfg.get("stt") if isinstance(cfg.get("stt"), dict) else {}
    stt_provider = stt_cfg.get("provider")
    stt: dict[str, Any] = {"provider": stt_provider, "language": stt_cfg.get("language")}
    section = stt_cfg.get(stt_provider) if isinstance(stt_cfg.get(stt_provider), dict) else {}
    for key in ("model", "device", "compute_type", "vad"):
        if key in section:
            stt[key] = section[key]
    tts_cfg = cfg.get("tts") if isinstance(cfg.get("tts"), dict) else {}
    tts_provider = tts_cfg.get("provider")
    tts: dict[str, Any] = {"provider": tts_provider}
    section = tts_cfg.get(tts_provider) if isinstance(tts_cfg.get(tts_provider), dict) else {}
    named = (tts_cfg.get("providers") or {}).get(tts_provider) if isinstance(tts_cfg.get("providers"), dict) else None
    if isinstance(named, dict) and named.get("type") == "command":
        command = str(named.get("command") or "").split()
        tts.update({"type": "command", "program": Path(command[0]).name if command else None})
    for key in ("model", "voice", "use_cuda", "pcm_sample_rate"):
        if key in section:
            tts[key] = section[key]
    if section.get("base_url"):
        tts["host"] = _host(section["base_url"])
    if tts_provider == "housemate-piper" and "voice" not in tts:
        from ..core.tts_piper import PiperEngine

        tts["voice"] = PiperEngine(config=lambda: cfg).voice_name()
    if isinstance(tts_cfg.get("streaming"), dict):
        tts["streaming"] = {k: v for k, v in tts_cfg["streaming"].items() if k in ("provider", "min_len")}
    platforms = cfg.get("platforms") if isinstance(cfg.get("platforms"), dict) else {}
    hm = platforms.get("housemate") if isinstance(platforms.get("housemate"), dict) else {}
    hm_extra = hm.get("extra") if isinstance(hm.get("extra"), dict) else {}
    from ..core.config import HousemateConfig

    hcfg = HousemateConfig.from_extra({**hm_extra, **hm})
    housemate = {"end_silence_ms": hcfg.end_silence_ms, "wake_model": hcfg.wake_model,
                 "wake_threshold": hcfg.wake_threshold, "vad": hcfg.vad, "conversation": hcfg.conversation}
    return {"llm": llm, "stt": stt, "tts": tts, "housemate": housemate}


# --- environment --------------------------------------------------------------------------------


def _run_text(cmd: list[str], timeout: float = 20.0) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hermes_version() -> str | None:
    """``hermes --version`` of the Hermes running this (a source checkout's package says 0.0.0)."""
    candidates = []
    if Path(sys.argv[0]).name.lower().startswith("hermes"):
        candidates.append(sys.argv[0])
    candidates += [str(Path(sys.executable).with_name(n)) for n in ("hermes.exe", "hermes")]
    candidates.append(shutil.which("hermes"))
    for exe in candidates:
        if exe and Path(exe).is_file():
            m = re.search(r"v(\d+\.\d+\.\d+)", _run_text([exe, "--version"], timeout=60))
            if m:
                return m.group(1)
    return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = _run_text([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def load_hermes_config() -> dict:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def environment(cfg: dict | None = None) -> dict:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hermes_version": hermes_version(),
        "python": platform.python_version(),
        "settings": settings_from_config(load_hermes_config() if cfg is None else cfg),
        "hardware": {"os": platform.platform(), "cpu": cpu_name(), "cores": os.cpu_count(), "gpu": gpu_name()},
    }


# --- wake ---------------------------------------------------------------------------------------


def _max_score(detector: Any, pcm: bytes) -> float:
    detector.reset()
    best = 0.0
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        score = detector.feed(pcm[i : i + FRAME_BYTES])
        if score is not None:
            best = max(best, float(score))
    return best


def _mix(speech: bytes, background: bytes, snr_db: float, pad_s: float = 1.0) -> bytes:
    """``speech`` with ``pad_s`` of lead-in, over ``background`` scaled to ``snr_db`` below it."""
    import numpy as np

    pad = int(16000 * pad_s)
    s = np.concatenate([np.zeros(pad), np.frombuffer(speech, "<i2").astype(np.float64), np.zeros(pad // 2)])
    b = np.frombuffer(background, "<i2").astype(np.float64)[: len(s)]
    if len(b) < len(s):
        b = np.pad(b, (0, len(s) - len(b)))
    rms = lambda x: float(np.sqrt(np.mean(x[np.abs(x) > 0] ** 2))) if np.any(x) else 0.0  # noqa: E731
    s_rms, b_rms = rms(s), rms(b)
    if b_rms > 0 and s_rms > 0:
        b *= s_rms / b_rms / (10 ** (snr_db / 20))
    return np.clip(s + b, -32768, 32767).astype("<i2").tobytes()


def _scan_negative(path: Path, models: Any, threshold: float, refractory_s: float) -> dict:
    detector = models.new_wake()
    counter = FalseAcceptCounter(threshold, refractory_s)
    pending = bytearray()
    chunks = 0
    for block in corpus.decode_16k(path):
        pending += block
        while len(pending) >= CHUNK_BYTES:
            score = detector.feed(bytes(pending[:CHUNK_BYTES]))
            del pending[:CHUNK_BYTES]
            chunks += 1
            if score is not None:
                counter.feed(float(score), chunks * 0.08)
    return {"file": path.name, "seconds": round(chunks * 0.08, 1), "false_accepts": counter.count, "at_s": counter.events}


def wake_bench(data_dir: Path, cache_dir: Path, *, wake_model: str, threshold: float, max_hours: float | None,
               workers: int, generate: bool, extra_positives: list[Path], refractory_s: float = 1.0,
               log: Callable[[str], None] = print) -> dict:
    from ..core.models import load_audio_models

    started = time.monotonic()
    models = load_audio_models(data_dir, wake_model=wake_model, vad="energy")
    if generate:
        corpus.make_positives(cache_dir, log)
    positives = corpus.positive_files(cache_dir, extra_positives)
    near_misses = sorted(corpus.FIXTURES.glob("neg_*.wav"))
    negatives = corpus.fetch_negatives(cache_dir, corpus.pick_negatives(max_hours), log)
    log(f"wake: {len(positives)} positives, {len(near_misses)} near misses, {len(negatives)} negative recordings")

    background = b"".join(corpus.decode_16k(negatives[0], seconds=180))  # speech to mix under the positives
    detector = models.new_wake()
    rows = []
    for i, path in enumerate(positives):
        pcm = corpus.read_wav16k(path)
        clean = _max_score(detector, b"\x00" * 32000 + pcm + b"\x00" * 16000)
        room_frames = max(1, (len(background) - len(pcm) - 64000) // FRAME_BYTES)
        offset = (i * 7919 % room_frames) * FRAME_BYTES  # a different stretch of speech under each clip
        noisy = _max_score(detector, _mix(pcm, background[offset:], 10.0))
        rows.append({"file": path.name, "clean": round(clean, 4), "speech_10db": round(noisy, 4)})
    misses = [{"file": p.name, "score": round(_max_score(detector, b"\x00" * 32000 + corpus.read_wav16k(p)), 4)}
              for p in near_misses]

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        scans = list(pool.map(lambda p: _scan_negative(p, models, threshold, refractory_s), negatives))
    hours = sum(s["seconds"] for s in scans) / 3600
    false_accepts = sum(s["false_accepts"] for s in scans)

    def recall(key: str) -> dict:
        hit = sum(1 for r in rows if r[key] >= threshold)
        return {"n": len(rows), "detected": hit, "recall": round(hit / len(rows), 4) if rows else None}

    return {
        "wake_model": models.wake_model,
        "threshold": threshold,
        "refractory_s": refractory_s,
        "positives": {"clean": recall("clean"), "speech_10db": recall("speech_10db"),
                      "sources": "tests/fixtures/wake_*.wav (ElevenLabs + Edge voices) + Edge neural voices made by "
                                 "bench/corpus.py make_positives (see bench/corpus.md)",
                      "files": rows},
        "near_misses": {"n": len(misses), "fired": sum(1 for m in misses if m["score"] >= threshold), "files": misses},
        "negatives": {"source": corpus.NEGATIVE_ITEM, "licence": corpus.NEGATIVE_LICENCE, "files": len(scans),
                      "hours": round(hours, 2), "false_accepts": false_accepts,
                      "fa_per_hour": round(false_accepts / hours, 4) if hours else None,
                      "events": [{"file": s["file"], "at_s": t} for s in scans for t in s["at_s"]]},
        "seconds_to_run": round(time.monotonic() - started, 1),
    }


# --- latency ------------------------------------------------------------------------------------


class _Device:
    """A scripted satellite that records what it gets and reports ``playback.done`` like a real one."""

    def __init__(self, sat: Any) -> None:
        self.sat = sat
        self.events: list[tuple[float, dict]] = []
        self.streams: dict[str, dict] = {}
        self.current: str | None = None
        self._changed = asyncio.Event()
        self._tasks = [asyncio.create_task(self._pump_control()), asyncio.create_task(self._pump_pcm())]

    def _note(self, msg: dict) -> None:
        self.events.append((time.time(), msg))
        self._changed.set()

    async def _pump_control(self) -> None:
        while True:
            msg = await self.sat.recv_control(timeout=86400)
            if msg["type"] == "play.begin":
                self.streams[msg["stream_id"]] = {"rate": msg["sample_rate"], "bytes": 0, "first": None}
                self.current = msg["stream_id"]
            elif msg["type"] == "play.end":
                st = self.streams.get(msg["stream_id"])
                if st is not None:
                    asyncio.create_task(self._done_later(msg["stream_id"], st))
            self._note(msg)

    async def _pump_pcm(self) -> None:
        while True:
            pcm = await self.sat.recv_pcm(timeout=86400)
            st = self.streams.get(self.current or "")
            if st is not None:
                if st["first"] is None:
                    st["first"] = time.time()
                    self._note({"type": "_first_pcm", "stream_id": self.current})
                st["bytes"] += len(pcm)

    async def _done_later(self, sid: str, st: dict) -> None:
        await asyncio.sleep(0.05)
        until = (st["first"] or time.time()) + st["bytes"] / (2 * st["rate"])
        await asyncio.sleep(max(0.0, until - time.time()))
        await self.sat.send_control({"type": "playback.done", "stream_id": sid})
        self._note({"type": "_played", "stream_id": sid})

    async def wait_for(self, pred: Callable[[dict], bool], timeout: float, since: float) -> dict | None:
        deadline = time.time() + timeout
        while True:
            for t, msg in self.events:
                if t >= since and pred(msg):
                    return msg
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            self._changed.clear()
            try:
                await asyncio.wait_for(self._changed.wait(), remaining)
            except asyncio.TimeoutError:
                pass

    def captions(self, since: float, role: str) -> list[str]:
        return [m["text"] for t, m in self.events if t >= since and m["type"] == "caption" and m.get("role") == role]

    def state(self) -> str | None:
        states = [m["state"] for _, m in self.events if m["type"] == "state"]
        return states[-1] if states else None

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.sat.close()


async def _speak(sat: Any, pcm: bytes, speech_end_s: float, tail_s: float = 1.5) -> float:
    """Send ``pcm`` as real-time 20 ms frames plus a silent tail; returns the wall-clock time the
    frame holding the last speech was sent (= when a real satellite would have sent it)."""
    frames = [pcm[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00") for i in range(0, len(pcm), FRAME_BYTES)]
    frames += [b"\x00" * FRAME_BYTES] * int(tail_s / FRAME_S)
    last_speech = max(0, math.ceil(speech_end_s / FRAME_S) - 1)
    loop = asyncio.get_running_loop()
    start = loop.time()
    speech_end = None
    for i, frame in enumerate(frames):
        delay = start + i * FRAME_S - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        await sat.send_audio(frame)
        if i == last_speech:
            speech_end = time.time()
    return speech_end if speech_end is not None else time.time()


async def latency_bench(client: Any, *, data_dir: Path, cache_dir: Path, runs: int, warmup: int, room: str | None,
                        clips: list[Path], timeout: float = 90.0, log: Callable[[str], None] = print) -> dict:
    sys.path.insert(0, str(ROOT / "tests"))
    from fake_satellite import FakeSatellite

    from ..core.models import load_audio_models

    info = await asyncio.to_thread(client.info)
    if not info.get("ok"):
        raise RuntimeError(info.get("error") or "hub not reachable")
    hub = urlparse(info.get("hub_url") or "https://127.0.0.1:8742")
    url = f"{'wss' if hub.scheme == 'https' else 'ws'}://127.0.0.1:{hub.port or 8742}/ws"
    rooms = (await asyncio.to_thread(client.rooms)).get("rooms") or []
    if room is None:
        if not rooms:
            raise RuntimeError("no rooms on the hub: add one with hermes housemate rooms add Kitchen")
        room = rooms[0].get("name") or rooms[0].get("room")

    models = await asyncio.to_thread(load_audio_models, data_dir, vad="silero")
    audio = []
    for path in clips:
        pcm = corpus.read_wav16k(path)
        audio.append((path.name, pcm, speech_end_offset(pcm, models.new_vad())))

    key_file = Path(cache_dir) / "bench-device.key"
    key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else None
    sat = FakeSatellite(url, "housemate-bench", "Bench satellite", key or None)
    await sat.connect()
    while True:
        msg = await sat.recv_control(timeout=30)
        if msg["type"] == "pair.code":
            log(f"pairing the bench satellite (code {msg['code']})")
            res = await asyncio.to_thread(client.pair_approve, msg["code"])
            if not res.get("ok"):
                raise RuntimeError(f"pairing failed: {res.get('error')}")
        elif msg["type"] == "paired":
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_text(sat.device_key or "", encoding="utf-8")
            break
    dev = _Device(sat)
    await sat.send_control({"type": "room.set", "room": room})
    await asyncio.sleep(2.0)  # the hub attaches a listener
    log(f"latency: {warmup} warm-up + {runs} runs in {room}")

    rows = []
    try:
        for i in range(warmup + runs):
            name, pcm, end_s = audio[i % len(audio)]
            if dev.state() not in (None, "idle"):
                await dev.wait_for(lambda m: m["type"] == "state" and m["state"] == "idle", 12, since=time.time() - 0.01)
            await asyncio.sleep(1.0)
            t0 = time.time()
            speech_end = await _speak(sat, pcm, end_s)
            begin = await dev.wait_for(lambda m: m["type"] == "play.begin" and m["stream_id"].startswith("rep-"),
                                       timeout, since=t0)
            row: dict[str, Any] = {"run": i + 1, "warm_up": i < warmup, "clip": name}
            if begin is None:
                row["error"] = "no spoken reply"
                rows.append(row)
                log(f"  run {i + 1}: no spoken reply")
                continue
            sid = begin["stream_id"]
            await dev.wait_for(lambda m: m["type"] == "_played" and m["stream_id"] == sid, timeout, since=t0)
            first = dev.streams[sid]["first"]
            row["latency_s"] = round(first - speech_end, 3) if first else None
            row["transcript"] = (dev.captions(t0, "user") or [None])[-1]
            row["reply"] = (dev.captions(t0, "assistant") or [None])[-1]
            await asyncio.sleep(0.5)
            turns = (await asyncio.to_thread(client.timings)).get("turns") or []
            turn = match_turn(turns, speech_end)
            if turn is not None:
                row["stages"] = breakdown(turn, speech_end, first)
            rows.append(row)
            log(f"  run {i + 1}{' (warm-up)' if i < warmup else ''}: {row['latency_s']} s  "
                f"{row['transcript']!r} -> {row['reply']!r}")
    finally:
        await dev.close()

    warm = [r for r in rows if not r["warm_up"] and r.get("latency_s") is not None]
    lat = [r["latency_s"] for r in warm]

    def med(key: str) -> float | None:
        vals = [r["stages"][key] for r in warm if key in (r.get("stages") or {})]
        return round(statistics.median(vals), 3) if vals else None

    marks = ("endpoint", "captured", "dispatch", "transcript", "llm_start", "tts_begin", "first_pcm_out",
             "satellite_first_pcm")
    at = {k: med(k) for k in marks}

    def gap(a: str, b: str) -> float | None:
        vals = [r["stages"][b] - r["stages"][a] for r in warm if {a, b} <= set(r.get("stages") or {})]
        return round(statistics.median(vals), 3) if vals else None

    return {
        "metric": "end of speech (last speech frame by Silero VAD, sent in real time) -> first reply audio "
                  "frame received by the satellite, same machine as the hub",
        "room": room,
        "clips": [{"file": n, "speech_end_s": round(e, 2)} for n, _, e in audio],
        "warm_runs": len(lat),
        "p50_s": percentile(lat, 50),
        "p95_s": percentile(lat, 95),
        "min_s": min(lat) if lat else None,
        "max_s": max(lat) if lat else None,
        "mean_s": round(statistics.mean(lat), 3) if lat else None,
        "warm_up_s": [r.get("latency_s") for r in rows if r["warm_up"]],
        "stage_median_after_speech_end_s": at,
        "stage_median_durations_s": {
            "endpointing (silence wait)": at["captured"],
            "speech_to_text": gap("dispatch", "transcript"),
            "transcript_to_llm_request": gap("transcript", "llm_start"),
            "llm_and_tts_to_first_audio": gap("llm_start", "first_pcm_out"),
            "hub_to_satellite": gap("first_pcm_out", "satellite_first_pcm"),
        },
        "runs": rows,
    }


# --- tests --------------------------------------------------------------------------------------


def tests_bench(pythons: list[str], log: Callable[[str], None] = print) -> dict:
    runs = []
    for py in pythons:
        version = _run_text([py, "-c", "import platform; print(platform.python_version())"])
        real = _run_text([py, "-c", "import gateway.platforms.base; print('yes')"]) == "yes"
        log(f"tests: {py} (Python {version}, real Hermes: {real})")
        proc = subprocess.run([py, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"], cwd=ROOT,
                              capture_output=True, text=True, timeout=1800, stdin=subprocess.DEVNULL)
        tail = (proc.stdout.strip().splitlines() or [""])[-1]
        counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?)", tail)}
        runs.append({"python": version, "real_hermes": real, "command": "pytest -q -m 'not slow'",
                     "passed": counts.get("passed", 0), "failed": counts.get("failed", 0),
                     "skipped": counts.get("skipped", 0), "deselected": counts.get("deselected", 0),
                     "errors": counts.get("error", 0) + counts.get("errors", 0), "summary": tail})
    return {"runs": runs}


# --- entry points -------------------------------------------------------------------------------


def _write(out_dir: Path, name: str, data: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_bytes((json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("parts", nargs="*", metavar="part", help="wake, latency, tests (default: wake latency)")
    parser.add_argument("--runs", type=int, default=20, help="warm latency runs (default 20)")
    parser.add_argument("--warmup", type=int, default=1, help="latency runs not counted (default 1)")
    parser.add_argument("--room", default=None, help="room the bench satellite joins (default: the first room)")
    parser.add_argument("--clip", action="append", default=None, help="16 kHz mono WAV to speak (repeatable)")
    parser.add_argument("--max-hours", type=float, default=None, help="limit the negative speech (default: all 9.2 h)")
    parser.add_argument("--no-generate", action="store_true", help="don't make Edge-voice positives")
    parser.add_argument("--positives", action="append", default=[], help="extra folder of positive WAVs")
    parser.add_argument("--threshold", type=float, default=None, help="wake threshold (default: the configured one)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--python", action="append", default=None, help="interpreter for the tests part (repeatable)")
    parser.add_argument("--out", default=str(RESULTS), help="results folder (default: results/ in the plugin)")
    parser.add_argument("--data-dir", default=None, help="Housemate data folder (default: Hermes plugin data)")


def run(args: argparse.Namespace, client: Any = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout

    def log(msg: str) -> None:
        print(msg, file=out, flush=True)

    parts = list(args.parts or ["wake", "latency"])
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        log(f"unknown bench part(s): {', '.join(unknown)}; use {', '.join(PARTS)}")
        return 2
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        from plugins.plugin_storage import plugin_data_dir

        data_dir = Path(plugin_data_dir("housemate"))
    cache_dir = data_dir / "bench"
    out_dir = Path(args.out)
    env = environment()
    hm = env["settings"]["housemate"]
    code = 0
    if "latency" in parts:
        if client is None:
            from ..core.client import HousemateClient

            client = HousemateClient(data_dir)
        clips = [Path(c) for c in args.clip] if args.clip else [corpus.FIXTURES / "wake_jarvis-math_enusaria.wav",
                                                                 corpus.CLIPS / "three_times_three_enusguy.wav"]
        try:
            result = asyncio.run(latency_bench(client, data_dir=data_dir, cache_dir=cache_dir, runs=args.runs,
                                               warmup=args.warmup, room=args.room, clips=clips, log=log))
        except Exception as exc:
            log(f"latency bench failed: {exc}")
            code = 1
        else:
            path = _write(out_dir, "latency.json", {"environment": env, **result})
            log(f"latency: p50 {result['p50_s']} s, p95 {result['p95_s']} s over {result['warm_runs']} runs -> {path}")
    if "wake" in parts:
        threshold = args.threshold if args.threshold is not None else hm["wake_threshold"]
        result = wake_bench(data_dir, cache_dir, wake_model=hm["wake_model"], threshold=threshold,
                            max_hours=args.max_hours, workers=args.workers, generate=not args.no_generate,
                            extra_positives=[Path(p) for p in args.positives], log=log)
        path = _write(out_dir, "wake.json", {"environment": env, **result})
        neg, pos = result["negatives"], result["positives"]
        log(f"wake: recall {pos['clean']['detected']}/{pos['clean']['n']} clean, "
            f"{pos['speech_10db']['detected']}/{pos['speech_10db']['n']} under speech; "
            f"{neg['false_accepts']} false accepts in {neg['hours']} h -> {path}")
    if "tests" in parts:
        result = tests_bench(args.python or [sys.executable], log)
        path = _write(out_dir, "tests.json", {"environment": env, **result})
        log(f"tests: {[r['summary'] for r in result['runs']]} -> {path}")
        code = code or (1 if any(r["failed"] or r["errors"] for r in result["runs"]) else 0)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="housemate bench", description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
