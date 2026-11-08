"""Wake-word and VAD models: pinned manifest, verified download, shared ONNX sessions.

openWakeWord's pre-trained models (melspectrogram, embedding, the stock wake heads) are
CC BY-NC-SA 4.0, so they are never shipped in this repo. ``ensure_model`` downloads them
once from the official openWakeWord release into ``<data_dir>/models/`` and checks each
file against a pinned SHA-256. Silero VAD is MIT.

Nothing here imports onnxruntime at module import time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "MODELS",
    "STOCK_WAKE_MODELS",
    "AudioModels",
    "ModelChecksumError",
    "ModelDownloadError",
    "ModelError",
    "ModelSpec",
    "ensure_model",
    "load_audio_models",
    "onnx_session",
    "resolve_wake_head",
    "sha256_file",
]

log = logging.getLogger(__name__)

_OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
_OWW_LICENCE = "CC BY-NC-SA 4.0 (openWakeWord pre-trained models, non-commercial)"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    url: str
    sha256: str
    size: int  # bytes; 0 = unknown
    licence: str


# Pinned on 2026-11-08. openWakeWord v0.6.0 has no release assets; its package downloads
# these same v0.5.1 files.
MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "melspectrogram",
            "melspectrogram.onnx",
            _OWW + "melspectrogram.onnx",
            "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
            1_087_958,
            _OWW_LICENCE,
        ),
        ModelSpec(
            "embedding",
            "embedding_model.onnx",
            _OWW + "embedding_model.onnx",
            "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
            1_326_578,
            _OWW_LICENCE,
        ),
        ModelSpec(
            "hey_jarvis",
            "hey_jarvis_v0.1.onnx",
            _OWW + "hey_jarvis_v0.1.onnx",
            "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
            1_271_370,
            _OWW_LICENCE,
        ),
        ModelSpec(
            "silero_vad",
            "silero_vad_v6.2.3.onnx",
            "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx",
            "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
            2_327_524,
            "MIT (Silero VAD)",
        ),
    )
}

STOCK_WAKE_MODELS = ("hey_jarvis",)

# Wake heads that ship as data inside the plugin (trained by us), looked up by name.
BUNDLED_WAKE_DIR = Path(__file__).resolve().parents[1] / "wakeword"

DOWNLOAD_TIMEOUT_S = 30.0
FAILURE_COOLDOWN_S = 300.0  # after a failed download, don't hit the network again for 5 min
_UNKNOWN_SIZE_CAP = 64 * 1024 * 1024


class ModelError(RuntimeError):
    """A model could not be made available."""


class ModelChecksumError(ModelError):
    """A downloaded (or cached) model does not match its pinned SHA-256."""


class ModelDownloadError(ModelError):
    """The download failed (network, timeout, too large) or is cooling down after a failure."""


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_dl_guard = threading.Lock()
_dl_locks: dict[str, threading.Lock] = {}
_failures: dict[str, tuple[float, str]] = {}


def _path_lock(key: str) -> threading.Lock:
    with _dl_guard:
        return _dl_locks.setdefault(key, threading.Lock())


def ensure_model(
    name: str,
    data_dir: Path | str,
    *,
    registry: Mapping[str, ModelSpec] | None = None,
    timeout: float = DOWNLOAD_TIMEOUT_S,
    clock: Callable[[], float] = time.monotonic,
) -> Path:
    """Return a verified local copy of a pinned model under ``<data_dir>/models/``.

    Downloads it if missing (or if the cached file fails its checksum), writing to a temp
    file in the same directory and renaming it into place only after the SHA-256 matches.
    Raises ``ModelChecksumError`` on a mismatch and ``ModelDownloadError`` on network
    failure. A failure is remembered for ``FAILURE_COOLDOWN_S`` so callers that retry on
    every device connection don't hammer the release server.
    """
    reg = MODELS if registry is None else registry
    try:
        spec = reg[name]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; known: {', '.join(sorted(reg))}") from None
    dest_dir = Path(data_dir) / "models"
    path = dest_dir / spec.filename
    key = os.path.normcase(str(path.absolute()))
    with _path_lock(key):
        if path.exists():
            if sha256_file(path) == spec.sha256:
                return path
            log.warning("housemate: %s does not match its pinned sha256; downloading it again", path)
        failed = _failures.get(key)
        if failed is not None and clock() - failed[0] < FAILURE_COOLDOWN_S:
            raise ModelDownloadError(f"{spec.name}: download failed recently ({failed[1]}); not retrying yet")
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            _download(spec, path, timeout=timeout, clock=clock)
        except ModelError as exc:
            _failures[key] = (clock(), str(exc))
            raise
        except Exception as exc:  # URLError, socket timeout, OSError
            _failures[key] = (clock(), f"{type(exc).__name__}: {exc}")
            raise ModelDownloadError(f"{spec.name}: download from {spec.url} failed: {exc}") from exc
        _failures.pop(key, None)
        log.info("housemate: downloaded %s (%s)", spec.filename, spec.licence)
        return path


def _download(spec: ModelSpec, path: Path, *, timeout: float, clock: Callable[[], float]) -> None:
    if not spec.url.startswith(("https://", "http://")):
        raise ModelDownloadError(f"{spec.name}: refusing non-http(s) url {spec.url!r}")
    cap = spec.size if spec.size > 0 else _UNKNOWN_SIZE_CAP
    deadline = clock() + max(4 * timeout, 60.0)
    req = urllib.request.Request(spec.url, headers={"User-Agent": "housemate-plugin"})
    fd, tmp = tempfile.mkstemp(prefix=f".{spec.filename}.", suffix=".part", dir=path.parent)
    try:
        h = hashlib.sha256()
        got = 0
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=timeout) as resp:
            while True:
                block = resp.read(1 << 16)
                if not block:
                    break
                got += len(block)
                if got > cap:
                    raise ModelDownloadError(f"{spec.name}: download larger than expected ({cap} bytes)")
                if clock() > deadline:
                    raise ModelDownloadError(f"{spec.name}: download took too long")
                h.update(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        digest = h.hexdigest()
        if digest != spec.sha256:
            raise ModelChecksumError(f"{spec.name}: sha256 {digest} does not match pinned {spec.sha256}")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- shared ONNX sessions -----------------------------------------------------------------

_sessions: dict[tuple[str, int], Any] = {}
_sessions_lock = threading.Lock()


def onnx_session(path: Path | str, *, threads: int = 1) -> Any:
    """One onnxruntime CPU session per model file, shared by every device stream.

    ``InferenceSession.run`` is thread-safe; per-device streaming state lives in the
    detector/VAD objects, never in the session.
    """
    key = (str(Path(path).resolve()), int(threads))
    with _sessions_lock:
        sess = _sessions.get(key)
    if sess is not None:
        return sess
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = int(threads)
    so.log_severity_level = 3
    sess = ort.InferenceSession(key[0], sess_options=so, providers=["CPUExecutionProvider"])
    with _sessions_lock:
        return _sessions.setdefault(key, sess)


# --- wiring helpers -------------------------------------------------------------------------


def resolve_wake_head(
    wake_model: str,
    data_dir: Path | str,
    *,
    bundled_dir: Path | None = None,
    registry: Mapping[str, ModelSpec] | None = None,
) -> Path:
    """Path of the wake head for ``wake_model``.

    ``wake_model`` is a stock name (``hey_jarvis``: downloaded and verified), the name of a
    head bundled with the plugin (``wakeword/<name>.onnx``, e.g. ``hey_hermes``), or a path
    to a custom openWakeWord-compatible head ``.onnx``.
    """
    reg = MODELS if registry is None else registry
    if wake_model in STOCK_WAKE_MODELS:
        return ensure_model(wake_model, data_dir, registry=reg)
    bundled = (bundled_dir or BUNDLED_WAKE_DIR) / f"{wake_model}.onnx"
    if bundled.is_file():
        return bundled
    candidate = Path(wake_model).expanduser()
    if candidate.suffix.lower() == ".onnx" and candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"wake model {wake_model!r} not found: use one of {', '.join(STOCK_WAKE_MODELS)}, "
        f"a head in {bundled.parent}, or a path to a .onnx head"
    )


@dataclass(frozen=True)
class AudioModels:
    """Verified model files plus factories for per-device pipeline objects.

    Sessions are shared (one per file per process); each ``new_*`` call returns an object
    with its own streaming state.
    """

    mel: Path
    embedding: Path
    head: Path
    vad: Path | None  # None = EnergyVad fallback
    wake_model: str

    def new_wake(self):
        from .wake import OnnxWakeDetector

        return OnnxWakeDetector(self.mel, self.embedding, self.head)

    def new_vad(self):
        from .vad import EnergyVad, SileroVad

        return SileroVad(self.vad) if self.vad is not None else EnergyVad()

    def new_endpointer(self, **kwargs):
        from .endpointer import Endpointer

        return Endpointer(self.new_vad(), **kwargs)


def load_audio_models(
    data_dir: Path | str,
    *,
    wake_model: str = "hey_jarvis",
    fallback_wake_model: str = "hey_jarvis",
    vad: str = "silero",
) -> AudioModels:
    """Download/verify everything the listeners need. Blocking: call it off the event loop.

    An unknown or missing wake model falls back to ``fallback_wake_model`` with a warning;
    a Silero download failure falls back to the energy VAD. Feature-model failures raise.
    """
    mel = ensure_model("melspectrogram", data_dir)
    emb = ensure_model("embedding", data_dir)
    try:
        head, used = resolve_wake_head(wake_model, data_dir), wake_model
    except (FileNotFoundError, ValueError) as exc:
        if wake_model == fallback_wake_model:
            raise
        log.warning("housemate: %s; falling back to %s", exc, fallback_wake_model)
        head, used = resolve_wake_head(fallback_wake_model, data_dir), fallback_wake_model
    vad_path: Path | None = None
    if vad == "silero":
        try:
            vad_path = ensure_model("silero_vad", data_dir)
        except ModelError as exc:
            log.warning("housemate: Silero VAD unavailable (%s); using the energy VAD", exc)
    elif vad != "energy":
        raise ValueError(f"vad must be 'silero' or 'energy', not {vad!r}")
    return AudioModels(mel=mel, embedding=emb, head=head, vad=vad_path, wake_model=used)
