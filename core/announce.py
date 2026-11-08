"""Announcements, intercom and room cards: hub-side and deterministic.

``Announcer`` pushes to the online devices of one or more rooms:

- ``announce(text, rooms, from_room)``: a chime, a caption and the spoken text to every online
  device in the target rooms. ``rooms="all"`` with ``from_room`` set is the intercom: it skips
  the room the announcement came from.
- ``say(room, text)``: only the speech (timer rings use it).
- ``show(room, title, body)``: a card on the room's screens.
- ``chime(room, kind)`` / ``push_timers(room, items)``: small control pushes.

No Hermes imports: ``tts`` is an injected ``async (text) -> Path``. The adapter backs it with
Hermes's ``text_to_speech`` tool (``tts_paths_from_result`` parses that tool's JSON). The audio
file is decoded to PCM16 mono with ffmpeg and streamed as ``play.begin`` / PCM / ``play.end``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shutil
import subprocess
import uuid
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Literal

from .rooms import RoomRegistry
from .store import slugify

__all__ = ["PLAY_RATE", "AnnounceResult", "Announcer", "decode_to_pcm", "tts_paths_from_result"]

log = logging.getLogger(__name__)

PLAY_RATE = 24_000
DECODE_TIMEOUT_S = 60.0
_CACHE_ENTRIES = 8  # timer rings repeat the same sentence every 20 s
_CACHE_MAX_BYTES = 2_000_000
_MAX_TITLE = 512
_MAX_TEXT = 16_000


@dataclass(frozen=True)
class AnnounceResult:
    ok: bool
    rooms: list[str] = field(default_factory=list)  # where it went; on "unknown room", the known rooms
    devices: list[str] = field(default_factory=list)  # devices that got the audio
    offline: list[str] = field(default_factory=list)  # target rooms with no device online
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "rooms": list(self.rooms),
            "devices": list(self.devices),
            "offline": list(self.offline),
            "error": self.error,
        }


# --- audio helpers -------------------------------------------------------------------------------


def tts_paths_from_result(result: str | bytes | dict) -> list[Path]:
    """Audio file(s) from a ``text_to_speech`` tool result; raises RuntimeError on failure."""
    data: Any = result
    if isinstance(result, (str, bytes)):
        try:
            data = json.loads(result)
        except ValueError:
            raise RuntimeError(f"text_to_speech returned something unexpected: {str(result)[:200]}") from None
    if not isinstance(data, dict):
        raise RuntimeError("text_to_speech returned something unexpected")
    if not data.get("success"):
        raise RuntimeError(str(data.get("error") or "text_to_speech failed"))
    paths = data.get("file_paths") or ([data["file_path"]] if data.get("file_path") else [])
    if not paths:
        raise RuntimeError("text_to_speech returned no audio file")
    return [Path(p) for p in paths]


def _read_wav_pcm16(path: Path, rate: int) -> bytes | None:
    """The samples of a WAV that already is PCM16 mono at ``rate``, else None."""
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) == (1, 2, rate, "NONE"):
                return w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError):
        return None
    return None


def _run_blocking(cmd: list[str]) -> tuple[bytes, bytes, int]:
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=DECODE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg timed out decoding the audio") from None
    return done.stdout, done.stderr, done.returncode


async def decode_to_pcm(path: str | Path, rate: int = PLAY_RATE) -> bytes:
    """Decode any audio file to PCM16 LE mono at ``rate`` (ffmpeg; matching WAVs are read directly)."""
    path = Path(path)
    direct = _read_wav_pcm16(path, rate)
    if direct is not None:
        return direct
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is needed to play this audio but was not found on PATH")
    cmd = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(int(rate)), "pipe:1",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except NotImplementedError:  # a selector event loop on Windows has no subprocess support
        out, err, code = await asyncio.to_thread(_run_blocking, cmd)
    else:
        try:
            out, err = await asyncio.wait_for(proc.communicate(), DECODE_TIMEOUT_S)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise RuntimeError("ffmpeg timed out decoding the audio") from None
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise
        code = proc.returncode
    if code != 0:
        detail = err.decode("utf-8", errors="replace").strip()[-300:] or f"exit code {code}"
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {detail}")
    return out


def _slug_or_none(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    try:
        return slugify(name)
    except ValueError:
        return None


# --- the announcer ---------------------------------------------------------------------------------


class Announcer:
    def __init__(
        self,
        hub: Any,
        rooms: RoomRegistry,
        tts: Callable[[str], Awaitable[Path | list[Path]]],
        *,
        decode: Callable[[Path, int], Awaitable[bytes]] = decode_to_pcm,
        sample_rate: int = PLAY_RATE,
    ) -> None:
        self._hub = hub
        self._rooms = rooms
        self._tts = tts
        self._decode = decode
        self.sample_rate = sample_rate
        self._cache: OrderedDict[str, bytes] = OrderedDict()

    def known_rooms(self) -> list[str]:
        """Room slugs from the store (configured or used before) plus any room online now."""
        slugs: set[str] = set()
        store = getattr(self._hub, "store", None)
        if store is not None:
            try:
                slugs.update(r.slug for r in store.list_rooms())
            except Exception:
                log.exception("housemate: listing rooms failed")
        slugs.update(state.room for state in self._rooms.snapshot() if state.room)
        return sorted(slugs)

    async def announce(
        self, text: str, rooms: list[str] | str | Literal["all"], from_room: str | None = None
    ) -> AnnounceResult:
        text = " ".join(str(text or "").split())
        if not text:
            return AnnounceResult(ok=False, error="nothing to announce")
        known = self.known_rooms()
        if isinstance(rooms, str) and rooms.strip().lower() == "all":
            origin = _slug_or_none(from_room)
            targets = [r for r in known if r != origin]
            if not targets:
                return AnnounceResult(ok=False, rooms=known, error="there is no other room to announce to")
        else:
            names = [rooms] if isinstance(rooms, str) else list(rooms or [])
            if not names:
                return AnnounceResult(ok=False, rooms=known, error="say which rooms to announce to, or 'all'")
            targets, unknown = [], []
            for name in names:
                slug = _slug_or_none(name)
                if slug is None or slug not in known:
                    unknown.append(str(name))
                elif slug not in targets:
                    targets.append(slug)
            if unknown:
                noun = "room" if len(unknown) == 1 else "rooms"
                listed = ", ".join(known) or "none yet"
                return AnnounceResult(
                    ok=False,
                    rooms=known,
                    error=f"unknown {noun} {', '.join(repr(u) for u in unknown)}; known rooms: {listed}",
                )

        online = {slug: self._online(slug) for slug in targets}
        offline = [slug for slug in targets if not online[slug]]
        device_ids = [d for slug in targets for d in online[slug]]
        if not device_ids:
            return AnnounceResult(
                ok=False, rooms=targets, offline=offline, error=f"no device is online in {', '.join(offline)}"
            )

        await self._send_all(device_ids, {"type": "chime", "kind": "announce"})
        await self._send_all(device_ids, {"type": "caption", "role": "announcement", "text": text[:_MAX_TEXT]})
        try:
            pcm = await self._speech_pcm(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("housemate: announcement TTS failed: %s", exc)
            return AnnounceResult(ok=False, rooms=targets, offline=offline, error=f"text-to-speech failed: {exc}")
        delivered = await self._stream(device_ids, pcm, "ann")
        if not delivered:
            return AnnounceResult(
                ok=False, rooms=targets, offline=offline, error="the devices went offline before it played"
            )
        return AnnounceResult(ok=True, rooms=targets, devices=delivered, offline=offline)

    async def say(self, room: str | Iterable[str], text: str) -> list[str]:
        """Speak ``text`` on the online devices of ``room``; returns the devices that got it.

        Raises RuntimeError when TTS or decoding fails. Cancelling it aborts playback.
        """
        names = [room] if isinstance(room, str) else list(room)
        device_ids = list(dict.fromkeys(d for name in names for d in self._online(_slug_or_none(name))))
        if not device_ids or not str(text or "").strip():
            return []
        pcm = await self._speech_pcm(str(text).strip())
        return await self._stream(device_ids, pcm, "say")

    async def show(self, room: str, title: str, body: str) -> bool:
        """Show a card in ``room``. False when the room has no device online."""
        msg = {
            "type": "card",
            "id": f"card-{uuid.uuid4().hex[:12]}",
            "title": str(title or "")[:_MAX_TITLE],
            "body": str(body or "")[:_MAX_TEXT],
        }
        return await self._send_room(room, msg) > 0

    async def chime(self, room: str, kind: str = "timer") -> int:
        return await self._send_room(room, {"type": "chime", "kind": kind})

    async def push_timers(self, room: str, items: list[dict]) -> int:
        return await self._send_room(room, {"type": "timers", "items": list(items)})

    # --- internals --------------------------------------------------------------------------------

    def _online(self, slug: str | None) -> list[str]:
        if not slug:
            return []
        try:
            return sorted(conn.device_id for conn in self._rooms.devices_in(slug))
        except ValueError:
            return []

    async def _speech_pcm(self, text: str) -> bytes:
        pcm = self._cache.get(text)
        if pcm is not None:
            self._cache.move_to_end(text)
            return pcm
        result = await self._tts(text)
        paths = [Path(p) for p in result] if isinstance(result, (list, tuple)) else [Path(result)]
        if not paths:
            raise RuntimeError("text-to-speech returned no audio")
        pcm = b"".join([await self._decode(p, self.sample_rate) for p in paths])
        if not pcm:
            raise RuntimeError("text-to-speech audio was empty")
        if len(pcm) <= _CACHE_MAX_BYTES:
            self._cache[text] = pcm
            while len(self._cache) > _CACHE_ENTRIES:
                self._cache.popitem(last=False)
        return pcm

    async def _stream(self, device_ids: list[str], pcm: bytes, prefix: str) -> list[str]:
        results = await asyncio.gather(
            *(self._stream_one(d, pcm, f"{prefix}-{uuid.uuid4().hex[:12]}") for d in device_ids)
        )
        return [d for d, ok in zip(device_ids, results) if ok]

    async def _stream_one(self, device_id: str, pcm: bytes, stream_id: str) -> bool:
        try:
            begin = {"type": "play.begin", "stream_id": stream_id, "sample_rate": self.sample_rate}
            if not await self._hub.send_control(device_id, begin):
                return False
            ok = await self._hub.send_pcm(device_id, pcm)
            await self._hub.send_control(device_id, {"type": "play.end" if ok else "play.abort", "stream_id": stream_id})
            return bool(ok)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self._hub.send_control(device_id, {"type": "play.abort", "stream_id": stream_id})
            raise
        except Exception:
            log.exception("housemate: streaming to %s failed", device_id)
            return False

    async def _send(self, device_id: str, msg: dict) -> bool:
        try:
            return bool(await self._hub.send_control(device_id, msg))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("housemate: sending %s to %s failed", msg.get("type"), device_id)
            return False

    async def _send_all(self, device_ids: list[str], msg: dict) -> int:
        results = await asyncio.gather(*(self._send(d, msg) for d in device_ids))
        return sum(1 for ok in results if ok)

    async def _send_room(self, room: str, msg: dict) -> int:
        return await self._send_all(self._online(_slug_or_none(room)), msg)
