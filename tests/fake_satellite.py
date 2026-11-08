"""A scriptable satellite: speaks the hub's WebSocket protocol without a browser.

Used by the hub tests, and runnable by hand against a live hub. First run: it prints a pairing
code (approve it with ``hermes housemate pair <code>``) and saves the device key; then it joins a
room, speaks a 16 kHz WAV in real time and saves the spoken reply:

    python tests/fake_satellite.py --key-file kitchen.key --room Kitchen \\
        --wav hey_jarvis_whats_two_plus_two.wav --save reply.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import aiohttp

MIC_PREFIX = b"\x01"
PLAY_PREFIX = 0x02

_CLOSED = object()


class FakeSatellite:
    def __init__(self, url: str, device_id: str, name: str, device_key: str | None = None) -> None:
        self.url = url
        self.device_id = device_id
        self.name = name
        self.device_key = device_key
        self.close_code: int | None = None
        self.pair_code: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._control: asyncio.Queue = asyncio.Queue()
        self._pcm: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._closed = asyncio.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    async def connect(self, *, send_hello: bool = True) -> None:
        self._session = aiohttp.ClientSession()
        # Self-signed hub certificate: tests and the CLI skip verification.
        extra = {"ssl": False} if self.url.startswith("wss://") else {}
        self._ws = await self._session.ws_connect(self.url, max_msg_size=0, **extra)
        self._reader = asyncio.create_task(self._read())
        if send_hello:
            hello = {"type": "hello", "device_id": self.device_id, "name": self.name, "ua": "fake-satellite"}
            if self.device_key:
                hello["device_key"] = self.device_key
            await self.send_control(hello)

    async def send_control(self, msg: dict) -> None:
        await self._ws.send_str(json.dumps(msg))

    async def send_raw(self, data: bytes | str) -> None:
        if isinstance(data, str):
            await self._ws.send_str(data)
        else:
            await self._ws.send_bytes(data)

    async def send_audio(self, frame: bytes) -> None:
        """Send one mic frame (normally 640 bytes = 20 ms of PCM16 @ 16 kHz)."""
        await self._ws.send_bytes(MIC_PREFIX + frame)

    async def recv_control(self, timeout: float = 2.0, *, type: str | None = None) -> dict:
        """Next control message (optionally the next one of ``type``). Raises TimeoutError."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError(f"no {type or 'control'} message within {timeout}s")
            item = await asyncio.wait_for(self._control.get(), remaining)
            if item is _CLOSED:
                self._control.put_nowait(_CLOSED)
                raise ConnectionError(f"socket closed (code {self.close_code})")
            if type is None or item.get("type") == type:
                return item

    async def recv_pcm(self, timeout: float = 2.0) -> bytes:
        item = await asyncio.wait_for(self._pcm.get(), timeout)
        if item is _CLOSED:
            self._pcm.put_nowait(_CLOSED)
            raise ConnectionError(f"socket closed (code {self.close_code})")
        return item

    async def wait_closed(self, timeout: float = 2.0) -> int | None:
        await asyncio.wait_for(self._closed.wait(), timeout)
        return self.close_code

    async def close(self) -> None:
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader, 2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._reader.cancel()
        if self._session is not None:
            await self._session.close()

    async def _read(self) -> None:
        ws = self._ws
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "pair.code":
                        self.pair_code = data.get("code")
                    elif data.get("type") == "paired":
                        self.device_key = data.get("device_key") or self.device_key
                    self._control.put_nowait(data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    if msg.data[:1] == bytes([PLAY_PREFIX]):
                        self._pcm.put_nowait(msg.data[1:])
        finally:
            self.close_code = ws.close_code
            self._control.put_nowait(_CLOSED)
            self._pcm.put_nowait(_CLOSED)
            self._closed.set()


FRAME_S = 0.02
FRAME_BYTES = 640


def _read_wav16k(path: str) -> bytes:
    import wave

    with wave.open(path, "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, 16000):
            raise SystemExit(f"{path}: need a 16 kHz mono PCM16 WAV")
        return w.readframes(w.getnframes())


async def stream_pcm(sat: "FakeSatellite", pcm: bytes, *, tail_s: float = 1.5) -> float:
    """Send ``pcm`` as real-time 20 ms mic frames, then ``tail_s`` of silence (so the hub's
    endpointer closes the utterance). Returns the wall-clock time the speech ended."""
    import time

    frames = [pcm[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\x00") for i in range(0, len(pcm), FRAME_BYTES)]
    frames += [b"\x00" * FRAME_BYTES] * int(tail_s / FRAME_S)
    loop = asyncio.get_running_loop()
    start = loop.time()
    speech_end = None
    for i, frame in enumerate(frames):
        if i == (len(pcm) + FRAME_BYTES - 1) // FRAME_BYTES:
            speech_end = time.time()
        delay = start + i * FRAME_S - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)
        await sat.send_audio(frame)
    return speech_end or time.time()


async def receive_reply(sat: "FakeSatellite", *, timeout: float = 60.0, prefix: str | None = None) -> dict:
    """Collect one played stream (``play.begin`` .. ``play.end``/``play.abort``) like a real
    satellite: report ``playback.done`` once it would have finished playing."""
    import time

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    captions: list[dict] = []
    begin = None
    while begin is None:
        msg = await sat.recv_control(timeout=max(0.1, deadline - loop.time()))
        if msg["type"] == "caption":
            captions.append(msg)
        elif msg["type"] == "play.begin" and (prefix is None or msg["stream_id"].startswith(prefix)):
            begin = msg
    rate, sid = begin["sample_rate"], begin["stream_id"]
    pcm = bytearray()
    first_pcm_at = None
    end = None
    while end is None:
        get_pcm = asyncio.ensure_future(sat.recv_pcm(timeout=max(0.1, deadline - loop.time())))
        get_ctl = asyncio.ensure_future(sat.recv_control(timeout=max(0.1, deadline - loop.time())))
        done, pending = await asyncio.wait({get_pcm, get_ctl}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            item = task.result()
            if isinstance(item, (bytes, bytearray)):
                if first_pcm_at is None:
                    first_pcm_at = time.time()
                pcm.extend(item)
            elif item["type"] == "caption":
                captions.append(item)
            elif item["type"] in ("play.end", "play.abort") and item["stream_id"] == sid:
                end = item
        # a PCM frame may have raced the end message: drain what is queued
    while not sat._pcm.empty():
        item = sat._pcm.get_nowait()
        if isinstance(item, (bytes, bytearray)):
            pcm.extend(item)
    audio_s = len(pcm) / (2 * rate)
    if end["type"] == "play.end":
        played_until = (first_pcm_at or time.time()) + audio_s
        await asyncio.sleep(max(0.0, played_until - time.time()))
        await sat.send_control({"type": "playback.done", "stream_id": sid})
    return {"stream_id": sid, "rate": rate, "pcm": bytes(pcm), "audio_s": audio_s,
            "first_pcm_at": first_pcm_at, "ended": end["type"], "captions": captions}


def save_wav(path: str, pcm: bytes, rate: int) -> None:
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


async def _run(args: argparse.Namespace) -> int:
    key = args.key
    if key is None and args.key_file:
        try:
            key = open(args.key_file, encoding="utf-8").read().strip() or None
        except OSError:
            key = None
    sat = FakeSatellite(args.url, args.device_id, args.name, key)
    await sat.connect()
    try:
        while True:  # pairing (if needed) until the hub trusts us
            msg = await sat.recv_control(timeout=args.wait)
            if msg.get("type") == "pair.code":
                print(f"pairing code: {msg['code']} (approve it: hermes housemate pair {msg['code']})", file=sys.stderr)
            elif msg.get("type") == "paired":
                if args.key_file and sat.device_key:
                    with open(args.key_file, "w", encoding="utf-8") as fh:
                        fh.write(sat.device_key)
                print(f"paired; device key: {sat.device_key}", file=sys.stderr)
                break
        if args.room:
            await sat.send_control({"type": "room.set", "room": args.room})
        if not args.wav:
            return 0
        speech_end = await stream_pcm(sat, _read_wav16k(args.wav))
        reply = await receive_reply(sat, timeout=args.reply_timeout)
        if args.save:
            save_wav(args.save, reply["pcm"], reply["rate"])
        latency = reply["first_pcm_at"] - speech_end if reply["first_pcm_at"] else None
        print(json.dumps({
            "speech_end_to_first_pcm_s": round(latency, 3) if latency is not None else None,
            "reply_audio_s": round(reply["audio_s"], 2),
            "ended": reply["ended"],
            "captions": [f"{c['role']}: {c['text']}" for c in reply["captions"]],
        }, indent=1))
        return 0
    except (asyncio.TimeoutError, ConnectionError) as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        await sat.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fake Housemate satellite: pair, pick a room, speak a WAV, save the reply")
    parser.add_argument("--url", default="wss://127.0.0.1:8742/ws")
    parser.add_argument("--device-id", default="fake-satellite-1")
    parser.add_argument("--name", default="Fake satellite")
    parser.add_argument("--key", default=None, help="device key from an earlier pairing")
    parser.add_argument("--key-file", default=None, help="read the device key from / save it to this file")
    parser.add_argument("--room", default=None, help="room to join, e.g. Kitchen")
    parser.add_argument("--wav", default=None, help="16 kHz mono WAV to speak, e.g. 'hey jarvis, what's two plus two'")
    parser.add_argument("--save", default=None, help="write the spoken reply to this WAV")
    parser.add_argument("--wait", type=float, default=600.0, help="seconds to wait for pairing")
    parser.add_argument("--reply-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
