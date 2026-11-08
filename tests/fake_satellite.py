"""A scriptable satellite: speaks the hub's WebSocket protocol without a browser.

Used by the hub tests, and runnable by hand against a live hub:

    python tests/fake_satellite.py --url wss://192.168.1.20:8787/ws --device-id test-sat-1
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


async def _run(args: argparse.Namespace) -> int:
    sat = FakeSatellite(args.url, args.device_id, args.name, args.key)
    await sat.connect()
    try:
        while True:
            msg = await sat.recv_control(timeout=args.wait)
            print(json.dumps(msg))
            if msg.get("type") == "paired":
                print(f"paired; device key: {sat.device_key}", file=sys.stderr)
                return 0
    except (asyncio.TimeoutError, ConnectionError) as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        await sat.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fake Housemate satellite")
    parser.add_argument("--url", default="wss://127.0.0.1:8787/ws")
    parser.add_argument("--device-id", default="fake-satellite-1")
    parser.add_argument("--name", default="Fake satellite")
    parser.add_argument("--key", default=None, help="device key from an earlier pairing")
    parser.add_argument("--wait", type=float, default=600.0, help="seconds to wait for the next message")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
