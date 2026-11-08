import asyncio
import sqlite3
from pathlib import Path

import aiohttp
import pytest
from yarl import URL

from housemate.core import protocol
from housemate.core.hub import Hub
from housemate.core.protocol import ProtocolError, decode, encode
from housemate.core.rooms import RoomRegistry
from housemate.core.store import HousemateStore
from housemate.core.tls import ensure_self_signed_cert

from fake_satellite import FakeSatellite

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
FRAME = b"\x00\x01" * 320  # 640 bytes = 20 ms PCM16 @ 16 kHz


# --- protocol ------------------------------------------------------------------


def test_encode_decode_roundtrip():
    msg = {"type": "play.begin", "stream_id": "s1", "sample_rate": 24000}
    assert decode(encode(msg)) == msg


def test_decode_rejects_unknown_type():
    with pytest.raises(ProtocolError):
        decode('{"type": "self.destruct"}')


def test_decode_rejects_missing_fields():
    with pytest.raises(ProtocolError):
        decode('{"type": "room.set"}')
    with pytest.raises(ProtocolError):
        decode('{"type": "hello", "device_id": "dev-1234"}')


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"no": "type"}', '{"type": 5}'])
def test_decode_rejects_garbage(text):
    with pytest.raises(ProtocolError):
        decode(text)


def test_decode_rejects_bad_field_types():
    with pytest.raises(ProtocolError):
        decode('{"type": "ptt", "on": "yes"}')
    with pytest.raises(ProtocolError):
        decode('{"type": "hello", "device_id": "../../etc", "name": "x", "ua": "y"}')


def test_encode_rejects_unknown_type():
    with pytest.raises(ProtocolError):
        encode({"type": "nope"})


def test_binary_prefixes_and_frame_size():
    assert protocol.MIC_PCM == 0x01 and protocol.PLAY_PCM == 0x02
    assert protocol.FRAME_BYTES == 640


# --- hub fixtures ------------------------------------------------------------------


class Env:
    def __init__(self, hub, store, rooms):
        self.hub, self.store, self.rooms = hub, store, rooms
        self.audio: list[tuple[str, bytes]] = []
        self.control: list[tuple[str, dict]] = []
        self.sats: list[FakeSatellite] = []
        hub.on_audio(lambda device_id, frame: self.audio.append((device_id, frame)))

        async def on_control(device_id, msg):
            self.control.append((device_id, msg))

        hub.on_control(on_control)

    async def sat(self, device_id="dev-kitchen", name="Kitchen phone", key=None, *, hello=True):
        s = FakeSatellite(self.hub.ws_url, device_id, name, key)
        await s.connect(send_hello=hello)
        self.sats.append(s)
        return s

    async def paired_sat(self, device_id="dev-kitchen", name="Kitchen phone", room="Kitchen"):
        """Pair a device, then reconnect with its key and choose a room."""
        first = await self.sat(device_id, name)
        code = (await first.recv_control(type="pair.code"))["code"]
        assert await self.hub.approve_pairing(code)
        await first.recv_control(type="paired")
        await first.close()
        s = await self.sat(device_id, name, first.device_key)
        await s.recv_control(type="paired")
        if room:
            await s.send_control({"type": "room.set", "room": room})
            await s.recv_control(type="state")
        return s

    async def close(self):
        for s in self.sats:
            await s.close()


async def _make_env(**hub_kwargs):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    store, rooms = HousemateStore(conn), RoomRegistry()
    kwargs = dict(host="127.0.0.1", port=0, cert=None, key=None, web_root=WEB_ROOT, tls=False)
    kwargs.update(hub_kwargs)
    hub = Hub(store, rooms, **kwargs)
    await hub.start()
    return Env(hub, store, rooms)


@pytest.fixture
async def env():
    e = await _make_env()
    yield e
    await e.close()
    await e.hub.stop()


async def _sync(sat):
    """Round-trip a bad message: once its error arrives, everything sent before was handled."""
    await sat.send_control({"type": "sync.probe"})
    await sat.recv_control(type="error")


async def _eventually(pred, timeout=2.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if pred():
            return True
        await asyncio.sleep(0.02)
    return pred()


# --- pairing and auth ----------------------------------------------------------------


async def test_unknown_device_gets_pairing_code(env):
    sat = await env.sat()
    msg = await sat.recv_control()
    assert msg["type"] == "pair.code"
    assert len(msg["code"]) == 6 and msg["code"].isdigit()


async def test_unpaired_device_audio_ignored(env):
    sat = await env.sat()
    await sat.recv_control(type="pair.code")
    for _ in range(5):
        await sat.send_audio(FRAME)
    await sat.send_control({"type": "room.set", "room": "Kitchen"})
    err = await sat.recv_control(type="error")
    assert err["code"] == "not_paired"
    await _sync(sat)
    assert env.audio == []
    assert env.rooms.devices_in("kitchen") == []
    assert await env.hub.send_control("dev-kitchen", {"type": "caption", "role": "assistant", "text": "hi"}) is False


async def test_bad_device_key_rejected(env):
    sat = await env.sat(key="forged-key-forged-key-forged-key")
    assert await sat.wait_closed() == 4401
    assert env.audio == [] and env.rooms.snapshot() == []


async def test_wrong_key_for_paired_device_rejected(env):
    await env.paired_sat()
    sat = await env.sat(key="not-the-key")
    assert await sat.wait_closed() == 4401


async def test_approve_pairing_notifies_waiting_device(env):
    sat = await env.sat()
    code = (await sat.recv_control(type="pair.code"))["code"]
    assert await env.hub.approve_pairing(code) is True
    paired = await sat.recv_control(type="paired")
    assert paired["device_key"] and env.store.verify_device("dev-kitchen", paired["device_key"])
    assert isinstance(paired.get("rooms"), list)
    # The same socket is trusted from now on.
    await sat.send_control({"type": "room.set", "room": "Office"})
    state = await sat.recv_control(type="state")
    assert state["room"] == "office"
    assert [c.device_id for c in env.rooms.devices_in("office")] == ["dev-kitchen"]


async def test_approve_unknown_or_used_code_returns_false(env):
    assert await env.hub.approve_pairing("000000") is False
    sat = await env.sat()
    code = (await sat.recv_control(type="pair.code"))["code"]
    assert await env.hub.approve_pairing(code) is True
    assert await env.hub.approve_pairing(code) is False


async def test_paired_device_attaches_to_room(env):
    sat = await env.paired_sat(room="Kitchen")
    assert [c.device_id for c in env.rooms.devices_in("kitchen")] == ["dev-kitchen"]
    assert env.store.get_device("dev-kitchen").room == "kitchen"
    # Reconnecting later lands back in the stored room without another room.set.
    key = sat.device_key
    await sat.close()
    assert await _eventually(lambda: env.rooms.devices_in("kitchen") == [])
    again = await env.sat(key=key)
    paired = await again.recv_control(type="paired")
    assert paired["room"] == "kitchen"
    assert [c.device_id for c in env.rooms.devices_in("kitchen")] == ["dev-kitchen"]


async def test_hello_forwarded_to_on_control_without_key(env):
    await env.paired_sat()
    hellos = [m for _, m in env.control if m["type"] == "hello"]
    assert hellos and all("device_key" not in m for m in hellos)


async def test_newer_connection_replaces_older(env):
    first = await env.paired_sat()
    second = await env.sat(key=first.device_key)
    await second.recv_control(type="paired")
    assert await first.wait_closed() == 4409
    assert len(env.rooms.devices_in("kitchen")) == 1
    assert await env.hub.send_control("dev-kitchen", {"type": "chime", "kind": "ding"})
    assert (await second.recv_control(type="chime"))["kind"] == "ding"


async def test_pairing_rate_limited_per_ip(env):
    sats = []
    for i in range(5):
        s = await env.sat(f"dev-pending-{i}")
        await s.recv_control(type="pair.code")
        sats.append(s)
    extra = await env.sat("dev-pending-5")
    err = await extra.recv_control(type="error")
    assert err["code"] == "rate_limited"
    assert await extra.wait_closed() == 4429
    # A pending device leaving frees its slot.
    await sats[0].close()
    await asyncio.sleep(0.1)
    late = await env.sat("dev-pending-6")
    assert (await late.recv_control())["type"] == "pair.code"


async def test_pending_flags_a_request_that_claims_a_paired_device(env):
    """Review M3: a keyless hello may claim a paired device id; approving it would rotate that
    device's key and kick the real one. The Rooms page gets a flag to warn about it."""
    await env.paired_sat("dev-kitchen", "Kitchen phone")
    imposter = await env.sat("dev-kitchen", "Kitchen phone")
    await imposter.recv_control(type="pair.code")
    [pending] = env.hub.pending_pairings()
    assert pending["repair"] is True and pending["name_taken"] is True


async def test_pending_flags_a_name_a_paired_device_has(env):
    await env.paired_sat("dev-kitchen", "Kitchen iPad")
    lookalike = await env.sat("dev-other", "kitchen ipad ")
    await lookalike.recv_control(type="pair.code")
    [pending] = env.hub.pending_pairings()
    assert pending["repair"] is False and pending["name_taken"] is True


async def test_pending_pairings_have_a_global_cap():
    env = await _make_env(max_pending_total=2, max_pending_per_ip=10)
    try:
        for i in range(2):
            s = await env.sat(f"dev-pending-{i}")
            await s.recv_control(type="pair.code")
        extra = await env.sat("dev-pending-x")
        assert (await extra.recv_control(type="error"))["code"] == "rate_limited"
        assert await extra.wait_closed() == 4429
    finally:
        await env.close()
        await env.hub.stop()


async def test_deny_cools_down_the_address_not_just_the_device():
    """A denied request from an address cannot come straight back as a fresh look-alike."""
    env = await _make_env(deny_ip_ttl_s=60)
    try:
        first = await env.sat("dev-evil-1", "Kitchen iPad")
        code = (await first.recv_control(type="pair.code"))["code"]
        assert await env.hub.deny_pairing(code)
        again = await env.sat("dev-evil-2", "Kitchen iPad")
        assert (await again.recv_control(type="error"))["code"] == "pairing_denied"
        assert env.hub.pending_pairings() == []
        # a paired device from that address still connects with its key
    finally:
        await env.close()
        await env.hub.stop()


async def test_revoked_socket_is_untrusted_at_once(env):
    """Review L6: between revoke and the close completing, nothing the socket sends is trusted."""
    await env.paired_sat()
    session = env.hub._sessions["dev-kitchen"]
    assert await env.hub.revoke_device("dev-kitchen") is True
    assert session.authed is False
    before = len(env.control)
    await env.hub._on_text(session, encode({"type": "ui.action", "kind": "approve", "id": "req-1"}))
    env.hub._on_binary(session, bytes([protocol.MIC_PCM]) + FRAME)
    assert len(env.control) == before
    assert env.audio == []


async def test_replaced_socket_is_untrusted_at_once(env):
    first = await env.paired_sat()
    old = env.hub._sessions["dev-kitchen"]
    newer = await env.sat("dev-kitchen", "Kitchen phone", first.device_key)
    await newer.recv_control(type="paired")
    assert old.authed is False
    assert env.hub._sessions["dev-kitchen"].authed is True


async def test_pairing_wait_times_out_cleanly():
    e = await _make_env(pairing_ttl_s=0.3)
    try:
        sat = await e.sat()
        code = (await sat.recv_control(type="pair.code"))["code"]
        err = await sat.recv_control(type="error", timeout=3)
        assert err["code"] == "pairing_expired"
        assert await sat.wait_closed() == 4408
        assert await e.hub.approve_pairing(code) is False
        assert e.store.get_device("dev-kitchen") is None
    finally:
        await e.close()
        await e.hub.stop()


async def test_silent_socket_closed_after_hello_timeout():
    e = await _make_env(hello_timeout_s=0.3)
    try:
        sat = await e.sat(hello=False)
        assert await sat.wait_closed(3) == 4408
    finally:
        await e.close()
        await e.hub.stop()


async def test_revoked_device_is_kicked(env):
    sat = await env.paired_sat()
    assert await env.hub.revoke_device("dev-kitchen") is True
    assert await sat.wait_closed() == 4401
    assert env.rooms.devices_in("kitchen") == []
    assert env.store.verify_device("dev-kitchen", sat.device_key) is False


# --- audio and control ----------------------------------------------------------------


async def test_paired_audio_reaches_on_audio(env):
    sat = await env.paired_sat()
    await sat.send_audio(FRAME)
    await sat.send_audio(FRAME)
    await _sync(sat)
    assert env.audio == [("dev-kitchen", FRAME), ("dev-kitchen", FRAME)]


async def test_bad_audio_frame_length_dropped_and_counted(env):
    sat = await env.paired_sat()
    before = env.hub.stats["dropped_frames"]
    await sat.send_audio(FRAME[:-2])
    await sat.send_audio(FRAME + b"\x00\x00")
    await sat.send_raw(b"\x02" + FRAME)  # playout prefix from a satellite
    await sat.send_raw(b"")
    await _sync(sat)
    assert env.audio == []
    assert env.hub.stats["dropped_frames"] - before == 4
    assert not sat.closed


async def test_unknown_control_type_gets_error_not_crash(env):
    sat = await env.paired_sat()
    await sat.send_raw("{this is not json")
    assert (await sat.recv_control(type="error"))["code"] == "bad_message"
    await sat.send_control({"type": "self.destruct"})
    assert (await sat.recv_control(type="error"))["code"] == "bad_message"
    await sat.send_control({"type": "caption", "role": "user", "text": "hub-only type"})
    assert (await sat.recv_control(type="error"))["code"] == "bad_message"
    # Still alive and trusted.
    await sat.send_audio(FRAME)
    await _sync(sat)
    assert env.audio == [("dev-kitchen", FRAME)]


async def test_oversized_frame_closes_socket(env):
    sat = await env.paired_sat()
    await sat.send_raw(b"\x01" + b"\x00" * (64 * 1024 + 1))
    assert await sat.wait_closed() == 1009
    assert await _eventually(lambda: env.rooms.devices_in("kitchen") == [])


async def test_control_messages_forwarded_to_on_control(env):
    sat = await env.paired_sat()
    await sat.send_control({"type": "ptt", "on": True})
    await sat.send_control({"type": "ui.action", "kind": "stop_timer", "id": "t1"})
    await _sync(sat)
    types = [(d, m["type"]) for d, m in env.control]
    assert ("dev-kitchen", "ptt") in types and ("dev-kitchen", "ui.action") in types


async def test_send_control_and_pcm_reach_device(env):
    sat = await env.paired_sat()
    assert await env.hub.send_control(
        "dev-kitchen", {"type": "play.begin", "stream_id": "s1", "sample_rate": 24000}
    )
    assert (await sat.recv_control(type="play.begin"))["sample_rate"] == 24000
    pcm = b"\x10\x00" * 1000
    assert await env.hub.send_pcm("dev-kitchen", pcm) is True
    got = b""
    while len(got) < len(pcm):
        got += await sat.recv_pcm()
    assert got == pcm
    assert await env.hub.send_pcm("dev-nobody", pcm) is False


async def test_state_message_updates_room_registry(env):
    await env.paired_sat()
    assert await env.hub.send_control("dev-kitchen", {"type": "state", "state": "speaking"})
    assert env.rooms.get("dev-kitchen").state == "speaking"


async def test_disconnect_detaches_device(env):
    sat = await env.paired_sat()
    await sat.close()
    assert await _eventually(lambda: env.rooms.snapshot() == [])
    assert await env.hub.send_control("dev-kitchen", {"type": "chime", "kind": "ding"}) is False


# --- static files ------------------------------------------------------------------------


async def test_static_satellite_served(env):
    async with aiohttp.ClientSession() as http:
        async with http.get(env.hub.url + "/") as resp:
            assert resp.status == 200
            assert resp.content_type == "text/html"
            assert "<html" in (await resp.text()).lower()


async def test_static_rejects_traversal(env):
    async with aiohttp.ClientSession() as http:
        for raw in ("/..%2fplugin.yaml", "/%2e%2e/plugin.yaml", "/..%5cplugin.yaml", "/%2e%2e%2f%2e%2e%2fplugin.yaml"):
            async with http.get(URL(env.hub.url + raw, encoded=True)) as resp:
                assert resp.status == 404, raw
                assert "requires_hermes" not in await resp.text()


# --- TLS ---------------------------------------------------------------------------------


async def test_hub_serves_tls_with_self_signed_cert(tmp_path):
    cert, key, _ = ensure_self_signed_cert(tmp_path, ["127.0.0.1", "localhost"])
    e = await _make_env(cert=cert, key=key, tls=True)
    try:
        assert e.hub.url.startswith("https://") and e.hub.ws_url.startswith("wss://")
        async with aiohttp.ClientSession() as http:
            async with http.get(e.hub.url + "/", ssl=False) as resp:
                assert resp.status == 200
        sat = await e.sat()
        assert (await sat.recv_control())["type"] == "pair.code"
    finally:
        await e.close()
        await e.hub.stop()


async def test_cross_origin_ws_refused(env):
    async with aiohttp.ClientSession() as http:
        with pytest.raises(aiohttp.WSServerHandshakeError) as info:
            await http.ws_connect(env.hub.ws_url, origin="https://evil.example")
        assert info.value.status == 403
        ws = await http.ws_connect(env.hub.ws_url, origin=env.hub.url)
        await ws.close()


# --- port clash and disconnects ---------------------------------------------------------------


async def test_hub_port_clash_fails_loudly():
    import socket

    from housemate.core.hub import HubPortInUse

    squatter = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    squatter.bind(("127.0.0.1", 0))
    squatter.listen(1)
    port = squatter.getsockname()[1]
    try:
        for host in ("0.0.0.0", "127.0.0.1"):
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            hub = Hub(HousemateStore(conn), RoomRegistry(), host=host, port=port, cert=None, key=None,
                      web_root=WEB_ROOT, tls=False)
            with pytest.raises(HubPortInUse):
                await hub.start()
            await hub.stop()
    finally:
        squatter.close()


async def test_disconnect_callback_fires_once_per_trusted_drop(env):
    gone: list[str] = []
    env.hub.on_disconnect(gone.append)
    sat = await env.paired_sat(device_id="dev-office", room="Office")
    gone.clear()  # paired_sat closed its (trusted) pairing socket first
    await sat.close()
    assert await _eventually(lambda: gone == ["dev-office"])
    await asyncio.sleep(0.05)
    assert gone == ["dev-office"]


def test_server_tls_context_ignores_an_injected_client_sslcontext(tmp_path, monkeypatch):
    """Hermes injects truststore's client-only SSLContext into ``ssl``; the hub must not serve with it."""
    import ssl

    from housemate.core.hub import server_ssl_context

    class InjectedContext(ssl.SSLContext):  # what truststore.inject_into_ssl() does
        pass

    cert, key, _ = ensure_self_signed_cert(tmp_path, ["localhost", "127.0.0.1"])
    monkeypatch.setattr(ssl, "SSLContext", InjectedContext)
    ctx = server_ssl_context(cert, key)
    assert type(ctx).__module__ == "ssl" and type(ctx).__name__ == "SSLContext"
    assert ctx.minimum_version >= ssl.TLSVersion.TLSv1_2


async def test_hub_serves_tls(tmp_path):
    import ssl

    cert, key, _ = ensure_self_signed_cert(tmp_path, ["localhost", "127.0.0.1"])
    e = await _make_env(cert=cert, key=key, tls=True)
    try:
        client = ssl.create_default_context()
        client.check_hostname, client.verify_mode = False, ssl.CERT_NONE
        reader, writer = await asyncio.open_connection("127.0.0.1", e.hub.port, ssl=client)
        writer.write(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        head = await asyncio.wait_for(reader.read(64), 5)
        writer.close()
        assert head.startswith(b"HTTP/1.1 200")
    finally:
        await e.hub.stop()
