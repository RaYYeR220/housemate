"""``hermes housemate ...``: manage the hub from a terminal.

The hub runs inside the gateway; these commands reach it through ``HousemateClient`` (loopback
control API). Room and device housekeeping also works while the gateway is stopped: it then edits
the saved state directly.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from datetime import datetime
from typing import Any, TextIO

from . import tools
from .core.client import HUB_DOWN

__all__ = ["DESCRIPTION", "HUB_PORT", "format_status", "main", "open_store", "render_qr", "setup_parser"]

HUB_PORT = 8742  # default LAN port of the hub (config: port)

DESCRIPTION = """\
Manage Housemate. The hub runs inside the gateway (hermes gateway run, housemate platform enabled).

  hermes housemate status                 rooms, devices and timers
  hermes housemate cert                   hub address, certificate fingerprint, QR code
  hermes housemate pair 123456            approve the device that shows this code
  hermes housemate devices                paired devices
  hermes housemate revoke <device-id>     unpair a device
  hermes housemate rooms add Living Room  add a room (also: remove, group)
  hermes housemate bench [wake latency tests]  measure; writes results/*.json
"""


# --- formatting -------------------------------------------------------------------------------------


def _clock(seconds: Any) -> str:
    try:
        total = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _hhmm(iso: Any) -> str:
    try:
        return datetime.fromisoformat(str(iso)).strftime("%H:%M")
    except ValueError:
        return "?"


def _timer_phrase(timer: dict) -> str:
    kind = timer.get("kind") or "timer"
    label = timer.get("label") or ""
    name = f"{label} {kind}" if label else kind
    if timer.get("state") == "ringing":
        return f"{name} RINGING"
    if kind == "alarm":
        return f"{name} at {_hhmm(timer.get('due'))}"
    return f"{name} {_clock(timer.get('remaining_s'))} left"


def format_status(rooms: dict, devices: dict | None = None) -> str:
    """Plain-text status for the terminal and for chat (/housemate)."""
    lines = []
    if rooms.get("hub_url"):
        lines.append(f"Housemate hub: {rooms['hub_url']}")
    if not rooms.get("rooms"):
        lines.append("No rooms yet. Open the hub address on a phone or laptop, pair it and pick a room.")
    for room in rooms.get("rooms") or []:
        parts = [f"{room.get('name') or room.get('room')}: {room.get('state', '?')}"]
        names = [d.get("name") or d.get("device_id") for d in room.get("devices") or []]
        if names:
            parts.append(", ".join(names))
        timers = [_timer_phrase(t) for t in room.get("timers") or []]
        if timers:
            parts.append(", ".join(timers))
        lines.append(" | ".join(parts))
    unassigned = rooms.get("unassigned_devices") or []
    if unassigned:
        lines.append("Online, no room picked yet: " + ", ".join(d.get("name") or d.get("device_id") for d in unassigned))
    if devices:
        waiting = [d for d in devices.get("devices") or [] if not d.get("paired")]
        if waiting:
            noun = "device" if len(waiting) == 1 else "devices"
            lines.append(f"Waiting to pair: {len(waiting)} {noun} (approve the code it shows: hermes housemate pair <code>)")
    return "\n".join(lines)


# --- QR ----------------------------------------------------------------------------------------------


def render_qr(text: str, *, ascii_only: bool = False) -> str | None:
    """A terminal QR code for ``text`` (None without the qrcode package).

    Default: qrcode's half-block output. ``ascii_only``: '#' and spaces, for consoles that cannot
    print block characters.
    """
    try:
        import qrcode
    except ImportError:
        return None
    qr = qrcode.QRCode(border=2)
    qr.add_data(text)
    qr.make(fit=True)
    if not ascii_only:
        buf = io.StringIO()
        qr.print_ascii(out=buf, invert=True)
        return buf.getvalue()
    rows = ["".join("  " if dark else "##" for dark in row) for row in qr.get_matrix()]
    return "\n".join(rows) + "\n"


def _write_qr(out: TextIO, text: str) -> None:
    qr = render_qr(text)
    if qr is None:
        print("(install the qrcode package to get a scannable code here)", file=out)
        return
    try:
        out.write(qr)
    except UnicodeEncodeError:
        out.write(render_qr(text, ascii_only=True) or "")
    out.flush()


# --- offline state --------------------------------------------------------------------------------------


def open_store():
    """The hub's saved state (``plugin_db("housemate")``), for edits while the gateway is stopped."""
    try:
        from plugins.plugin_storage import plugin_db
    except Exception:
        return None
    from .core.store import HousemateStore

    return HousemateStore(plugin_db(tools.PLUGIN_NAME))


def _offline_op(op: str, payload: dict) -> dict | None:
    store = open_store()
    if store is None:
        return None
    from .core.control import HubServices, run_op
    from .core.rooms import RoomRegistry

    services = HubServices(store=store, registry=RoomRegistry(), hub=None, timers=None, announcer=None)
    return asyncio.run(run_op(services, op, payload))


def _hub_info(client) -> dict:
    info = client.call("info")
    if info.get("ok") or info.get("error") != HUB_DOWN:
        return info
    directory = tools.data_dir()
    cert = directory / "hub-cert.pem" if directory is not None else None
    if cert is None or not cert.is_file():
        return {"ok": False, "error": f"{HUB_DOWN} (no hub certificate has been made yet)"}
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes

    from .core.tls import hub_urls

    fingerprint = x509.load_pem_x509_certificate(cert.read_bytes()).fingerprint(hashes.SHA256()).hex(":").upper()
    urls = hub_urls("https", HUB_PORT)
    return {"ok": True, "hub_url": urls[0], "urls": urls, "fingerprint": fingerprint, "offline": True}


# --- commands ------------------------------------------------------------------------------------------


def _status(args, client, out) -> int:
    rooms = client.call("rooms")
    if not rooms.get("ok"):
        print(f"Housemate: {rooms.get('error')}", file=out)
        return 1
    devices = client.call("devices")
    print(format_status(rooms, devices if devices.get("ok") else None), file=out)
    return 0


def _pair(args, client, out) -> int:
    result = client.call("pair_approve", {"code": args.code})
    if not result.get("ok"):
        print(f"Not paired: {result.get('error')}", file=out)
        return 1
    print("Paired. If the device asks, pick its room on its screen.", file=out)
    return 0


def _print_devices(devices: list[dict], out) -> None:
    if not devices:
        print("No devices yet. Open the hub address on a phone or laptop (hermes housemate cert).", file=out)
    for d in devices:
        flags = "online" if d.get("online") else "offline"
        if not d.get("paired"):
            flags += ", waiting to pair"
        print(f"{d.get('device_id')}  {d.get('name')}  room: {d.get('room') or '-'}  ({flags})", file=out)


def _devices(args, client, out) -> int:
    result = client.call("devices")
    if result.get("ok"):
        _print_devices(result.get("devices") or [], out)
        return 0
    if result.get("error") != HUB_DOWN:
        print(f"Housemate: {result.get('error')}", file=out)
        return 1
    store = open_store()
    if store is None:
        print(f"Housemate: {HUB_DOWN}", file=out)
        return 1
    print(f"({HUB_DOWN}; showing saved devices)", file=out)
    _print_devices([
        {"device_id": d.device_id, "name": d.name, "room": d.room, "paired": d.paired, "online": False}
        for d in store.list_devices()
    ], out)
    return 0


def _revoke(args, client, out) -> int:
    result = client.call("revoke", {"device_id": args.device_id})
    if not result.get("ok") and result.get("error") == HUB_DOWN:
        store = open_store()
        if store is not None:
            ok = store.revoke_device(args.device_id)
            result = {"ok": ok, "error": None if ok else f"no device {args.device_id!r}"}
    if not result.get("ok"):
        print(f"Not revoked: {result.get('error')}", file=out)
        return 1
    print(f"Revoked {args.device_id}. It has to pair again to be used.", file=out)
    return 0


def _room_op(client, op: str, payload: dict) -> dict:
    result = client.call(op, payload)
    if not result.get("ok") and result.get("error") == HUB_DOWN:
        result = _offline_op(op, payload) or result
    return result


def _rooms(args, client, out) -> int:
    action = getattr(args, "rooms_command", None)
    if action == "add":
        result = _room_op(client, "room_add", {"name": " ".join(args.name)})
        done = lambda r: f"Added room {r['room']['name']} ({r['room']['slug']})."  # noqa: E731
    elif action == "remove":
        result = _room_op(client, "room_remove", {"name": " ".join(args.name)})
        done = lambda r: f"Removed room {r['room']['name']} ({r['room']['slug']})."  # noqa: E731
    elif action == "group":
        result = _room_op(client, "room_group", {"room": args.room, "group": args.group})
        done = lambda r: (f"{r['room']} is in earshot group {r['group']}." if r.get("group")  # noqa: E731
                          else f"{r['room']} is in no earshot group now.")
    else:
        return _status(args, client, out)
    if not result.get("ok"):
        print(f"Housemate: {result.get('error')}", file=out)
        return 1
    print(done(result), file=out)
    return 0


def _cert(args, client, out) -> int:
    info = _hub_info(client)
    if not info.get("ok"):
        print(f"Housemate: {info.get('error')}", file=out)
        return 1
    urls = info.get("urls") or ([info["hub_url"]] if info.get("hub_url") else [])
    if not urls:
        print("Housemate: the hub did not report an address", file=out)
        return 1
    if info.get("offline"):
        print("(hub not running: address and fingerprint from the saved certificate; start the gateway first)", file=out)
    print("Open this on each phone, tablet or laptop:", file=out)
    print(f"  {urls[0]}", file=out)
    for other in urls[1:]:
        print(f"  also: {other}", file=out)
    print("Certificate SHA-256 fingerprint:", file=out)
    print(f"  {info.get('fingerprint') or 'unknown'}", file=out)
    print("The browser warns about the self-signed certificate: continue only if the fingerprint matches.", file=out)
    _write_qr(out, urls[0])
    return 0


def _bench(args, client, out) -> int:
    from .bench.run import run

    return run(args, client, out)


_COMMANDS = {
    "status": _status,
    "pair": _pair,
    "devices": _devices,
    "revoke": _revoke,
    "rooms": _rooms,
    "cert": _cert,
    "bench": _bench,
}


def setup_parser(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="housemate_command", metavar="<command>")
    subs.add_parser("status", help="rooms, devices and timers")
    pair = subs.add_parser("pair", help="approve the device that shows this 6-digit code")
    pair.add_argument("code")
    subs.add_parser("devices", help="paired devices")
    revoke = subs.add_parser("revoke", help="unpair a device")
    revoke.add_argument("device_id")
    rooms = subs.add_parser("rooms", help="add, remove or group rooms")
    actions = rooms.add_subparsers(dest="rooms_command", metavar="<action>")
    add = actions.add_parser("add", help="add a room")
    add.add_argument("name", nargs="+")
    remove = actions.add_parser("remove", help="remove a room (no device online in it, no timers)")
    remove.add_argument("name", nargs="+")
    group = actions.add_parser("group", help="put a room in an earshot group: rooms whose devices hear each other")
    group.add_argument("room")
    group.add_argument("group", nargs="?", default=None, help="group name; leave out to clear")
    subs.add_parser("cert", help="hub address, certificate fingerprint and a QR code to open it")
    from .bench.run import add_arguments

    add_arguments(subs.add_parser("bench", help="wake-word and latency benchmark: writes results/*.json"))


def main(args: argparse.Namespace, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    command = getattr(args, "housemate_command", None) or "status"
    try:
        return _COMMANDS[command](args, tools.client(), out)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"housemate {command} failed: {exc}", file=out)
        return 1
