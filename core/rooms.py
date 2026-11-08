"""Live, in-memory view of which device is online in which room, and what it is doing."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .store import slugify

__all__ = ["DEVICE_STATES", "DeviceConn", "DeviceInfo", "DeviceState", "RoomRegistry", "RoomState"]

log = logging.getLogger(__name__)

DeviceState = Literal["idle", "listening", "thinking", "speaking", "ringing", "offline"]
DEVICE_STATES: tuple[str, ...] = ("idle", "listening", "thinking", "speaking", "ringing", "offline")

# The room shows its "loudest" device state.
_ROOM_PRIORITY = {"speaking": 5, "ringing": 4, "thinking": 3, "listening": 2, "idle": 1, "offline": 0}


@dataclass(eq=False)
class DeviceConn:
    """One online satellite. ``transport`` is opaque to the registry (the hub's session)."""

    device_id: str
    name: str
    room: str | None = None
    state: str = "idle"
    transport: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class DeviceInfo:
    device_id: str
    name: str
    state: str

    def to_dict(self) -> dict:
        return {"device_id": self.device_id, "name": self.name, "state": self.state}


@dataclass(frozen=True)
class RoomState:
    room: str
    state: str
    devices: tuple[DeviceInfo, ...]

    def to_dict(self) -> dict:
        return {"room": self.room, "state": self.state, "devices": [d.to_dict() for d in self.devices]}


class RoomRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conns: dict[str, DeviceConn] = {}
        self._listeners: list[Callable[[list[RoomState]], None]] = []

    def attach(self, device_id: str, room: str, conn: DeviceConn) -> None:
        slug = slugify(room)
        with self._lock:
            conn.room = slug
            self._conns[device_id] = conn
        self._changed()

    def detach(self, device_id: str) -> None:
        with self._lock:
            removed = self._conns.pop(device_id, None)
        if removed is not None:
            self._changed()

    def get(self, device_id: str) -> DeviceConn | None:
        with self._lock:
            return self._conns.get(device_id)

    def devices_in(self, room: str) -> list[DeviceConn]:
        slug = slugify(room)
        with self._lock:
            return [c for c in self._conns.values() if c.room == slug]

    def set_device_state(self, device_id: str, state: DeviceState) -> None:
        if state not in DEVICE_STATES:
            raise ValueError(f"unknown device state: {state!r}")
        with self._lock:
            conn = self._conns.get(device_id)
            if conn is None or conn.state == state:
                return
            conn.state = state
        self._changed()

    def snapshot(self) -> list[RoomState]:
        with self._lock:
            by_room: dict[str, list[DeviceConn]] = {}
            for conn in self._conns.values():
                by_room.setdefault(conn.room or "", []).append(conn)
            out = []
            for room in sorted(by_room):
                conns = sorted(by_room[room], key=lambda c: c.device_id)
                state = max((c.state for c in conns), key=lambda s: _ROOM_PRIORITY.get(s, 0))
                out.append(
                    RoomState(
                        room=room,
                        state=state,
                        devices=tuple(DeviceInfo(c.device_id, c.name, c.state) for c in conns),
                    )
                )
            return out

    def on_change(self, cb: Callable[[list[RoomState]], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def _changed(self) -> None:
        with self._lock:
            listeners = list(self._listeners)
        if not listeners:
            return
        snap = self.snapshot()
        for cb in listeners:
            try:
                cb(snap)
            except Exception:
                log.exception("housemate: room change listener failed")
