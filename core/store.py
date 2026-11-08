"""Persistent hub state: paired devices, pairing codes, rooms and timers.

All methods are synchronous and safe to call from any thread: the connection is
expected to be opened with ``check_same_thread=False`` (``plugin_db`` does this)
and every statement runs under one lock. Device keys are never stored, only
their SHA-256.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass

__all__ = ["Device", "HousemateStore", "Room", "TimerRow", "slugify"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    room       TEXT,
    key_hash   TEXT,
    created_at REAL NOT NULL,
    paired_at  REAL
);
CREATE TABLE IF NOT EXISTS pairing (
    code       TEXT PRIMARY KEY,
    device_id  TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS pairing_device ON pairing(device_id);
CREATE TABLE IF NOT EXISTS rooms (
    slug          TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    earshot_group TEXT
);
CREATE TABLE IF NOT EXISTS timers (
    id         TEXT PRIMARY KEY,
    room       TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'timer',
    label      TEXT NOT NULL DEFAULT '',
    due_at     REAL NOT NULL,
    created_at REAL NOT NULL,
    duration_s REAL,
    state      TEXT NOT NULL DEFAULT 'pending',
    fired_at   REAL
);
CREATE INDEX IF NOT EXISTS timers_state_due ON timers(state, due_at);
"""

# Compared against when a device has no key, so a miss costs the same as a hit.
_NO_KEY_HASH = "0" * 64


@dataclass(frozen=True)
class Device:
    device_id: str
    name: str
    room: str | None
    paired: bool
    created_at: float
    paired_at: float | None


@dataclass(frozen=True)
class Room:
    slug: str
    name: str
    earshot_group: str | None


@dataclass(frozen=True)
class TimerRow:
    """One row of the ``timers`` table. ``due_at``/``created_at``/``fired_at`` are UTC epoch seconds."""

    id: str
    room: str
    kind: str
    label: str
    due_at: float
    created_at: float
    duration_s: float | None
    state: str
    fired_at: float | None = None


_TIMER_COLS = "id, room, kind, label, due_at, created_at, duration_s, state, fired_at"


def slugify(name: str) -> str:
    """"Living Room" -> "living-room". Raises ValueError when nothing is left."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"not a usable room name: {name!r}")
    return slug


def _hash_key(device_key: str) -> str:
    return hashlib.sha256(device_key.encode("utf-8")).hexdigest()


class HousemateStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # --- devices -------------------------------------------------------------

    def upsert_device(self, device_id: str, name: str) -> Device:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO devices (device_id, name, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(device_id) DO UPDATE SET name = excluded.name",
                (device_id, name, time.time()),
            )
            return self._device(device_id)

    def get_device(self, device_id: str) -> Device | None:
        with self._lock:
            return self._device(device_id)

    def list_devices(self) -> list[Device]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT device_id, name, room, key_hash, created_at, paired_at "
                "FROM devices ORDER BY created_at, device_id"
            ).fetchall()
        return [self._row_to_device(r) for r in rows]

    def set_room(self, device_id: str, room: str) -> None:
        with self._lock, self._conn:
            if self._device(device_id) is None:
                raise KeyError(device_id)
            slug = self._ensure_room(room).slug
            self._conn.execute("UPDATE devices SET room = ? WHERE device_id = ?", (slug, device_id))

    def verify_device(self, device_id: str, device_key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT key_hash FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
        stored = row[0] if row and row[0] else None
        given = _hash_key(device_key or "")
        match = hmac.compare_digest(given, stored or _NO_KEY_HASH)
        return bool(stored) and match

    def revoke_device(self, device_id: str) -> bool:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM pairing WHERE device_id = ?", (device_id,))
            cur = self._conn.execute("DELETE FROM devices WHERE device_id = ?", (device_id,))
            return cur.rowcount > 0

    # --- pairing ---------------------------------------------------------------

    def create_pairing_code(self, device_id: str, now: float, ttl_s: int = 600) -> str:
        """A fresh 6-digit code for ``device_id``; any older code for it is dropped."""
        with self._lock, self._conn:
            self._purge_expired(now)
            self._conn.execute("DELETE FROM pairing WHERE device_id = ?", (device_id,))
            while True:
                code = f"{secrets.randbelow(1_000_000):06d}"
                taken = self._conn.execute("SELECT 1 FROM pairing WHERE code = ?", (code,)).fetchone()
                if not taken:
                    break
            self._conn.execute(
                "INSERT INTO pairing (code, device_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (code, device_id, now, now + ttl_s),
            )
            return code

    def approve_pairing(self, code: str, now: float) -> tuple[str, str] | None:
        """Consume ``code``; returns (device_id, device_key) or None if unknown/expired."""
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT device_id, expires_at FROM pairing WHERE code = ?", (code,)
            ).fetchone()
            if row is None:
                return None
            device_id, expires_at = row
            if now >= expires_at:
                self._conn.execute("DELETE FROM pairing WHERE code = ?", (code,))
                return None
            device_key = secrets.token_urlsafe(32)
            self._conn.execute(
                "INSERT INTO devices (device_id, name, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(device_id) DO NOTHING",
                (device_id, device_id, now),
            )
            self._conn.execute(
                "UPDATE devices SET key_hash = ?, paired_at = ? WHERE device_id = ?",
                (_hash_key(device_key), now, device_id),
            )
            self._conn.execute("DELETE FROM pairing WHERE device_id = ?", (device_id,))
            return device_id, device_key

    def cancel_pairing(self, code: str) -> None:
        """Drop a pending code; an unpaired device that only existed for it is forgotten."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT device_id FROM pairing WHERE code = ?", (code,)).fetchone()
            if row is None:
                return
            self._conn.execute("DELETE FROM pairing WHERE code = ?", (code,))
            self._conn.execute(
                "DELETE FROM devices WHERE device_id = ? AND key_hash IS NULL", (row[0],)
            )

    # --- rooms -----------------------------------------------------------------

    def ensure_room(self, name: str) -> Room:
        with self._lock, self._conn:
            return self._ensure_room(name)

    def list_rooms(self) -> list[Room]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT slug, name, earshot_group FROM rooms ORDER BY slug"
            ).fetchall()
        return [Room(*r) for r in rows]

    def set_earshot_group(self, room: str, group: str | None) -> None:
        with self._lock, self._conn:
            slug = self._ensure_room(room).slug
            value = group.strip() if isinstance(group, str) and group.strip() else None
            self._conn.execute("UPDATE rooms SET earshot_group = ? WHERE slug = ?", (value, slug))

    # --- timers --------------------------------------------------------------------

    def add_timer(self, row: TimerRow) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                f"INSERT INTO timers ({_TIMER_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row.id, row.room, row.kind, row.label, row.due_at, row.created_at,
                 row.duration_s, row.state, row.fired_at),
            )

    def get_timer(self, timer_id: str) -> TimerRow | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_TIMER_COLS} FROM timers WHERE id = ?", (timer_id,)
            ).fetchone()
        return TimerRow(*row) if row else None

    def list_timers(
        self,
        *,
        states: tuple[str, ...] | list[str] | None = None,
        room: str | None = None,
        due_before: float | None = None,
    ) -> list[TimerRow]:
        """Timers ordered by due time, then creation order. ``due_before`` is inclusive."""
        where, args = [], []
        if states is not None:
            states = tuple(states)
            if not states:
                return []
            where.append(f"state IN ({', '.join('?' * len(states))})")
            args.extend(states)
        if room is not None:
            where.append("room = ?")
            args.append(room)
        if due_before is not None:
            where.append("due_at <= ?")
            args.append(due_before)
        sql = f"SELECT {_TIMER_COLS} FROM timers"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY due_at, created_at, rowid"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [TimerRow(*r) for r in rows]

    def transition_timer(
        self, timer_id: str, from_states: tuple[str, ...], to_state: str, *, fired_at: float | None = None
    ) -> bool:
        """Compare-and-set the state. False when the timer is gone or not in ``from_states``,
        which is what makes "ring exactly once" hold across callers and restarts."""
        from_states = tuple(from_states)
        if not from_states:
            return False
        with self._lock, self._conn:
            cur = self._conn.execute(
                f"UPDATE timers SET state = ?, fired_at = COALESCE(?, fired_at) "
                f"WHERE id = ? AND state IN ({', '.join('?' * len(from_states))})",
                (to_state, fired_at, timer_id, *from_states),
            )
            return cur.rowcount == 1

    # --- internals (call with the lock held) -------------------------------------------

    def _ensure_room(self, name: str) -> Room:
        slug = slugify(name)
        self._conn.execute(
            "INSERT INTO rooms (slug, name) VALUES (?, ?) ON CONFLICT(slug) DO NOTHING",
            (slug, str(name).strip()),
        )
        row = self._conn.execute(
            "SELECT slug, name, earshot_group FROM rooms WHERE slug = ?", (slug,)
        ).fetchone()
        return Room(*row)

    def _device(self, device_id: str) -> Device | None:
        row = self._conn.execute(
            "SELECT device_id, name, room, key_hash, created_at, paired_at "
            "FROM devices WHERE device_id = ?",
            (device_id,),
        ).fetchone()
        return self._row_to_device(row) if row else None

    @staticmethod
    def _row_to_device(row) -> Device:
        device_id, name, room, key_hash, created_at, paired_at = row
        return Device(
            device_id=device_id,
            name=name,
            room=room,
            paired=bool(key_hash),
            created_at=created_at,
            paired_at=paired_at,
        )

    def _purge_expired(self, now: float) -> None:
        expired = self._conn.execute(
            "SELECT code FROM pairing WHERE expires_at <= ?", (now,)
        ).fetchall()
        for (code,) in expired:
            row = self._conn.execute("SELECT device_id FROM pairing WHERE code = ?", (code,)).fetchone()
            self._conn.execute("DELETE FROM pairing WHERE code = ?", (code,))
            if row:
                self._conn.execute(
                    "DELETE FROM devices WHERE device_id = ? AND key_hash IS NULL "
                    "AND NOT EXISTS (SELECT 1 FROM pairing WHERE device_id = ?)",
                    (row[0], row[0]),
                )
