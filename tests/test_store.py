import hashlib
import sqlite3
import threading

import pytest

from housemate.core.rooms import DeviceConn, RoomRegistry
from housemate.core.store import HousemateStore, slugify

T0 = 1_700_000_000.0


@pytest.fixture
def store():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    yield HousemateStore(conn)
    conn.close()


# --- pairing -----------------------------------------------------------------


def test_pairing_roundtrip(store):
    store.upsert_device("dev-kitchen", "Kitchen phone")
    code = store.create_pairing_code("dev-kitchen", now=T0)
    assert len(code) == 6 and code.isdigit()

    result = store.approve_pairing(code, now=T0 + 5)
    assert result is not None
    device_id, key = result
    assert device_id == "dev-kitchen"
    assert len(key) >= 32

    assert store.verify_device("dev-kitchen", key) is True
    assert store.verify_device("dev-kitchen", key + "x") is False
    assert store.verify_device("dev-other", key) is False


def test_pairing_code_expires(store):
    store.upsert_device("dev-kitchen", "Kitchen phone")
    code = store.create_pairing_code("dev-kitchen", now=T0)
    assert store.approve_pairing(code, now=T0 + 601) is None


def test_pairing_code_is_single_use(store):
    store.upsert_device("dev-kitchen", "Kitchen phone")
    code = store.create_pairing_code("dev-kitchen", now=T0)
    assert store.approve_pairing(code, now=T0 + 1) is not None
    assert store.approve_pairing(code, now=T0 + 2) is None


def test_new_code_replaces_previous_code_for_device(store):
    store.upsert_device("dev-kitchen", "Kitchen phone")
    first = store.create_pairing_code("dev-kitchen", now=T0)
    second = store.create_pairing_code("dev-kitchen", now=T0 + 1)
    if first != second:
        assert store.approve_pairing(first, now=T0 + 2) is None
    assert store.approve_pairing(second, now=T0 + 2) is not None


def test_device_key_stored_only_as_hash(store):
    store.upsert_device("dev-kitchen", "Kitchen phone")
    code = store.create_pairing_code("dev-kitchen", now=T0)
    _, key = store.approve_pairing(code, now=T0 + 1)
    rows = store._conn.execute("SELECT * FROM devices").fetchall()
    flat = [str(v) for row in rows for v in row]
    assert key not in flat
    assert hashlib.sha256(key.encode()).hexdigest() in flat


def test_revoke_blocks_verify(store):
    store.upsert_device("dev-kitchen", "Kitchen phone")
    code = store.create_pairing_code("dev-kitchen", now=T0)
    _, key = store.approve_pairing(code, now=T0 + 1)
    assert store.revoke_device("dev-kitchen") is True
    assert store.verify_device("dev-kitchen", key) is False
    assert store.revoke_device("dev-kitchen") is False


def test_cancel_pairing_forgets_unpaired_device(store):
    store.upsert_device("dev-x", "Stranger")
    code = store.create_pairing_code("dev-x", now=T0)
    store.cancel_pairing(code)
    assert store.approve_pairing(code, now=T0 + 1) is None
    assert store.get_device("dev-x") is None


def test_cancel_pairing_keeps_paired_device(store):
    store.upsert_device("dev-k", "Kitchen")
    _, key = store.approve_pairing(store.create_pairing_code("dev-k", now=T0), now=T0 + 1)
    code = store.create_pairing_code("dev-k", now=T0 + 2)
    store.cancel_pairing(code)
    assert store.verify_device("dev-k", key) is True


# --- devices and rooms ----------------------------------------------------------


def test_upsert_device_updates_name_and_lists(store):
    store.upsert_device("dev-a", "Old name")
    dev = store.upsert_device("dev-a", "New name")
    assert dev.device_id == "dev-a" and dev.name == "New name"
    assert dev.paired is False and dev.room is None
    assert [d.device_id for d in store.list_devices()] == ["dev-a"]


def test_slugify_room_names():
    assert slugify("Kitchen") == "kitchen"
    assert slugify("Living Room") == "living-room"
    assert slugify("  Kid's   Room!! ") == "kid-s-room"
    with pytest.raises(ValueError):
        slugify("  !! ")


def test_ensure_room_is_idempotent(store):
    room = store.ensure_room("Living Room")
    assert room.slug == "living-room" and room.name == "Living Room"
    again = store.ensure_room("living room")
    assert again.slug == "living-room" and again.name == "Living Room"
    assert [r.slug for r in store.list_rooms()] == ["living-room"]


def test_set_room_and_earshot_group(store):
    store.upsert_device("dev-a", "Tablet")
    store.set_room("dev-a", "Living Room")
    assert store.get_device("dev-a").room == "living-room"

    store.set_earshot_group("Living Room", "downstairs")
    assert store.ensure_room("living-room").earshot_group == "downstairs"
    store.set_earshot_group("living-room", None)
    assert store.ensure_room("living-room").earshot_group is None


def test_remove_room_unassigns_devices(store):
    store.upsert_device("dev-a", "Tablet")
    store.set_room("dev-a", "Living Room")
    assert store.get_room("LIVING ROOM").slug == "living-room"
    assert store.remove_room("living room") is True
    assert store.get_room("living-room") is None and store.list_rooms() == []
    assert store.get_device("dev-a").room is None
    assert store.remove_room("living room") is False
    assert store.get_room("!!!") is None


def test_set_room_unknown_device_raises(store):
    with pytest.raises(KeyError):
        store.set_room("nope", "Kitchen")


def test_schema_has_timers_table(store):
    cols = {row[1] for row in store._conn.execute("PRAGMA table_info(timers)")}
    assert {"id", "room", "kind", "label", "due_at", "created_at", "state"} <= cols


def test_store_survives_concurrent_threads(store):
    errors = []

    def worker(n):
        try:
            for i in range(25):
                store.upsert_device(f"dev-{n}-{i}", "x")
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.list_devices()) == 100


# --- live room registry ------------------------------------------------------------


def _conn(device_id, name="dev"):
    return DeviceConn(device_id=device_id, name=name)


def test_registry_attach_and_devices_in():
    reg = RoomRegistry()
    a, b = _conn("a"), _conn("b")
    reg.attach("a", "kitchen", a)
    reg.attach("b", "Kitchen", b)
    assert {c.device_id for c in reg.devices_in("kitchen")} == {"a", "b"}
    assert reg.devices_in("office") == []
    assert reg.get("a") is a and a.room == "kitchen"

    reg.detach("a")
    assert [c.device_id for c in reg.devices_in("kitchen")] == ["b"]
    assert reg.get("a") is None


def test_registry_attach_moves_device_between_rooms():
    reg = RoomRegistry()
    a = _conn("a")
    reg.attach("a", "kitchen", a)
    reg.attach("a", "office", a)
    assert reg.devices_in("kitchen") == []
    assert reg.devices_in("office") == [a]


def test_registry_on_change_fires_on_attach_state_and_detach():
    reg = RoomRegistry()
    seen = []
    reg.on_change(lambda snap: seen.append(snap))

    reg.attach("a", "kitchen", _conn("a", "Kitchen phone"))
    reg.set_device_state("a", "speaking")
    reg.set_device_state("a", "speaking")  # unchanged: no event
    reg.detach("a")
    reg.detach("a")  # unknown: no event

    assert len(seen) == 3
    kitchen = seen[1][0]
    assert kitchen.room == "kitchen" and kitchen.state == "speaking"
    assert kitchen.devices[0].device_id == "a"
    assert kitchen.devices[0].state == "speaking"
    assert seen[2] == []


def test_registry_snapshot_room_state_priority():
    reg = RoomRegistry()
    reg.attach("a", "kitchen", _conn("a"))
    reg.attach("b", "kitchen", _conn("b"))
    reg.set_device_state("b", "listening")
    (room,) = reg.snapshot()
    assert room.state == "listening"
    assert room.to_dict()["devices"][0]["device_id"] in {"a", "b"}


def test_registry_rejects_unknown_state():
    reg = RoomRegistry()
    reg.attach("a", "kitchen", _conn("a"))
    with pytest.raises(ValueError):
        reg.set_device_state("a", "dancing")


def test_registry_callback_errors_do_not_break_registry():
    reg = RoomRegistry()

    def boom(_):
        raise RuntimeError("listener bug")

    reg.on_change(boom)
    reg.attach("a", "kitchen", _conn("a"))
    assert reg.devices_in("kitchen")
