"""Which conversation an utterance joins, where the reply goes, and the household queue.

``household`` mode (default): every room shares one chat (``"household"``) and so one Hermes
session; "start in the kitchen, carry on in the office" just works. ``per_room`` mode gives each
room its own chat id (the room slug).

A *turn* runs from the moment an utterance is handed to Hermes until its reply has been delivered
(the adapter calls ``on_turn_end`` on the final ``send()`` whose ``reply_to`` is the utterance's
message id, on ``on_processing_complete`` for it, or after ``turn_timeout_s`` as a safety net; a
turn waiting on an approval or clarify prompt is held, never timed out).
While a chat has a turn running, new utterances for that chat are queued here instead of being
handed to Hermes: in a shared household session a second message mid-turn would interrupt the
running turn (and cancel a pending approval). ``on_turn_end`` hands back the next queued
utterance and reserves the chat for it, so nothing can slip in before its turn begins.

Pure and synchronous: call it from the adapter's event loop only.
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable

from .store import slugify

__all__ = ["HOUSEHOLD_CHAT", "SessionRouter", "Turn"]

HOUSEHOLD_CHAT = "household"
_MODES = ("household", "per_room")


@dataclass
class Turn:
    chat_id: str
    room: str
    message_id: str
    device_id: str | None
    started_at: float
    audio: bool = False  # spoken audio reached a device for this turn
    stream_open: bool = False  # Hermes opened a TTS stream for it (audio may still come)
    reserved: bool = False  # released from the queue, not yet handed to Hermes
    ended_at: float | None = None
    clock: Callable[[], float] | None = None

    def age_s(self) -> float:
        """Seconds since the turn ended (0 while it runs)."""
        if self.ended_at is None or self.clock is None:
            return 0.0
        return self.clock() - self.ended_at


@dataclass
class _Queued:
    item: Any
    at: float


class SessionRouter:
    def __init__(
        self,
        mode: str = "household",
        *,
        clock: Callable[[], float] = time.monotonic,
        turn_timeout_s: float = 360.0,
        queue_ttl_s: float = 300.0,
        max_queue: int = 5,
        history: int = 64,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"conversation mode must be one of {', '.join(_MODES)}, not {mode!r}")
        self.mode = mode
        self._clock = clock
        self.turn_timeout_s = float(turn_timeout_s)
        self.queue_ttl_s = float(queue_ttl_s)
        self.max_queue = max(1, int(max_queue))
        self._history = max(4, int(history))
        self._active: dict[str, Turn] = {}
        self._last: dict[str, Turn] = {}
        self._turns: OrderedDict[str, Turn] = OrderedDict()  # message id -> turn (recent)
        self._queues: dict[str, deque[_Queued]] = {}
        self.dropped: list[Any] = []  # queued utterances given up on (overflow, too old): tell the room
        self.expired: list[Turn] = []  # turns ended by the safety timeout

    # --- routing -------------------------------------------------------------------------------

    def chat_id_for(self, room: str) -> str:
        """``"household"`` in household mode, else the room's slug."""
        if self.mode == "household":
            return HOUSEHOLD_CHAT
        return slugify(room)

    def begin_turn(self, chat_id: str, room: str, message_id: str, device_id: str | None = None) -> Turn:
        turn = Turn(chat_id, slugify(room), message_id, device_id, self._clock(), clock=self._clock)
        self._active[chat_id] = turn
        self._last[chat_id] = turn
        self._turns[message_id] = turn
        self._turns.move_to_end(message_id)
        while len(self._turns) > self._history:
            self._turns.popitem(last=False)
        return turn

    def active_turn(self, chat_id: str) -> Turn | None:
        return self._active.get(chat_id)

    def turn_for(self, message_id: str | None) -> Turn | None:
        if not message_id:
            return None
        return self._turns.get(message_id)

    def latest_turn(self, chat_id: str) -> Turn | None:
        return self._active.get(chat_id) or self._last.get(chat_id)

    def reply_room(self, chat_id: str, reply_to: str | None) -> str | None:
        """The room of ``reply_to``'s turn, else the chat's latest turn's room, else None."""
        turn = self._reply_turn(chat_id, reply_to)
        return turn.room if turn is not None else None

    def reply_device(self, chat_id: str, reply_to: str | None) -> str | None:
        """The device that asked (it answers; a second device in the same room stays quiet)."""
        turn = self._reply_turn(chat_id, reply_to)
        return turn.device_id if turn is not None else None

    def _reply_turn(self, chat_id: str, reply_to: str | None) -> Turn | None:
        turn = self.turn_for(reply_to)
        if turn is not None and turn.chat_id == chat_id:
            return turn
        return self.latest_turn(chat_id)

    # --- the queue -------------------------------------------------------------------------------

    def household_busy(self, chat_id: str) -> bool:
        """A turn is running (or reserved) in this chat: new utterances must queue."""
        return chat_id in self._active

    def enqueue(self, item: Any, chat_id: str | None = None) -> None:
        if chat_id is None:
            chat_id = item.source.chat_id
        queue = self._queues.setdefault(chat_id, deque())
        queue.append(_Queued(item, self._clock()))
        while len(queue) > self.max_queue:
            self.dropped.append(queue.popleft().item)

    def take_dropped(self) -> tuple[list[Any], list[Turn]]:
        """Utterances dropped from the queue and turns that timed out since the last call."""
        dropped, expired = self.dropped, self.expired
        self.dropped, self.expired = [], []
        return dropped, expired

    def queued(self, chat_id: str) -> int:
        return len(self._queues.get(chat_id) or ())

    def on_turn_end(self, chat_id: str, message_id: str | None = None) -> Any | None:
        """End the chat's running turn and return the next queued utterance (or None).

        With ``message_id`` only that turn ends: a late signal from an older turn is ignored.
        A returned item keeps the chat busy until ``begin_turn`` is called for it.
        """
        turn = self._active.get(chat_id)
        if turn is None:
            return None
        if message_id is not None and turn.message_id != message_id:
            return None
        del self._active[chat_id]
        turn.ended_at = self._clock()
        return self._release(chat_id)

    def expire(self, hold: Callable[[str], bool] | None = None) -> list[tuple[str, Any]]:
        """End turns older than ``turn_timeout_s``; returns ``(chat_id, next_item)`` released.

        ``hold(chat_id)`` true: the chat's turn is waiting on the user (an open approval) and never
        times out; its clock restarts, so it gets a full timeout once the prompt is settled.
        """
        now = self._clock()
        released: list[tuple[str, Any]] = []
        for chat_id, turn in list(self._active.items()):
            if hold is not None and hold(chat_id):
                turn.started_at = now
                continue
            if now - turn.started_at >= self.turn_timeout_s:
                del self._active[chat_id]
                turn.ended_at = now
                self.expired.append(turn)
                item = self._release(chat_id)
                if item is not None:
                    released.append((chat_id, item))
        return released

    def _release(self, chat_id: str) -> Any | None:
        queue = self._queues.get(chat_id)
        now = self._clock()
        while queue:
            entry = queue.popleft()
            if now - entry.at > self.queue_ttl_s:
                self.dropped.append(entry.item)  # asked too long ago to answer now
                continue
            item = entry.item
            message_id = getattr(item, "message_id", None) or f"queued-{id(item)}"
            room = getattr(item, "room", None) or "unknown"
            reserved = Turn(chat_id, room, message_id, getattr(item, "device_id", None), now, reserved=True)
            self._active[chat_id] = reserved
            return item
        return None
