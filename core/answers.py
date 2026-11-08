"""Spoken answers to prompts (approve/deny, a clarify choice, "stop"), using public surfaces only.

While Hermes waits on a dangerous-command approval, a voice message in the same session cannot
answer it: the gateway routes busy messages *before* transcription (the text is still empty), so
the voice note interrupts the turn and the approval is cancelled. So the adapter sends the answer
clip somewhere else: a VOICE event into a separate side chat (``SIDE_CHAT``) whose
``channel_context`` carries a one-time marker ``[housemate-answer:<token>]``. Hermes transcribes it
with the user's own STT settings; our ``pre_llm_call`` hook (``pre_llm_call`` below) finds the
marker, parses the transcript, applies the answer (``resolve_gateway_approval_card``,
``resolve_gateway_clarify``, stop a ringing timer) and vetoes the side turn, so no model call is
made.

Fail closed: an expired or unknown token, or an answer that is not clearly "approve", never
approves anything; Hermes's own approval timeout then denies.
"""

from __future__ import annotations

import logging
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "BOOK",
    "SIDE_CHAT",
    "AnswerBook",
    "apply_answer",
    "last_llm_call_after",
    "PendingAnswer",
    "make_marker",
    "new_token",
    "pre_llm_call",
    "split_marker",
]

log = logging.getLogger(__name__)

SIDE_CHAT = "housemate-answers"
PLATFORM = "housemate"
_MARKER_RE = re.compile(r"\[housemate-answer:([0-9a-f]{32})\]")
_BRACKETED = re.compile(r"\[[^\]]{0,200}\]")
_QUOTES = "\"'“”‘’«» \n\t"


def new_token() -> str:
    return secrets.token_hex(16)


def make_marker(token: str) -> str:
    return f"[housemate-answer:{token}]"


def split_marker(text: Any) -> tuple[str, str] | None:
    """``(token, transcript)`` when ``text`` carries an answer marker, else None.

    The gateway hands the hook something like ``"Approve."\\n\\n[housemate-answer:…]\\n\\n[New
    message]``: the transcript (quoted) plus the channel context. Bracketed notes are dropped.
    """
    if not isinstance(text, str):
        return None
    match = _MARKER_RE.search(text)
    if match is None:
        return None
    rest = _BRACKETED.sub(" ", text)
    transcript = " ".join(rest.split()).strip(_QUOTES)
    return match.group(1), transcript


@dataclass
class PendingAnswer:
    """One prompt waiting for a spoken answer from one device."""

    token: str
    kind: str  # "approval" | "clarify" | "stop"
    device_id: str
    room: str
    expires_at: float  # time.monotonic()
    handle: Callable[[str], str]  # transcript -> short note for the log; runs in the hook's thread
    key: str = ""  # what it answers (approval request id, clarify id, room)
    created_at: float = field(default_factory=time.monotonic)


class AnswerBook:
    """Thread-safe: the adapter (event loop) adds entries, the hook (agent thread) takes them."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, PendingAnswer] = {}

    def add(self, entry: PendingAnswer) -> None:
        with self._lock:
            self._purge()
            self._entries[entry.token] = entry

    def take(self, token: str) -> PendingAnswer | None:
        """Remove and return the entry, or None when unknown or expired (one use only)."""
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is None or entry.expires_at <= self._clock():
            return None
        return entry

    def for_device(self, device_id: str) -> PendingAnswer | None:
        """The newest live entry waiting on this device."""
        with self._lock:
            self._purge()
            live = [e for e in self._entries.values() if e.device_id == device_id]
        return live[-1] if live else None  # insertion order: the newest prompt wins

    def drop(self, *, token: str | None = None, key: str | None = None, kind: str | None = None,
             device_id: str | None = None, room: str | None = None) -> list[PendingAnswer]:
        """Remove every entry matching all the given fields."""
        with self._lock:
            gone = [
                e for e in self._entries.values()
                if (token is None or e.token == token)
                and (key is None or e.key == key)
                and (kind is None or e.kind == kind)
                and (device_id is None or e.device_id == device_id)
                and (room is None or e.room == room)
            ]
            for e in gone:
                self._entries.pop(e.token, None)
        return gone

    def entries(self) -> list[PendingAnswer]:
        with self._lock:
            self._purge()
            return list(self._entries.values())

    def _purge(self) -> None:
        now = self._clock()
        for token in [t for t, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[token]


BOOK = AnswerBook()


_llm_calls: deque[float] = deque(maxlen=64)


def last_llm_call_after(t: float) -> float | None:
    """Wall-clock time of the first Housemate model call at or after ``t`` (latency breakdown)."""
    for at in list(_llm_calls):
        if at >= t:
            return at
    return None


def pre_llm_call(**kwargs: Any) -> dict | None:
    """``pre_llm_call`` hook: apply a spoken answer from the side chat and veto that turn.

    Returns None (no opinion) for every message that is not a Housemate answer.
    """
    return apply_answer(BOOK, kwargs)


def apply_answer(book: AnswerBook, kwargs: dict) -> dict | None:
    if str(kwargs.get("platform") or "").strip().lower() != PLATFORM:
        return None
    found = split_marker(kwargs.get("user_message"))
    if found is None:
        _llm_calls.append(time.time())
        return None
    token, transcript = found
    entry = book.take(token)
    if entry is None:
        note = "that answer came too late"
    else:
        try:
            note = entry.handle(transcript) or "ok"
        except Exception:
            log.exception("housemate: applying a spoken %s answer failed", entry.kind)
            note = "the answer could not be applied"
    log.info("housemate: spoken answer %r -> %s", transcript[:80], note)
    return {"action": "block", "message": f"(housemate: {note})"}
