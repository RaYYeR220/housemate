"""Tiny deterministic parsers for spoken answers: yes/no, "the second one", "stop".

Used where a full agent turn would be wrong or too slow: answering an approval prompt, picking a
clarify choice, silencing a ringing timer. Everything here fails closed: only a short utterance that
is entirely an approve phrase reads as approve, and anything ambiguous returns None so the caller
can ask again or let the prompt time out.
"""

from __future__ import annotations

import re
from typing import Literal, Sequence

__all__ = ["is_skip", "is_stop", "normalize", "parse_choice", "parse_yes_no", "spoken_label"]

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_NOT_WORD = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    """Lowercase, straight apostrophes, punctuation to spaces, single spaces."""
    s = str(text or "").translate(_APOSTROPHES).lower()
    s = _NOT_WORD.sub(" ", s)
    s = " ".join(tok.strip("'") for tok in s.split())
    return " ".join(s.split())


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


# --- yes / no ----------------------------------------------------------------------------------

# The whole answer must be one of these (plus a little filler), like ``is_stop``. A word anywhere in a
# longer sentence never counts: "yes, but what will it delete?" and "your approve timer is done" are
# not answers. "yeah" is left out on purpose: it is what people (and TVs) say in passing.
_APPROVE = [
    ("approve",), ("approved",), ("yes",), ("yes", "please"), ("do", "it"), ("go", "ahead"), ("confirm",),
]
_DENY = [
    ("deny",), ("denied",), ("no",), ("nope",), ("nah",), ("don't",), ("dont",), ("do", "not"),
    ("don't", "do", "it"), ("dont", "do", "it"), ("do", "not", "do", "it"), ("stop",), ("cancel",),
    ("cancel", "that"), ("cancel", "it"), ("no", "thanks"), ("no", "thank", "you"),
]
_YES_NO_FILLER = {"ok", "okay", "please", "hey", "hermes", "jarvis"}
MAX_ANSWER_WORDS = 4


def _covered_by(tokens: list[str], phrases: list[tuple[str, ...]]) -> bool:
    """Is ``tokens`` exactly a run of ``phrases`` (longest match first)?"""
    ordered = sorted(phrases, key=len, reverse=True)
    i = 0
    while i < len(tokens):
        for phrase in ordered:
            if tuple(tokens[i : i + len(phrase)]) == phrase:
                i += len(phrase)
                break
        else:
            return False
    return True


def parse_yes_no(text: str) -> Literal["approve", "deny"] | None:
    """"Approve." -> approve, "no thanks" -> deny, anything else -> None.

    Strict: at most four words, no question, and every word (after "ok", "please" and the wake
    phrase) part of an approve phrase, or every word part of a deny phrase. Hedged ("yes, but…",
    "wait"), mixed ("yes no", "don't approve"), questions and longer sentences are None, so the
    caller asks again and in the end lets the prompt time out (deny).
    """
    raw = str(text or "")
    if "?" in raw:
        return None
    tokens = _tokens(raw)
    if not tokens or len(tokens) > MAX_ANSWER_WORDS:
        return None
    words = [t for t in tokens if t not in _YES_NO_FILLER]
    if not words:
        return None
    if _covered_by(words, _APPROVE):
        return "approve"
    if _covered_by(words, _DENY):
        return "deny"
    return None


# --- numbered / labelled choices -----------------------------------------------------------------

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
    **{f"{n}{suffix}": n for n, suffix in zip(range(1, 10), ("st", "nd", "rd", "th", "th", "th", "th", "th", "th"))},
}
_CARDINALS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    **{str(n): n for n in range(1, 10)},
}
_CHOICE_FILLER = {
    "the", "a", "an", "option", "number", "choice", "item", "please", "go", "with", "pick", "choose",
    "i", "i'll", "ill", "take", "want", "that", "is", "it", "let's", "lets", "say", "answer", "my",
    "its", "it's", "um", "uh", "ok", "okay",
}


def _spoken_number(tokens: list[str]) -> int | None:
    rest = [t for t in tokens if t not in _CHOICE_FILLER]
    if any(t in _ORDINALS for t in rest):
        rest = [t for t in rest if t not in ("one", "ones")]  # "the second one"
    if len(rest) != 1:
        return None
    return _ORDINALS.get(rest[0]) or _CARDINALS.get(rest[0])


_RECOMMENDED = re.compile(r"\s*\(recommended\)\s*$", re.I)


def spoken_label(choice: str) -> str:
    """A choice as it is read aloud: without Hermes's "(Recommended)" tag."""
    return _RECOMMENDED.sub("", str(choice)).strip()


_SKIP_FILLER = {"please", "it", "that", "this", "the", "question", "one", "just", "ok", "okay", "lets", "let's"}


def is_skip(text: str) -> bool:
    """A spoken "skip" for a clarify question ("Skip.", "skip it", "please skip this question")."""
    words = _tokens(text)
    return "skip" in words and all(w == "skip" or w in _SKIP_FILLER for w in words)


def parse_choice(text: str, choices: Sequence[str]) -> str | None:
    """Which of ``choices`` was said: exact label, a number/ordinal (1-9), or an unambiguous
    label prefix ("pizza" -> "Pizza with mushrooms", "pasta please" -> "Pasta")."""
    said = normalize(text)
    if not said or not choices:
        return None
    labels = [(choice, normalize(spoken_label(choice))) for choice in choices]

    exact = [choice for choice, label in labels if label and label == said]
    if len(exact) == 1:
        return exact[0]

    number = _spoken_number(said.split())
    if number is not None:
        return choices[number - 1] if number <= len(choices) else None

    prefixed = [
        choice for choice, label in labels
        if label and (label.startswith(said + " ") or said.startswith(label + " "))
    ]
    return prefixed[0] if len(prefixed) == 1 else None


# --- stop ------------------------------------------------------------------------------------------

_STOP_PHRASES = [
    ("turn", "it", "off"), ("turn", "that", "off"), ("turn", "off"), ("switch", "it", "off"),
    ("switch", "off"), ("shut", "up"), ("be", "quiet"), ("that's", "enough"), ("thats", "enough"),
    ("that", "is", "enough"),
]
_STOP_WORDS = {"stop", "quiet", "silence", "enough", "cancel", "dismiss", "off"}
_STOP_FILLER = {
    "hey", "hi", "hermes", "jarvis", "ok", "okay", "please", "now", "it", "that", "this", "the",
    "timer", "timers", "alarm", "alarms", "music", "ringing", "sound", "noise", "thanks", "thank",
    "you", "all", "right", "alright", "oh", "um", "uh", "just", "already",
}


def is_stop(text: str) -> bool:
    """A short "stop" utterance ("OK, stop.", "hey hermes stop", "turn it off", "be quiet").

    Everything said must be stop words or filler, so "bus stop times" and "don't stop" are not.
    """
    tokens = _tokens(text)
    for phrase in _STOP_PHRASES:
        n = len(phrase)
        i = 0
        while i <= len(tokens) - n:
            if tuple(tokens[i : i + n]) == phrase:
                tokens[i : i + n] = ["stop"]
            i += 1
    rest = [t for t in tokens if t not in _STOP_FILLER]
    return bool(rest) and all(t in _STOP_WORDS for t in rest)
