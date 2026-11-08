from housemate.core.answers import (
    AnswerBook,
    PendingAnswer,
    apply_answer,
    make_marker,
    new_token,
    split_marker,
)


class Clock:
    def __init__(self) -> None:
        self.t = 50.0

    def __call__(self) -> float:
        return self.t


def entry(book_clock, token, *, device="dev-k", kind="approval", ttl=10.0, seen=None, note="applied"):
    def handle(transcript: str) -> str:
        if seen is not None:
            seen.append(transcript)
        return note

    return PendingAnswer(token=token, kind=kind, device_id=device, room="kitchen",
                         expires_at=book_clock() + ttl, handle=handle, key="req-1")


def test_marker_roundtrip_with_gateway_framing():
    token = new_token()
    assert len(token) == 32
    message = f'"Approve."\n\n{make_marker(token)}\n\n[New message]\n'
    assert split_marker(message) == (token, "Approve.")


def test_marker_with_hermes_framing_keeps_brackets_in_the_transcript():
    """Hermes prepends channel_context: ``<marker>\n\n[New message]\n"<transcript>"``. Brackets the
    speech-to-text put inside the transcript stay (review H1: "yes [no]" must not become "yes")."""
    token = new_token()
    message = f'{make_marker(token)}\n\n[New message]\n"yes [no]"'
    assert split_marker(message) == (token, "yes [no]")
    assert split_marker(f'{make_marker(token)}\n\n[New message]\n"Approve."') == (token, "Approve.")


def test_marker_with_a_gateway_note_only_is_an_empty_transcript():
    token = new_token()
    note = ("[The user sent a voice message but it came through empty or inaudible — speech-to-text "
            "returned no words. Do not guess at the content; ask the user to resend or type it out.]")
    assert split_marker(f"{make_marker(token)}\n\n[New message]\n{note}") == (token, "")


def test_marker_absent_or_malformed():
    assert split_marker("What is two plus two?") is None
    assert split_marker("[housemate-answer:short]") is None
    assert split_marker(None) is None


def test_hook_resolves_and_blocks_the_side_turn():
    clock = Clock()
    book = AnswerBook(clock)
    token = new_token()
    seen: list[str] = []
    book.add(entry(clock, token, seen=seen))
    out = apply_answer(book, {"platform": "housemate", "user_message": f'"Deny."\n\n{make_marker(token)}'})
    assert out == {"action": "block", "message": "(housemate: applied)"}
    assert seen == ["Deny."]
    # one use only: the same marker again resolves nothing but is still blocked
    again = apply_answer(book, {"platform": "housemate", "user_message": f'"Approve."\n\n{make_marker(token)}'})
    assert again["action"] == "block" and "too late" in again["message"]
    assert seen == ["Deny."]


def test_hook_ignores_other_platforms_and_plain_messages():
    clock = Clock()
    book = AnswerBook(clock)
    token = new_token()
    seen: list[str] = []
    book.add(entry(clock, token, seen=seen))
    assert apply_answer(book, {"platform": "telegram", "user_message": make_marker(token)}) is None
    assert apply_answer(book, {"platform": "housemate", "user_message": "hello"}) is None
    assert seen == []


def test_expired_answer_is_never_applied():
    clock = Clock()
    book = AnswerBook(clock)
    token = new_token()
    seen: list[str] = []
    book.add(entry(clock, token, ttl=10, seen=seen))
    clock.t += 11
    out = apply_answer(book, {"platform": "housemate", "user_message": f'"Approve." {make_marker(token)}'})
    assert out["action"] == "block"
    assert seen == []


def test_handler_error_still_blocks():
    clock = Clock()
    book = AnswerBook(clock)
    token = new_token()

    def boom(_):
        raise RuntimeError("resolver exploded")

    book.add(PendingAnswer(token=token, kind="approval", device_id="d", room="kitchen",
                           expires_at=clock() + 5, handle=boom))
    out = apply_answer(book, {"platform": "housemate", "user_message": make_marker(token)})
    assert out["action"] == "block" and "could not" in out["message"]


def test_book_for_device_and_drop():
    clock = Clock()
    book = AnswerBook(clock)
    a, b = new_token(), new_token()
    book.add(entry(clock, a, device="dev-k"))
    clock.t += 1
    book.add(entry(clock, b, device="dev-k", kind="stop"))
    # a pending approval outranks a newer "stop" for a ringing timer: approving stays possible by voice
    assert book.for_device("dev-k").token == a
    assert book.for_device("dev-o") is None
    book.drop(kind="approval")
    assert book.for_device("dev-k").token == b
    assert [e.token for e in book.drop(kind="stop")] == [b]
    assert book.for_device("dev-k") is None


def test_newest_entry_of_the_same_rank_wins():
    clock = Clock()
    book = AnswerBook(clock)
    a, b = new_token(), new_token()
    book.add(entry(clock, a, device="dev-k", kind="stop"))
    book.add(entry(clock, b, device="dev-k", kind="stop"))
    assert book.for_device("dev-k").token == b


def test_spoken_answer_transcript_is_not_logged_at_info(caplog):
    """Review L12: what was said is DEBUG only."""
    import logging

    clock = Clock()
    book = AnswerBook(clock)
    token = new_token()
    book.add(entry(clock, token))
    with caplog.at_level(logging.INFO, logger="housemate.core.answers"):
        apply_answer(book, {"platform": "housemate", "user_message": f'{make_marker(token)}\n\n[New message]\n"my secret words"'})
    assert caplog.records and all("secret" not in r.getMessage() for r in caplog.records)
