import pytest

from housemate.core.intents import is_skip, is_stop, parse_choice, parse_yes_no


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Approve.", "approve"),
        ("yes", "approve"),
        ("Yes, do it!", "approve"),
        ("yes please", "approve"),
        ("Approved.", "approve"),
        ("OK, approve.", "approve"),
        ("hey jarvis approve", "approve"),
        ("go ahead", "approve"),
        ("Confirm", "approve"),
        ("no thanks", "deny"),
        ("Deny.", "deny"),
        ("nope", "deny"),
        ("Don't.", "deny"),
        ("don't do it", "deny"),
        ("stop", "deny"),
        ("cancel that", "deny"),
        ("maybe", None),
        ("", None),
        ("what was the question", None),
        ("yes no", None),
        ("No, don't do it.", "deny"),
        ("no thank you", "deny"),
    ],
)
def test_yes_no_variants(text, expected):
    assert parse_yes_no(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        # review H1: hedged answers, questions back, and our own ring text
        "Yes, but what will it delete first?",
        "Yeah, wait, what does that do?",
        "Hold on, yeah, let me think",
        "Do it later",
        "Confirm what it does first",
        "Your approve timer is done.",
        "Your go ahead timer is done.",
        # more of the same shape
        "Yes?",
        "approve?",
        "yeah",
        "yeah thanks",
        "yes but",
        "wait, yes",
        "what",
        "let me think",
        "no wait yes",
        "don't approve",
        "yes please go ahead now",
        "approve approve approve approve approve",
        "ok",
        "please",
        "yes [no]",
    ],
)
def test_yes_no_is_strict(text):
    """Only a short, whole-utterance approve phrase approves; anything else is not an answer."""
    assert parse_yes_no(text) is None


def test_yes_no_ignores_case_and_curly_apostrophes():
    assert parse_yes_no("DON’T") == "deny"
    assert parse_yes_no("  YES PLEASE  ") == "approve"


CHOICES = ["Pasta", "Pizza with mushrooms", "Salad"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("two", "Pizza with mushrooms"),
        ("2", "Pizza with mushrooms"),
        ("the second one", "Pizza with mushrooms"),
        ("number three", "Salad"),
        ("option 1", "Pasta"),
        ("First.", "Pasta"),
        ("salad", "Salad"),
        ("Pizza", "Pizza with mushrooms"),
        ("pasta please", "Pasta"),
        ("four", None),
        ("the ninth", None),
        ("soup", None),
        ("", None),
        ("I want one with cheese", None),
    ],
)
def test_choice_by_number_and_label(text, expected):
    assert parse_choice(text, CHOICES) == expected


def test_choice_ambiguous_prefix_returns_none():
    assert parse_choice("approve", ["Approve once", "Approve always", "Deny"]) is None
    assert parse_choice("approve always", ["Approve once", "Approve always", "Deny"]) == "Approve always"


def test_choice_exact_label_beats_number():
    assert parse_choice("one", ["Two", "One"]) == "One"


@pytest.mark.parametrize(
    "text",
    ["Stop!", "stop the timer", "OK, stop.", "hey hermes stop", "be quiet", "that's enough", "turn it off", "stop stop"],
)
def test_stop_detection(text):
    assert is_stop(text) is True


@pytest.mark.parametrize(
    "text",
    ["what's the weather", "don't stop", "bus stop times for tomorrow", "", "start the timer"],
)
def test_not_stop(text):
    assert is_stop(text) is False


@pytest.mark.parametrize("text, expected", [
    ("Skip.", True), ("skip it", True), ("Skip this question, please.", True), ("please skip", True),
    ("skip the pasta", False), ("Pasta", False), ("", False), ("don't skip", False),
])
def test_skip_variants(text, expected):
    assert is_skip(text) is expected


def test_choice_ignores_the_recommended_label():
    assert parse_choice("tea", ["Tea (Recommended)", "Coffee"]) == "Tea (Recommended)"
    assert parse_choice("Coffee, please.", ["Tea (Recommended)", "Coffee"]) == "Coffee"
