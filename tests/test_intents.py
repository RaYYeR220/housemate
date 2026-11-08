import pytest

from housemate.core.intents import is_stop, parse_choice, parse_yes_no


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Approve.", "approve"),
        ("yes", "approve"),
        ("Yeah, do it!", "approve"),
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
    ],
)
def test_yes_no_variants(text, expected):
    assert parse_yes_no(text) == expected


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
