"""Reading a field out of a JSON document that is still arriving.

This decides when the robot starts speaking, so a wrong answer is audible: a
false positive speaks a fragment or the wrong field, a false negative throws
away the latency gain.
"""

import pytest

from embodied_common.incremental_json import first_complete_string_field

FULL = '{"kind": "workflow", "user_message": "准备执行：挥手。", "summary": "wave", "steps": []}'


def test_returns_the_value_once_its_closing_quote_arrived():
    assert first_complete_string_field(FULL, "user_message") == "准备执行：挥手。"


@pytest.mark.parametrize("cut", range(1, len(FULL)))
def test_never_returns_a_fragment_at_any_truncation(cut):
    """Fed every possible prefix, it must return either nothing or the whole value."""
    got = first_complete_string_field(FULL[:cut], "user_message")
    assert got in (None, "准备执行：挥手。"), f"fragment at cut={cut}: {got!r}"


def test_a_prefix_ending_inside_the_value_returns_nothing():
    partial = '{"kind": "workflow", "user_message": "准备执行：挥'
    assert first_complete_string_field(partial, "user_message") is None


def test_the_field_name_inside_another_value_is_not_mistaken_for_it():
    """A substring search would speak the wrong text here."""
    buffer = '{"summary": "the user_message field explains it", "user_message": "真正的播报", "steps": []}'
    assert first_complete_string_field(buffer, "user_message") == "真正的播报"


def test_the_field_name_in_a_nested_object_is_ignored():
    buffer = '{"steps": [{"user_message": "嵌套的"}], "user_message": "顶层的"}'
    assert first_complete_string_field(buffer, "user_message") == "顶层的"


def test_an_escaped_quote_inside_the_value_does_not_end_it():
    """Counting quotes would close the string early and speak a fragment."""
    buffer = '{"user_message": "他说\\"好的\\"然后点头", "steps": []}'
    assert first_complete_string_field(buffer, "user_message") == '他说"好的"然后点头'


def test_escapes_are_decoded():
    buffer = '{"user_message": "\\u597d\\u7684\\n\\u6267\\u884c"}'
    assert first_complete_string_field(buffer, "user_message") == "好的\n执行"


def test_a_trailing_backslash_is_treated_as_incomplete():
    assert first_complete_string_field('{"user_message": "他说\\', "user_message") is None


@pytest.mark.parametrize(
    "buffer",
    ['{"kind": "workflow"}', "", "{", '{"user_message"', '{"user_message":', '{"user_message": '],
)
def test_absent_or_unstarted_returns_nothing(buffer):
    assert first_complete_string_field(buffer, "user_message") is None


@pytest.mark.parametrize("value", ["42", "null", "[]", "{}", "true"])
def test_a_non_string_field_returns_nothing(value):
    """Nothing speakable, and it must not be coerced into text."""
    assert first_complete_string_field(f'{{"user_message": {value}}}', "user_message") is None


def test_an_empty_value_is_returned_as_empty_not_as_missing():
    """The caller decides what to do with an empty announcement."""
    assert first_complete_string_field('{"user_message": ""}', "user_message") == ""


def test_the_first_occurrence_wins():
    buffer = '{"user_message": "第一个", "user_message": "第二个"}'
    assert first_complete_string_field(buffer, "user_message") == "第一个"
