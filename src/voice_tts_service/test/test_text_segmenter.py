import pytest

from voice_tts_service.errors import TTSError
from voice_tts_service.text_segmenter import normalize_text, segment_text


def test_normalize_and_segment_retains_supported_text_and_is_bounded():
    text = "  你好， 世界！\n这是一个很长的句子，没有丢字。  "

    normalized, segments = segment_text(text, max_chars=8, max_segments=20)

    assert normalized == normalize_text(text)
    assert "".join(segments) == normalized
    assert all(0 < len(segment) <= 8 for segment in segments)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好😊世界", "你好。世界"),
        ("你好，😊世界", "你好,世界"),
        ("你好😊！世界", "你好!世界"),
        ("你好😊🎉世界", "你好。世界"),
        ("任务完成🎉", "任务完成。"),
        ("🎉任务完成", "。任务完成"),
    ],
)
def test_symbols_are_removed_next_to_punctuation_or_replaced_with_a_period(text, expected):
    assert normalize_text(text) == expected


def test_markdown_formatting_and_controls_are_removed_without_extra_pauses():
    assert normalize_text("**你好**\u200b #世界") == "你好 世界"


def test_line_breaks_become_sentence_boundaries():
    assert normalize_text("第一项完成\n\n第二项开始") == "第一项完成。第二项开始"


def test_long_clause_falls_back_to_character_windows():
    normalized, segments = segment_text("abcdefghijkl", max_chars=5, max_segments=3)

    assert normalized == "abcdefghijkl"
    assert segments == ["abcde", "fghij", "kl"]


def test_empty_and_excessive_segment_count_have_stable_errors():
    with pytest.raises(TTSError, match="empty") as empty:
        segment_text(" \n ", max_chars=5, max_segments=2)
    assert empty.value.code == "INVALID_TEXT"

    with pytest.raises(TTSError, match="limit") as excessive:
        segment_text("abcdefghijkl", max_chars=3, max_segments=3)
    assert excessive.value.code == "REQUEST_TOO_LARGE"
