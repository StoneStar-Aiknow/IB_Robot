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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("温度是25℃，湿度50%。", "温度是25℃,湿度50%。"),
        ("日期2024-10-21，比例3/4，3~5个", "日期2024-10-21,比例3/4,3~5个"),
        ("零下-5度", "零下-5度"),
        ("don't stop, state-of-the-art", "don't stop, state-of-the-art"),
        ("路径a/b", "路径a/b"),
    ],
)
def test_symbols_spoken_in_numbers_and_words_are_kept(text, expected):
    assert normalize_text(text) == expected


def test_unspoken_inline_separators_become_clause_pauses():
    assert normalize_text("抓取——放下") == "抓取,放下"
    assert normalize_text("抓取/放下") == "抓取,放下"


def test_periods_split_sentences_but_not_decimal_numbers():
    _normalized, segments = segment_text("Go now. Pi is 3.14159 ok", max_chars=12, max_segments=4)

    assert segments[0] == "Go now."
    assert all("3.14159" in segment for segment in segments if "3." in segment)


def test_english_without_punctuation_splits_between_words():
    _normalized, segments = segment_text("one two three four five", max_chars=10, max_segments=5)

    assert segments == ["one two ", "three ", "four five"]


def test_forced_character_windows_do_not_split_numbers():
    _normalized, segments = segment_text("圆周率是3.14159", max_chars=7, max_segments=3)

    assert segments == ["圆周率是", "3.14159"]
