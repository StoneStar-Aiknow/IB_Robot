import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from voice_tts_service.errors import BackendInferenceError, BackendLoadError
from voice_tts_service.zipvoice_shared import (
    _CHINESE_UNIT,
    _MODEL_NUMBER,
    _UNIT_WORDS,
    DurationModel,
    EmiliaTokenizer,
    PromptProfile,
    TokenGroup,
    chunk_conditions,
    content_frame_count,
    join_chunks,
    layout_condition,
    plan_chunks,
    token_embeddings,
)

SENTINEL = (" ", "n0", "n2", ".")


def _word(*tokens, kind="zh"):
    return TokenGroup(tuple(tokens), kind)


def _punct(mark):
    return TokenGroup((mark,), "punct")


def test_tokens_get_their_own_durations_instead_of_an_even_split():
    durations = DurationModel(300, 30, english_token_scale=2.0, spoken_token_scale=0.8)
    groups = [_word("h0", "ai3"), TokenGroup((" ",), "space"), _word("h", "i", kind="en"), _punct(","), _punct(".")]

    # Chinese at 0.8 of the prompt rate, English and spaces shorter, punctuation as pause frames.
    assert durations.token_frames(groups) == [8, 8, 4, 4, 4, 20, 30]


def test_speed_scales_every_generated_frame():
    durations = DurationModel(300, 30, speed=2.0, spoken_token_scale=1.0)

    assert durations.target_frames([_word("h0", "ai3")]) == 10


def _tokenizer(tmp_path, monkeypatch, *, lexicon=None):
    token_file = tmp_path / "tokens.txt"
    tokens = ["_", ".", ",", " ", "a1", "p", "e", "n", "o", "san1", "yi1", "ling2"]
    token_file.write_text("".join(f"{token}\t{index}\n" for index, token in enumerate(tokens)), encoding="utf-8")
    fake_tone = SimpleNamespace(to_finals_tone3=lambda syllable, **kwargs: syllable, to_initials=lambda *a, **k: "")
    modules = {
        "cn2an": SimpleNamespace(transform=lambda value, _mode: value),
        "jieba": SimpleNamespace(
            default_logger=SimpleNamespace(setLevel=lambda _level: None),
            initialize=lambda: None,
            cut=lambda value: list(value),
        ),
        "pypinyin": SimpleNamespace(
            Style=SimpleNamespace(TONE3="tone3"),
            lazy_pinyin=lambda words, **kwargs: ["a1" for word in words for _char in word],
        ),
        "pypinyin.contrib": SimpleNamespace(tone_convert=fake_tone),
        "pypinyin.contrib.tone_convert": fake_tone,
        "piper_phonemize": SimpleNamespace(phonemize_espeak=lambda text, voice: [list(text.lower())]),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return EmiliaTokenizer(token_file, lexicon=lexicon)


def test_model_numbers_after_chinese_are_read_digit_by_digit(tmp_path, monkeypatch):
    tokenizer = _tokenizer(tmp_path, monkeypatch)

    # Digits bypass pypinyin, so "一" keeps yi1 instead of a tone-sandhi yi4.
    assert tokenizer.text_to_tokens("腾310P.") == ["a1", "san1", "yi1", "ling2", "p", "."]
    # A quantity followed by a Chinese unit is still a number, and "X2" keeps its English reading.
    assert _MODEL_NUMBER.search("共310个,灵犀X2") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("延迟是35ms,", "延迟是35毫秒,"),
        ("前方2m处", "前方2米处"),
        ("频率是2.4GHz", "频率是2.4吉赫兹"),
        # Units need a number after Chinese text; English text and model names keep their letters.
        ("wait 5s", "wait 5s"),
        ("昇腾310P", "昇腾310P"),
        ("跑到5more", "跑到5more"),
    ],
)
def test_units_after_numbers_in_chinese_are_read_in_chinese(text, expected):
    assert _CHINESE_UNIT.sub(lambda match: match[1] + _UNIT_WORDS[match[2]], text) == expected


def test_lexicon_replaces_whole_words_before_g2p(tmp_path, monkeypatch):
    tokenizer = _tokenizer(tmp_path, monkeypatch, lexicon={"nope": "no"})

    assert tokenizer.text_to_tokens("nope.") == ["n", "o", "."]
    assert tokenizer.text_to_tokens("nopen.") == ["n", "o", "p", "e", "n", "."]
    # The default lexicon reads openEuler as "open" plus Chinese "欧拉".
    assert [group.kind for group in tokenizer.text_to_groups("openEuler")] == ["en", "zh", "zh", "punct"]


def test_invalid_lexicon_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(BackendLoadError, match="lexicon"):
        _tokenizer(tmp_path, monkeypatch, lexicon={"openEuler": 1})


def _unit_durations():
    # One frame per spoken token, so chunk capacity tests count tokens.
    return DurationModel(10, 10, spoken_token_scale=1.0)


def test_chunks_keep_whole_sentences_and_reserve_the_sentinel():
    first = [_word("a1", "a1"), _word("b1", "b1"), _punct(","), _word("c1"), _punct(".")]
    second = [_word("d1", "d1"), _punct("!")]

    plans = plan_chunks([*first, *second], _unit_durations(), max_tokens=12, max_target_frames=100, sentinel=SENTINEL)

    assert [plan.tokens[: plan.content_tokens] for plan in plans] == [
        ("a1", "a1", "b1", "b1", ",", "c1", "."),
        ("d1", "d1", "!"),
    ]
    assert all(plan.tokens[plan.content_tokens :] == SENTINEL for plan in plans)
    assert [plan.pause_after_ms for plan in plans] == [250, 250]


def test_frame_budget_limits_chunks_including_pause_frames():
    groups = [_word("a1"), _punct("."), _word("b1"), _punct(".")]

    # Each sentence needs 1 + 3 frames; two sentences do not fit in 7 frames.
    plans = plan_chunks(groups, _unit_durations(), max_tokens=100, max_target_frames=7)

    assert [plan.tokens for plan in plans] == [("a1", "."), ("b1", ".")]


def test_long_sentences_split_at_clauses_then_words_never_inside_a_word():
    groups = [_word("a1", "a1"), _word("b1", "b1"), _word("c1", "c1"), _punct(","), _word("d1"), _punct(".")]

    plans = plan_chunks(groups, _unit_durations(), max_tokens=5, max_target_frames=100)

    # The first clause is too long and splits between words; the next clause still fills the chunk.
    assert [plan.tokens for plan in plans] == [("a1", "a1", "b1", "b1"), ("c1", "c1", ",", "d1", ".")]
    assert [plan.pause_after_ms for plan in plans] == [60, 250]


def test_chunk_edges_drop_spaces_between_english_words():
    space = TokenGroup((" ",), "space")
    groups = [_word("a", kind="en"), space, _word("b", kind="en"), space, _word("c", kind="en"), _punct(".")]

    plans = plan_chunks(groups, _unit_durations(), max_tokens=3, max_target_frames=100)

    assert [plan.tokens for plan in plans] == [("a", " ", "b"), ("c", ".")]


def test_a_word_larger_than_the_chunk_capacity_is_rejected():
    with pytest.raises(BackendInferenceError, match="single ZipVoice word"):
        plan_chunks([_word("a1", "a1", "a1")], _unit_durations(), max_tokens=2, max_target_frames=10)


def test_clause_chunks_end_after_long_clauses_and_merge_short_ones():
    groups = [_word("a1"), _punct(","), _word("b1", "b1", "b1"), _punct(","), _word("c1"), _punct(".")]

    plans = plan_chunks(groups, _unit_durations(), max_tokens=100, max_target_frames=100, clause_chunk_tokens=4)

    # "a1," is too short to stand alone; once a chunk reaches 4 tokens it ends at the next clause.
    assert [plan.tokens for plan in plans] == [("a1", ",", "b1", "b1", "b1", ","), ("c1", ".")]
    assert [plan.pause_after_ms for plan in plans] == [120, 250]


def test_sentinel_crop_follows_the_start_offset_to_the_quietest_frame():
    (plan,) = plan_chunks(
        [_word("a1"), _punct(".")], _unit_durations(), max_tokens=20, max_target_frames=100, sentinel=SENTINEL
    )
    # The exported layout starts the text two frames before the end of the prompt audio.
    plan = replace(plan, start_offset=-2)
    generated = np.full((9, 4), -0.1, dtype=np.float32)
    generated[2] = -1.6

    # Content (a1=1 + "."=3 frames) ends at frame -2 + 4 = 2, which is also the quietest frame.
    assert content_frame_count(plan, generated) == 2


def test_sentinel_crop_without_final_punctuation_uses_the_quietest_frame():
    (plan,) = plan_chunks(
        [_word("a1", "b1")], _unit_durations(), max_tokens=20, max_target_frames=100, sentinel=SENTINEL
    )
    generated = np.full((6, 4), -0.1, dtype=np.float32)
    generated[2] = -1.6

    assert content_frame_count(plan, generated) == 2


def test_without_a_sentinel_all_generated_frames_are_content():
    (plan,) = plan_chunks([_word("a1"), _punct(".")], _unit_durations(), max_tokens=20, max_target_frames=100)

    assert content_frame_count(plan, np.zeros((9, 4), dtype=np.float32)) == 9


def _repeated_condition(token_count, token_dur, tail):
    embeddings = np.arange(token_count + 1, dtype=np.float32)[:, None] * np.ones((1, 3), dtype=np.float32)
    rows = np.repeat(embeddings[:token_count], token_dur, axis=0)
    return np.concatenate([rows, np.repeat(embeddings[-1:], tail, axis=0)]), embeddings[:token_count]


def test_token_embeddings_recovers_rows_from_the_even_repeat_layout():
    condition, embeddings = _repeated_condition(3, 4, tail=2)

    assert np.array_equal(token_embeddings(condition, 3, condition.shape[0]), embeddings)
    # The last row of the first token span differs from its first row.
    condition[3] = 99.0
    assert token_embeddings(condition, 3, condition.shape[0]) is None


def test_layout_condition_keeps_the_encoder_prompt_layout():
    embeddings = np.arange(4, dtype=np.float32)[:, None]

    laid = layout_condition(embeddings, prompt_tokens=2, prompt_frames=5, frames=[1, 2])

    # Prompt tokens keep 5 // 2 = 2 frames each, so the text starts one frame before the prompt audio ends.
    assert laid[:, 0].tolist() == [0, 0, 1, 1, 2, 3, 3]


def test_chunk_conditions_relays_target_tokens_and_reports_the_start_offset():
    features = np.arange(15, dtype=np.float32).reshape(1, 5, 3)
    prompt = PromptProfile(tokens=np.zeros((1, 2), dtype=np.int64), features=features)
    (plan,) = plan_chunks(
        [_word("a1"), _punct(".")], DurationModel(5, 2, spoken_token_scale=1.0), max_tokens=10, max_target_frames=100
    )
    condition, _embeddings = _repeated_condition(4, 3, tail=1)

    text, speech, effective = chunk_conditions(plan, prompt, condition, condition.shape[0])

    assert effective.frames == plan.frames
    assert effective.start_offset == -1
    assert text.shape[0] == 4 + sum(plan.frames)
    assert np.array_equal(speech[:5], features[0])
    assert not speech[5:].any()


def test_chunk_conditions_keeps_the_encoder_layout_when_rows_are_not_repeats():
    prompt = PromptProfile(tokens=np.zeros((1, 2), dtype=np.int64), features=np.zeros((1, 4, 3), dtype=np.float32))
    (plan,) = plan_chunks([_word("a1"), _punct(".")], DurationModel(4, 2), max_tokens=10, max_target_frames=100)
    condition = np.random.default_rng(0).standard_normal((12, 3)).astype(np.float32)

    text, _speech, effective = chunk_conditions(plan, prompt, condition, 12)

    # 12 frames over 2 prompt + 2 target tokens: 3 frames per token, text starting at frame 6.
    assert np.array_equal(text, condition)
    assert effective.frames == (3, 3)
    assert effective.start_offset == 2


def test_join_chunks_trims_edge_silence_and_inserts_pauses():
    sample_rate = 1000
    speech = np.concatenate([np.zeros(100), np.ones(200), np.zeros(100)]).astype(np.float32)

    joined = join_chunks([speech, speech], [250, 0], sample_rate, tail_silence_ms=100)

    # 30 ms of margin survives on each side of the 200 ms of speech.
    assert joined.size == 2 * 260 + 250 + 100
    assert np.all(joined[260 : 260 + 250] == 0.0)
    assert np.all(joined[-100:] == 0.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("日期2026-10-21", "日期2026年10月21日"),
        ("10:30开会", "10点30分开会"),
        ("1,000元", "1000元"),
        ("3~5个", "3至5个"),
    ],
)
def test_chinese_number_formats_are_rewritten_before_cn2an(text, expected):
    assert EmiliaTokenizer._normalize_chinese_numbers(text) == expected


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("打开ROS系统.", ["zh", "en", "zh", "punct"]),
        ("第3号.", ["zh", "punct"]),
        ("I have 3 apples.", ["en", "punct"]),
        ("圆周率3.14.", ["zh", "punct"]),
        ("2026,", ["zh", "punct"]),
    ],
)
def test_digits_and_spaces_follow_the_language_of_the_preceding_word(text, kinds):
    assert [kind for kind, _run in EmiliaTokenizer._split_languages(text)] == kinds
