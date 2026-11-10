"""The clip manifest's two views and the file choice shared by the executor and the voice bridge."""

import pytest

from embodied_common.recorded_clips import choose_clip_file, clip_files, skill_clips, spoken_clips, spoken_key

MANIFEST = {
    "schema_version": 1,
    "codes": {
        "X02": {
            "clip": "gesture_done",
            "mode": "random",
            "consumer": "speech",
            "lines": [{"file": "X02_01.wav", "text": "搞定啦！"}, {"file": "X02_02.wav", "text": "嘿嘿，好啦！"}],
        },
        "Q17A": {
            "clip": "price_first",
            "mode": "sequence",
            "consumer": "skill:answer_price",
            "lines": [{"file": "Q17A_01.wav", "text": "开源软件哦！"}, {"file": "Q17A_02.wav", "text": "扫码看看～"}],
        },
    },
}


def test_clip_names_map_to_their_files_and_a_sequence_to_one_joined_file():
    assert clip_files(MANIFEST, "/a") == {
        "gesture_done": ["/a/X02_01.wav", "/a/X02_02.wav"],
        "price_first": ["/a/Q17A_seq.wav"],
    }
    assert skill_clips(MANIFEST) == {"answer_price": "price_first"}


def test_spoken_text_plays_any_line_of_its_code():
    spoken = spoken_clips(MANIFEST, "/a")
    assert spoken[spoken_key("嘿嘿，好啦")] == ["/a/X02_01.wav", "/a/X02_02.wav"]
    assert spoken[spoken_key("搞定啦")] == ["/a/X02_01.wav", "/a/X02_02.wav"]
    assert spoken[spoken_key("开源软件哦！扫码看看～")] == ["/a/Q17A_seq.wav"]
    assert spoken_key("IB ROBOT，是什么？") == "ibrobot是什么"


def test_a_text_recorded_under_two_codes_or_a_duplicate_clip_is_refused():
    twice = {
        "schema_version": 1,
        "codes": {
            **MANIFEST["codes"],
            "S01": {
                "clip": "x",
                "mode": "random",
                "consumer": "speech",
                "lines": [{"file": "S01_01.wav", "text": "搞定啦"}],
            },
        },
    }
    with pytest.raises(ValueError):
        spoken_clips(twice, "/a")
    with pytest.raises(ValueError):
        clip_files({"schema_version": 1, "codes": {"A": MANIFEST["codes"]["X02"], "B": MANIFEST["codes"]["X02"]}}, "/a")


def test_the_choice_avoids_the_last_file_and_skips_missing_ones():
    assert choose_clip_file(["a", "b"], last="a") == "b"
    assert choose_clip_file(["a"], last="a") == "a"
    assert choose_clip_file(["a", "b"], exists=lambda path: path == "b") == "b"
    assert choose_clip_file(["a"], exists=lambda path: False) == ""
    assert choose_clip_file("one.wav") == "one.wav"
