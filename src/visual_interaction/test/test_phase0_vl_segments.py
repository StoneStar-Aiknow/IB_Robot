"""Reading a demo run log: which rounds count, and how an outcome is labelled."""

import json

import pytest

from visual_interaction.phase0.vl_segments import outcome_of, read_rounds


def write_log(tmp_path, records) -> "object":
    path = tmp_path / "talk.jsonl"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return path


class TestReadRounds:
    def test_each_line_is_one_round(self, tmp_path):
        path = write_log(tmp_path, [{"round": 1}, {"round": 2}])

        assert [record["round"] for record in read_rounds(path)] == [1, 2]

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "talk.jsonl"
        path.write_text('{"round": 1}\n\n{"round": 2}\n', encoding="utf-8")

        assert len(list(read_rounds(path))) == 2

    def test_an_unreadable_line_names_its_number(self, tmp_path):
        """A truncated log is a broken measurement, not fewer rounds."""
        path = tmp_path / "talk.jsonl"
        path.write_text('{"round": 1}\n{"round":\n', encoding="utf-8")

        with pytest.raises(ValueError, match="talk.jsonl:2"):
            list(read_rounds(path))

    def test_a_line_that_is_not_an_object_is_refused(self, tmp_path):
        path = tmp_path / "talk.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")

        with pytest.raises(ValueError, match="not a JSON object"):
            list(read_rounds(path))


class TestOutcome:
    def test_a_round_with_text_spoke(self):
        assert outcome_of({"text": "你举起的玩偶真有趣。", "nobody": False}) == "spoke"

    def test_no_person_outranks_an_empty_text(self):
        """Both are quiet; they are counted apart so a blind run shows."""
        assert outcome_of({"text": "", "nobody": True}) == "no_person"

    def test_retrying_out_is_its_own_outcome(self):
        assert outcome_of({"text": "", "nobody": False, "tries": 4}) == "no_usable_text"

    def test_whitespace_is_not_text(self):
        assert outcome_of({"text": "   ", "nobody": False}) == "no_usable_text"

    def test_a_missing_field_does_not_raise(self):
        """An older log is summarised, not rejected."""
        assert outcome_of({}) == "no_usable_text"
