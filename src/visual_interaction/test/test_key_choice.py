"""key_choice: the key-action multiple-choice question and its scores."""

import pytest

from visual_interaction.phase0.prompts import IMAGE_PLACEHOLDER
from visual_interaction.session import key_choice
from visual_interaction.session.key_actions import KEY_ACTIONS


class TestQuestions:
    def test_both_orders_end_with_none(self):
        assert key_choice.options(key_choice.ORDERS["fwd"]) == (*KEY_ACTIONS, "none")
        assert key_choice.options(key_choice.ORDERS["rev"]) == (*reversed(KEY_ACTIONS), "none")

    def test_one_question_per_picture_count_and_order(self):
        questions = key_choice.questions(frames=(1, 4))
        assert set(questions) == {"key.1.fwd", "key.1.rev", "key.4.fwd", "key.4.rev"}
        assert all(letters == key_choice.LETTERS for _, letters in questions.values())

    def test_the_prompt_lists_the_options_under_their_letters(self):
        text = key_choice.prompt(1, key_choice.ORDERS["rev"])
        assert text.count(IMAGE_PLACEHOLDER) == 1 and key_choice.SINGLE_LEAD in text
        assert f"A. {key_choice.OPTION_TEXT[KEY_ACTIONS[-1]]}" in text
        assert f"F. {key_choice.OPTION_TEXT['none']}" in text
        assert key_choice.CHOICE_SYSTEM in text

    def test_several_pictures_get_the_sequence_lead(self):
        text = key_choice.prompt(4, key_choice.ORDERS["fwd"])
        assert text.count(IMAGE_PLACEHOLDER) == 4 and key_choice.MULTI_LEAD in text


class TestScores:
    fwd = {"A": 0.30, "B": 0.05, "C": 0.05, "D": 0.05, "E": 0.05, "F": 0.10}

    def test_letters_map_back_to_answers_in_either_order(self):
        assert key_choice.by_answer(self.fwd, "fwd")[KEY_ACTIONS[0]] == 0.30
        rev = key_choice.by_answer(self.fwd, "rev")
        assert rev[KEY_ACTIONS[-1]] == 0.30 and rev["none"] == 0.10

    def test_shares_sum_to_one_and_survive_zero(self):
        shares = key_choice.shares({"a": 0.2, "b": 0.6})
        assert shares == pytest.approx({"a": 0.25, "b": 0.75})
        assert key_choice.shares({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}

    def test_average_is_of_shares(self):
        assert key_choice.average({"a": 0.1, "b": 0.1}, {"a": 0.9, "b": 0.3}) == pytest.approx({"a": 0.625, "b": 0.375})

    def test_calibration_removes_what_a_blank_picture_prefers(self):
        blank = {"a": 0.6, "b": 0.2, "none": 0.2}
        assert key_choice.calibrate({"a": 0.6, "b": 0.2, "none": 0.2}, blank) == pytest.approx(
            {"a": 1 / 3, "b": 1 / 3, "none": 1 / 3}
        )
        calibrated = key_choice.calibrate({"a": 0.5, "b": 0.4, "none": 0.1}, blank)
        assert calibrated["b"] > calibrated["a"] > calibrated["none"]

    def test_only_actions_beating_none_score(self):
        scores = key_choice.action_scores({"heart": 0.5, "thumb": 0.2, "wave": 0.05, "none": 0.25})
        assert scores == {"heart": 0.5}
        assert key_choice.action_scores({"heart": 0.4, "none": 0.6}) == {}
