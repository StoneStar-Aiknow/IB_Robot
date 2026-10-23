"""The prompts this feature would send: grounded in the contract, and budgeted.

These pin the structure and the invariants, not token counts: the counts
depend on a tokenizer this test does not carry, and ``token_budget`` measures
them against the real one.
"""

import json

import pytest

from embodied_common.visual_interaction_contract import CATEGORIES, KINDS, REGIONS
from visual_interaction.phase0.prompts import (
    IMAGE_PLACEHOLDER,
    RULES_FULL,
    RULES_MINIMAL,
    SCHEMA,
    candidate_list,
    chat,
    default_scenarios,
    frame_header,
    observe_scenario,
    spoken_memory,
)


class TestChatTemplate:
    def test_it_wraps_the_question_in_the_model_s_markers(self):
        wrapped = chat("问题")

        assert wrapped.startswith("<|im_start|>system\n")
        assert "<|im_start|>user\n问题<|im_end|>\n" in wrapped
        assert wrapped.endswith("<|im_start|>assistant\n")

    def test_the_question_appears_exactly_once_so_it_can_be_priced_apart(self):
        """observe_scenario splits the wrapper on the body; two copies would break that."""
        assert chat("问题").count("问题") == 1


class TestFrameHeader:
    def test_each_frame_gets_one_numbered_placeholder(self):
        header = frame_header(3)

        assert header.count(IMAGE_PLACEHOLDER) == 3
        assert "Frame1:" in header and "Frame3:" in header

    def test_no_frames_is_an_empty_header(self):
        assert frame_header(0) == ""


class TestSchema:
    @pytest.mark.parametrize("language", ["zh", "en"])
    def test_every_closed_value_is_named_in_both_languages(self, language):
        """The model cannot answer inside a whitelist it was not told."""
        text = SCHEMA[language]

        for value in (*KINDS, *CATEGORIES, *REGIONS):
            assert value in text, f"{value} missing from {language} schema"

    @pytest.mark.parametrize("language", ["zh", "en"])
    def test_it_asks_for_exactly_the_four_contract_fields(self, language):
        text = SCHEMA[language]

        for field in ("kind", "category", "region", "reply"):
            assert field in text

    def test_no_schema_offers_a_fallback_category(self):
        """A generic "other" would hand the sensitivity judgement back to the model."""
        for text in SCHEMA.values():
            assert "other" not in text.lower()


class TestRules:
    def test_the_minimal_rules_cover_the_privacy_boundary(self):
        """What semantics.check_reply refuses, the model must be told up front."""
        assert "none" in RULES_MINIMAL and "uncertain" in RULES_MINIMAL
        assert "文字" in RULES_MINIMAL
        assert "外貌" in RULES_MINIMAL

    def test_the_full_rules_extend_rather_than_replace_the_minimal_ones(self):
        assert RULES_FULL.startswith(RULES_MINIMAL)
        assert len(RULES_FULL) > len(RULES_MINIMAL)

    def test_the_full_rules_say_text_in_frame_is_data(self):
        """The injection boundary, stated to the model as well as enforced after."""
        assert "指令" in RULES_FULL


class TestMemoryAndCandidates:
    def test_nothing_spoken_adds_nothing_to_the_prompt(self):
        assert spoken_memory([]) == ""
        assert candidate_list([]) == ""

    def test_each_item_appears(self):
        text = spoken_memory(["toy/hand", "hat/head"])

        assert "toy/hand" in text and "hat/head" in text


class TestObserveScenario:
    def test_it_asks_for_one_frame_slot_per_frame(self):
        scenario = observe_scenario("s", frames=4)

        assert scenario.text.count(IMAGE_PLACEHOLDER) == 4
        assert scenario.with_images

    def test_the_parts_reassemble_into_the_prompt(self):
        """Pricing the parts must not leave anything out of the total."""
        scenario = observe_scenario("s", frames=4, memory=["toy/hand"], candidates=["候选1"])

        assert "".join(text for _, text in scenario.parts) == scenario.text

    def test_the_template_is_priced_apart_from_the_body(self):
        scenario = observe_scenario("s", frames=4)
        names = [name for name, _ in scenario.parts]

        assert names[0] == "template_prefix"
        assert names[-1] == "template_suffix"
        assert "schema" in names and "rules" in names

    def test_memory_and_candidates_appear_only_when_present(self):
        bare = [name for name, _ in observe_scenario("s", frames=4).parts]
        full = [name for name, _ in observe_scenario("s", frames=4, memory=["a"], candidates=["b"]).parts]

        assert "memory" not in bare and "candidates" not in bare
        assert "memory" in full and "candidates" in full

    def test_an_unknown_label_language_is_refused(self):
        with pytest.raises(ValueError, match="language must be"):
            observe_scenario("s", frames=4, language="fr")

    def test_the_whole_prompt_is_one_call(self):
        """The first version gets the classification and the reply in one answer."""
        scenario = observe_scenario("s", frames=4)
        schema = dict(scenario.parts)["schema"]

        assert "reply" in schema


class TestDefaultScenarios:
    def test_the_matrix_covers_what_the_design_asks_to_vary(self):
        names = default_scenarios(frames=4).names()

        assert any("en" in name for name in names), "label language"
        assert any("full_rules" in name for name in names), "ruleset size"
        assert any("memory" in name for name in names), "A/B/C memory"
        assert any("candidates" in name for name in names), "candidate list"
        assert any("worst_case" in name for name in names), "combined worst case"

    def test_every_scenario_has_a_unique_name(self):
        names = default_scenarios(frames=4).names()

        assert len(set(names)) == len(names)

    def test_every_scenario_asks_for_the_configured_frame_count(self):
        for scenario in default_scenarios(frames=2).scenarios:
            assert scenario.text.count(IMAGE_PLACEHOLDER) == 2

    def test_the_worst_case_is_the_longest_prompt(self):
        """It exists to find the overflow before the board does."""
        scenarios = {s.name: s for s in default_scenarios(frames=4).scenarios}
        worst = scenarios["observe.zh.worst_case"]

        assert all(len(worst.text) >= len(s.text) for s in scenarios.values())

    def test_no_scenario_leaks_a_category_outside_the_contract(self):
        """The prompt is where a widened whitelist would start."""
        for scenario in default_scenarios(frames=4).scenarios:
            body = json.dumps(scenario.text, ensure_ascii=False)
            for forbidden in ("pendant", "挂坠", "necklace"):
                assert forbidden not in body
