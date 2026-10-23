"""The prompt budget: what fits the fixed OM shapes, and what overflows.

These tests pin the arithmetic. The measured counts for the real tokenizer are
recorded in the design doc, not asserted here: a tokenizer is data this test
does not have, and a prompt reword would otherwise break the suite rather than
the budget.
"""

import pytest

from visual_interaction.phase0.budget import (
    IMAGE_CONTEXT_TOKEN,
    BudgetError,
    OmShape,
    assess,
    check_image_slots,
    expand_images,
    image_slots,
    require_fit,
)


def shape(**overrides) -> OmShape:
    values = {"prefill_len": 1280, "context_len": 1344, "frames": 4, "image_tokens_per_frame": 256}
    values.update(overrides)
    return OmShape(**values)


class TestOmShape:
    def test_the_board_export_leaves_this_much_for_text(self):
        """4 frames x 256 = 1024 of 1280 positions go to images before any text."""
        board = shape()

        assert board.image_positions == 1024
        assert board.text_capacity_with_images == 256

    def test_a_cache_shorter_than_the_prefill_is_refused(self):
        with pytest.raises(ValueError, match="cannot be shorter"):
            shape(context_len=1000)

    def test_images_that_fill_the_prefill_are_refused(self):
        """Nothing could be asked, so it is a broken export rather than a tight one."""
        with pytest.raises(ValueError, match="images alone"):
            shape(prefill_len=1024)

    @pytest.mark.parametrize("field", ["prefill_len", "context_len", "frames", "image_tokens_per_frame"])
    def test_every_shape_must_be_positive(self, field):
        with pytest.raises(ValueError, match="positive"):
            shape(**{field: 0})


class TestImageExpansion:
    def test_each_placeholder_becomes_one_frame_of_context_slots(self):
        expanded = expand_images("a<image>b", shape(image_tokens_per_frame=3))

        assert expanded == "a<img>" + IMAGE_CONTEXT_TOKEN * 3 + "</img>b"

    def test_slots_are_counted_before_expansion(self):
        assert image_slots("<image><image>") == 2

    def test_a_prompt_with_the_wrong_number_of_frames_is_caught_from_the_text(self):
        """The generator only notices after the vision OM has run; this costs nothing."""
        with pytest.raises(BudgetError, match="3 <image> slot"):
            check_image_slots("p", "<image><image><image>", with_images=True, shape=shape())

    def test_a_text_only_prompt_may_not_carry_a_frame(self):
        with pytest.raises(BudgetError, match="needs 0"):
            check_image_slots("p", "<image>", with_images=False, shape=shape())

    def test_the_right_number_passes(self):
        check_image_slots("p", "<image>" * 4, with_images=True, shape=shape())


class TestAssessment:
    def test_text_tokens_exclude_the_image_positions(self):
        item = assess("p", prompt_tokens=1238, max_new_tokens=64, shape=shape(), with_images=True)

        assert item.text_tokens == 214

    def test_a_text_only_prompt_is_all_text(self):
        item = assess("p", prompt_tokens=100, max_new_tokens=48, shape=shape(), with_images=False)

        assert item.text_tokens == 100

    def test_a_prompt_at_the_prefill_length_still_fits(self):
        assert assess("p", prompt_tokens=1280, max_new_tokens=1, shape=shape(), with_images=True).fits_prefill

    def test_one_token_over_the_prefill_does_not(self):
        assert not assess("p", prompt_tokens=1281, max_new_tokens=1, shape=shape(), with_images=True).fits_prefill

    def test_the_cache_bound_matches_the_generator(self):
        """The generator refuses n + max_new_tokens - 1 > L; the last token needs no slot."""
        board = shape()
        exact = assess("p", prompt_tokens=1280, max_new_tokens=65, shape=board, with_images=True)
        over = assess("p", prompt_tokens=1280, max_new_tokens=66, shape=board, with_images=True)

        assert exact.fits_cache and not over.fits_cache

    def test_it_reports_how_many_new_tokens_would_fit(self):
        item = assess("p", prompt_tokens=1238, max_new_tokens=64, shape=shape(), with_images=True)

        assert item.max_new_tokens_allowed == 107
        assert item.fits

    def test_fitting_needs_both_bounds(self):
        """A prompt inside the prefill can still leave no room to answer."""
        item = assess("p", prompt_tokens=1280, max_new_tokens=200, shape=shape(), with_images=True)

        assert item.fits_prefill and not item.fits_cache and not item.fits

    def test_the_parts_are_reported_with_their_boundary_difference(self):
        """Encoding parts apart and together differs where a merge spans a boundary."""
        item = assess(
            "p",
            prompt_tokens=100,
            max_new_tokens=10,
            shape=shape(),
            with_images=False,
            parts=[("a", 60), ("b", 39)],
        )
        payload = item.as_dict()

        assert payload["parts"] == {"a": 60, "b": 39}
        assert payload["sum_of_parts"] == 99
        assert payload["boundary_difference"] == 1

    def test_no_parts_reports_no_difference(self):
        payload = assess("p", prompt_tokens=10, max_new_tokens=1, shape=shape(), with_images=False).as_dict()

        assert payload["boundary_difference"] == 0

    @pytest.mark.parametrize(("tokens", "new"), [(0, 10), (10, 0)])
    def test_degenerate_counts_are_refused(self, tokens, new):
        with pytest.raises(ValueError):
            assess("p", prompt_tokens=tokens, max_new_tokens=new, shape=shape(), with_images=False)


class TestRequireFit:
    def test_it_names_every_scenario_and_the_bound_it_breaks(self):
        board = shape()
        items = [
            assess("over_prefill", prompt_tokens=1400, max_new_tokens=1, shape=board, with_images=True),
            assess("over_cache", prompt_tokens=1200, max_new_tokens=400, shape=board, with_images=True),
        ]

        with pytest.raises(BudgetError) as error:
            require_fit(items)

        assert "over_prefill" in str(error.value)
        assert "over_cache" in str(error.value)
        assert "KV cache" in str(error.value)

    def test_fitting_scenarios_raise_nothing(self):
        require_fit([assess("p", prompt_tokens=1238, max_new_tokens=64, shape=shape(), with_images=True)])
