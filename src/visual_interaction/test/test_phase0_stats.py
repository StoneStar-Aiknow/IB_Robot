"""Summary statistics: the shape phase zero reports, and the small-sample flag."""

import pytest

from visual_interaction.phase0.stats import MIN_EXPLORATORY_RUNS, counts, percentile, summarise


class TestPercentile:
    def test_it_interpolates_between_closest_ranks(self):
        """numpy's default, so a figure here matches one recomputed from raw samples."""
        assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
        assert percentile([1.0, 2.0, 3.0, 4.0], 25.0) == 1.75

    def test_the_ends_are_the_extremes(self):
        values = [5.0, 1.0, 3.0]
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 100.0) == 5.0

    def test_one_value_is_every_percentile(self):
        assert percentile([7.0], 95.0) == 7.0

    def test_input_order_does_not_matter(self):
        assert percentile([3.0, 1.0, 2.0], 50.0) == percentile([1.0, 2.0, 3.0], 50.0)

    @pytest.mark.parametrize("q", [-1.0, 101.0])
    def test_a_percentile_outside_the_range_is_refused(self, q):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile([1.0], q)

    def test_no_values_has_no_percentile(self):
        with pytest.raises(ValueError, match="no values"):
            percentile([], 50.0)


class TestSummarise:
    def test_it_reports_the_fields_the_design_asks_for(self):
        summary = summarise(range(1, 101))

        assert summary["n"] == 100
        assert summary["p50"] == pytest.approx(50.5)
        assert summary["p95"] == pytest.approx(95.05)
        assert summary["max"] == 100
        assert summary["min"] == 1

    def test_an_empty_sample_is_a_result_rather_than_an_error(self):
        """A condition where every round was quiet has no speak timings, and that must show."""
        assert summarise([]) == {"n": 0}

    def test_a_sample_below_thirty_is_flagged(self):
        """§12: thirty runs is the floor for citing a distribution at all."""
        assert summarise([1.0] * (MIN_EXPLORATORY_RUNS - 1))["below_exploratory_minimum"] is True
        assert summarise([1.0] * MIN_EXPLORATORY_RUNS)["below_exploratory_minimum"] is False

    @pytest.mark.parametrize("value", [float("inf"), float("nan"), float("-inf")])
    def test_a_non_finite_timing_is_refused(self, value):
        """A timeout recorded as inf would make every summary meaningless."""
        with pytest.raises(ValueError, match="finite"):
            summarise([1.0, value])

    def test_it_accepts_any_iterable_once(self):
        assert summarise(x for x in [2.0, 4.0])["mean"] == 3.0


class TestCounts:
    def test_it_tallies_by_text_in_a_stable_order(self):
        assert counts(["b", "a", "b"]) == {"a": 1, "b": 2}

    def test_numbers_become_their_text(self):
        """Retry counts and error codes are tallied the same way."""
        assert counts([1, 1, 2]) == {"1": 2, "2": 1}

    def test_nothing_counted_is_an_empty_tally(self):
        assert counts([]) == {}
