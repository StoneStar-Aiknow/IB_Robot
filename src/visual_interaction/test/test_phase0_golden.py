"""Golden comparison: what counts as unchanged, and what must never pass."""

import json

import numpy as np
import pytest

from visual_interaction.phase0.golden import (
    Tolerance,
    ToleranceError,
    Tolerances,
    compare_archives,
    compare_exact,
    compare_tensor,
)


def tolerances(**overrides) -> Tolerances:
    values = {"tensors": {"vit": Tolerance(atol=1e-3, rtol=1e-3)}, "exact": frozenset({"tokens"})}
    values.update(overrides)
    return Tolerances(**values)


class TestToleranceFile:
    def test_it_loads_tensors_and_exact_names(self, tmp_path):
        path = tmp_path / "tol.json"
        path.write_text(json.dumps({"tensors": {"vit": {"atol": 0.001, "rtol": 0.0}}, "exact": ["tokens"]}))
        loaded = Tolerances.load(path)

        assert loaded.tensors["vit"].atol == 0.001
        assert "tokens" in loaded.exact

    def test_a_note_is_allowed_so_the_freeze_can_be_explained(self, tmp_path):
        path = tmp_path / "tol.json"
        path.write_text(json.dumps({"exact": ["tokens"], "note": "frozen before the 0C run"}))

        assert Tolerances.load(path).exact == frozenset({"tokens"})

    def test_an_unknown_key_is_refused(self, tmp_path):
        """A typo must not silently drop a tolerance."""
        path = tmp_path / "tol.json"
        path.write_text(json.dumps({"tensor": {}}))

        with pytest.raises(ToleranceError, match="unknown tolerance keys"):
            Tolerances.load(path)

    def test_a_tensor_missing_a_bound_is_refused(self, tmp_path):
        path = tmp_path / "tol.json"
        path.write_text(json.dumps({"tensors": {"vit": {"atol": 0.1}}}))

        with pytest.raises(ToleranceError, match="exactly atol and rtol"):
            Tolerances.load(path)

    def test_a_missing_file_is_a_tolerance_error(self, tmp_path):
        with pytest.raises(ToleranceError, match="cannot read"):
            Tolerances.load(tmp_path / "absent.json")

    def test_a_name_cannot_be_both_exact_and_toleranced(self):
        with pytest.raises(ToleranceError, match="both exact and toleranced"):
            Tolerances(tensors={"tokens": Tolerance(atol=0.0, rtol=0.0)}, exact=frozenset({"tokens"}))

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_a_nonsense_bound_is_refused(self, value):
        with pytest.raises(ToleranceError, match="finite non-negative"):
            Tolerance(atol=value, rtol=0.0)


class TestExactComparison:
    def test_identical_tokens_pass(self):
        tokens = np.array([1, 2, 3])

        assert compare_exact(tokens, tokens.copy())["passed"]

    def test_it_reports_where_two_token_streams_part(self):
        """The first divergence is where a decode started to drift."""
        result = compare_exact(np.array([1, 2, 3, 4]), np.array([1, 2, 9, 4]))

        assert not result["passed"]
        assert result["first_divergence"] == 2

    def test_a_shorter_stream_fails_at_its_end(self):
        """A generation that stopped early matches up to there and is still a failure."""
        result = compare_exact(np.array([1, 2, 3]), np.array([1, 2]))

        assert not result["passed"]
        assert result["first_divergence"] == 2


class TestTensorComparison:
    def test_a_difference_inside_the_tolerance_passes(self):
        reference = np.array([1.0, 2.0])
        candidate = reference + 5e-4

        assert compare_tensor(reference, candidate, Tolerance(atol=1e-3, rtol=0.0))["passed"]

    def test_a_difference_outside_it_fails_and_is_located(self):
        result = compare_tensor(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 3.0]), Tolerance(atol=1e-3, rtol=0.0))

        assert not result["passed"]
        assert result["violations"] == 1
        assert result["worst_index"] == [1]
        assert result["max_abs"] == pytest.approx(0.5)

    def test_the_relative_term_scales_with_the_reference(self):
        """A logit of 100 tolerates more absolute drift than one of 0.1."""
        loose = Tolerance(atol=0.0, rtol=1e-2)

        assert compare_tensor(np.array([100.0]), np.array([100.5]), loose)["passed"]
        assert not compare_tensor(np.array([0.1]), np.array([0.6]), loose)["passed"]

    def test_a_shape_change_beyond_the_leading_length_fails_immediately(self):
        result = compare_tensor(np.zeros((2, 3)), np.zeros((2, 4)), Tolerance(atol=1.0, rtol=1.0))

        assert not result["passed"]
        assert "shapes differ" in result["detail"]

    def test_a_different_number_of_steps_is_compared_over_the_prefix_and_still_fails(self):
        """Per-step logits from a run that stopped sooner: diagnosable, not passing."""
        result = compare_tensor(np.zeros((5, 3)), np.zeros((3, 3)), Tolerance(atol=1e-3, rtol=0.0))

        assert not result["passed"]
        assert result["compared_leading"] == 3

    def test_a_non_finite_value_fails_rather_than_comparing(self):
        """NaN compares false against everything, which would read as a pass."""
        result = compare_tensor(np.array([1.0]), np.array([float("nan")]), Tolerance(atol=1e9, rtol=1e9))

        assert not result["passed"]
        assert result["detail"] == "non-finite values"


class TestArchiveComparison:
    def test_matching_archives_pass(self):
        reference = {"vit": np.zeros((2, 2)), "tokens": np.array([1, 2])}

        assert compare_archives(reference, dict(reference), tolerances())["passed"]

    def test_one_failing_array_fails_the_comparison(self):
        reference = {"vit": np.zeros((2, 2)), "tokens": np.array([1, 2])}
        candidate = {"vit": np.zeros((2, 2)), "tokens": np.array([1, 3])}
        result = compare_archives(reference, candidate, tolerances())

        assert not result["passed"]
        assert not result["arrays"]["tokens"]["passed"]
        assert result["arrays"]["vit"]["passed"]

    def test_an_array_with_no_frozen_tolerance_is_an_error(self):
        """Skipping it would let a new output pass unexamined."""
        with pytest.raises(ToleranceError, match="no tolerance for"):
            compare_archives({"surprise": np.zeros(2)}, {"surprise": np.zeros(2)}, tolerances())

    def test_an_array_missing_from_one_side_fails_and_says_which(self):
        result = compare_archives({"tokens": np.array([1])}, {}, tolerances())

        assert not result["passed"]
        assert result["arrays"]["tokens"]["detail"] == "missing from candidate"
