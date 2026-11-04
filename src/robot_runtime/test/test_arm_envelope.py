"""Unit tests for the shared arm-envelope / rest-window logic (review rework).

Pure-sample tests: the freshness rule covers the WHOLE interval — the
leading edge (the first sample must arrive within the gap bound of the
start), the internal gaps and the trailing edge (the last sample must be
within the bound of the end) — with missing joint fields and non-finite
values failing, never skipping. The rest verdict accumulates the worst
velocity over the whole window (a moving start with a still final frame is
motion) and treats NO observations as a failure, not a clean zero.
"""

from __future__ import annotations

import math

import pytest

from robot_runtime.testing.arm_envelope import envelope_verdict, rest_verdict

RESET = {"1": 0.0, "2": -1.5854}
JOINTS = ["1", "2"]


def _positions(stamps, joint2=-1.5854):
    return [(stamp, {"1": 0.0, "2": joint2}) for stamp in stamps]


class TestEnvelopeVerdict:
    def test_a_fresh_stowed_stream_is_within(self):
        verdict = envelope_verdict(
            _positions([10.0, 10.1, 10.2, 10.3, 10.4]), 10.0, 10.4, reset=RESET, tolerance=0.1, max_gap=0.5
        )
        assert verdict["within"], verdict["violations"]
        assert verdict["worst_rad"] == 0.0 and verdict["samples"] == 5

    def test_a_late_first_sample_fails_the_leading_edge(self):
        # the stream only starts 0.8 s into the window: the arm was
        # unobserved at the start, which the internal-gap-only rule missed
        verdict = envelope_verdict(
            _positions([10.8, 10.9, 11.0, 11.2]), 10.0, 11.2, reset=RESET, tolerance=0.1, max_gap=0.5
        )
        assert not verdict["within"]
        assert any("after the window start" in problem for problem in verdict["violations"])

    def test_a_tail_outage_fails_the_trailing_edge(self):
        # samples stop 0.8 s before the window ends
        verdict = envelope_verdict(
            _positions([10.0, 10.1, 10.2, 10.4]), 10.0, 11.2, reset=RESET, tolerance=0.1, max_gap=0.5
        )
        assert not verdict["within"]
        assert any("before the window end" in problem for problem in verdict["violations"])

    def test_an_internal_gap_fails(self):
        verdict = envelope_verdict(
            _positions([10.0, 10.1, 10.9, 11.0]), 10.0, 11.0, reset=RESET, tolerance=0.1, max_gap=0.5
        )
        assert not verdict["within"]
        assert any("internal sample gap" in problem for problem in verdict["violations"])

    def test_a_missing_joint_field_fails_instead_of_skipping(self):
        samples = [(10.0, {"1": 0.0}), (10.1, {"1": 0.0, "2": -1.5854}), (10.2, {"1": 0.0, "2": -1.5854})]
        verdict = envelope_verdict(samples, 10.0, 10.2, reset=RESET, tolerance=0.1, max_gap=0.5)
        assert not verdict["within"]
        assert any("missing joints" in problem for problem in verdict["violations"])

    def test_a_non_finite_value_fails(self):
        samples = [(10.0, {"1": math.nan, "2": -1.5854}), (10.1, {"1": 0.0, "2": -1.5854})]
        verdict = envelope_verdict(samples, 10.0, 10.1, reset=RESET, tolerance=0.1, max_gap=0.5)
        assert not verdict["within"]
        assert any("non-finite" in problem for problem in verdict["violations"])

    def test_an_out_of_envelope_sample_fails_with_the_value(self):
        verdict = envelope_verdict(
            _positions([10.0, 10.1, 10.2], joint2=-1.2), 10.0, 10.2, reset=RESET, tolerance=0.1, max_gap=0.5
        )
        assert not verdict["within"]
        assert verdict["worst_rad"] == pytest.approx(0.3854)

    def test_no_samples_at_all_fails(self):
        verdict = envelope_verdict([], 10.0, 10.2, reset=RESET, tolerance=0.1, max_gap=0.5)
        assert not verdict["within"]
        assert verdict["violations"] == ["no arm samples in the window"]

    def test_samples_outside_the_window_are_ignored(self):
        verdict = envelope_verdict(
            _positions([8.0, 8.1] + [10.0, 10.1, 10.2] + [12.0, 12.1]),
            10.0,
            10.2,
            reset=RESET,
            tolerance=0.1,
            max_gap=0.5,
        )
        assert verdict["within"], verdict["violations"]
        assert verdict["samples"] == 3


class TestRestVerdict:
    def test_a_fresh_stationary_stream_is_at_rest(self):
        samples = [(10.0, {"1": 0.0, "2": 0.0}), (10.1, {"1": 0.0, "2": 1e-12}), (10.2, {"1": 0.0, "2": 0.0})]
        verdict = rest_verdict(samples, 10.0, 10.2, joints=JOINTS, max_velocity=1e-6, max_gap=0.5)
        assert verdict["at_rest"], verdict["violations"]
        assert verdict["worst_velocity_rad_s"] == 1e-12

    def test_moving_early_with_a_still_final_frame_is_still_motion(self):
        # the old per-frame max reported the LAST sample's velocity: a spike
        # early in the window followed by a still final frame read as rest
        samples = [
            (10.0, {"1": 0.0, "2": 5.0}),
            (10.1, {"1": 0.0, "2": 1.0}),
            (10.2, {"1": 0.0, "2": 0.0}),
        ]
        verdict = rest_verdict(samples, 10.0, 10.2, joints=JOINTS, max_velocity=1e-6, max_gap=0.5)
        assert not verdict["at_rest"]
        assert verdict["worst_velocity_rad_s"] == 5.0
        assert any("worst |velocity|" in problem for problem in verdict["violations"])

    def test_no_observations_is_a_failure_not_a_clean_zero(self):
        verdict = rest_verdict([], 10.0, 10.2, joints=JOINTS, max_velocity=1e-6, max_gap=0.5)
        assert not verdict["at_rest"]
        assert verdict["violations"] == ["no arm samples in the window"]
        assert verdict["worst_velocity_rad_s"] == math.inf

    def test_a_missing_joint_field_fails(self):
        samples = [(10.0, {"1": 0.0}), (10.1, {"1": 0.0, "2": 0.0})]
        verdict = rest_verdict(samples, 10.0, 10.1, joints=JOINTS, max_velocity=1e-6, max_gap=0.5)
        assert not verdict["at_rest"]
        assert any("missing joints" in problem for problem in verdict["violations"])

    def test_the_freshness_rule_applies_to_the_whole_window(self):
        # samples only cover the middle of the window (0.6 s uncovered at
        # each edge, both beyond the 0.5 s bound)
        samples = [(10.6, {"1": 0.0, "2": 0.0}), (10.65, {"1": 0.0, "2": 0.0})]
        verdict = rest_verdict(samples, 10.0, 11.25, joints=JOINTS, max_velocity=1e-6, max_gap=0.5)
        assert not verdict["at_rest"]
        assert any("after the window start" in problem for problem in verdict["violations"])
        assert any("before the window end" in problem for problem in verdict["violations"])
