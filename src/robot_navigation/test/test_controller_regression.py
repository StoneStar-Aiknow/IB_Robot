import pytest

from robot_navigation.controller_regression import evaluate_regression

GATE = {
    "success_rate_delta_min": 0.0,
    "additional_collisions_max": 0,
    "additional_sustained_oscillations_max": 0,
    "command_limit_violations_max": 0,
    "controller_computation_p95_ms_max_exclusive": 50.0,
}


def _run(result="succeeded", collisions=0, oscillations=0, violations=0, computation=None):
    return {
        "goal_result": result,
        "collision_count": collisions,
        "sustained_oscillation_count": oscillations,
        "command_limit_violation_count": violations,
        "controller_computation_ms": computation or [10.0, 12.0, 15.0],
    }


def test_regression_gate_accepts_non_degrading_candidate():
    assert evaluate_regression([_run()], [_run(computation=[20.0, 30.0, 49.9])], GATE) == []


@pytest.mark.parametrize(
    "candidate,expected",
    [
        ([_run(result="failed")], "success rate"),
        ([_run(collisions=1)], "collision_count"),
        ([_run(oscillations=1)], "sustained_oscillation_count"),
        ([_run(violations=1)], "command limit"),
        ([_run(computation=[50.0])], "p95"),
    ],
)
def test_regression_gate_rejects_each_failure_mode(candidate, expected):
    failures = evaluate_regression([_run()], candidate, GATE)
    assert any(expected in failure for failure in failures)


def test_regression_gate_requires_measured_runs():
    with pytest.raises(ValueError):
        evaluate_regression([], [_run()], GATE)
