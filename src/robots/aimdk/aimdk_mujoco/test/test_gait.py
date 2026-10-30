"""A translating torso alone must never qualify as walking."""

from aimdk_mujoco.gait import GaitMonitor


def monitor(samples):
    # These tests exercise the independent reduction, not a fabricated simulation/node.
    class Samples:
        pass

    recorded = Samples()
    recorded.samples = samples
    return GaitMonitor.summary(recorded)


def test_continuous_contact_translation_is_not_walking():
    result = monitor([{"clearance": [0, 0], "load": [200, 200], "slip": [0.2, 0.2]}] * 500)
    assert not result["passed"]
    assert result["left"]["swing_count"] == 0
    assert result["left"]["loaded_slip_median_m_s"] == 0.2


def test_low_clearance_shuffle_is_reported_not_certified():
    samples = []
    for i in range(500):
        left = i % 20 < 10
        samples.append(
            {"clearance": [0.006, 0] if left else [0, 0.006], "load": [0, 400] if left else [400, 0], "slip": [0, 0]}
        )
    result = monitor(samples)
    assert not result["passed"]
    assert result["left"]["swing_count"] > 3
    assert result["classification"] == "shallow_shuffle_or_sliding"


def test_alternating_clear_steps_pass():
    samples = []
    for i in range(500):
        left = i % 20 < 10
        samples.append(
            {"clearance": [0.03, 0] if left else [0, 0.03], "load": [0, 400] if left else [400, 0], "slip": [0, 0]}
        )
    result = monitor(samples)
    assert result["passed"]
    assert result["single_support_fraction"] == 1.0


def test_lifting_feet_does_not_excuse_large_loaded_slip():
    samples = []
    for i in range(500):
        left = i % 20 < 10
        samples.append(
            {"clearance": [0.03, 0] if left else [0, 0.03], "load": [0, 400] if left else [400, 0], "slip": [0.2, 0.2]}
        )
    result = monitor(samples)
    assert not result["passed"]
