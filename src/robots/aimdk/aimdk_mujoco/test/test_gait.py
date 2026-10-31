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
