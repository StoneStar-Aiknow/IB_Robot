from robot_navigation.global_localization_gate import GlobalLocalizationGate


def test_gate_waits_for_delay_and_required_scans_then_triggers_once():
    gate = GlobalLocalizationGate(startup_delay=3.0, required_scans=5)

    for _ in range(4):
        gate.record_scan()

    assert gate.should_trigger(elapsed=10.0) is False

    gate.record_scan()
    assert gate.should_trigger(elapsed=2.9) is False
    assert gate.should_trigger(elapsed=3.0) is True

    gate.mark_triggered()
    assert gate.should_trigger(elapsed=10.0) is False
