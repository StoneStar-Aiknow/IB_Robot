from semantic_mapping.write_policy import SemanticWritePolicy


def test_write_policy_pauses_manipulation_imitation_and_unstable_base():
    policy = SemanticWritePolicy()

    assert policy.admit(mode="mapping", base_stable=True, frame_scan_epoch=2, active_scan_epoch=2).allowed
    assert not policy.admit(mode="manipulation", base_stable=True, frame_scan_epoch=2, active_scan_epoch=2).allowed
    assert not policy.admit(mode="imitation", base_stable=True, frame_scan_epoch=2, active_scan_epoch=2).allowed
    assert not policy.admit(mode="navigation", base_stable=False, frame_scan_epoch=2, active_scan_epoch=2).allowed
    assert not policy.admit(mode="mapping", base_stable=True, frame_scan_epoch=1, active_scan_epoch=2).allowed


def test_explicit_override_allows_diagnostic_operator_write():
    admission = SemanticWritePolicy().admit(
        mode="manipulation",
        base_stable=False,
        frame_scan_epoch=1,
        active_scan_epoch=2,
        override=True,
    )

    assert admission.allowed
