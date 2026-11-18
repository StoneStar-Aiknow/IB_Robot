"""The runtime refuses mirroring before it starts when an enforcing CBF could only hold the arm.

The data plane reports in its status whether the hands the platform reports
are the ones its CBF geometry was built for. Every case below has a control
that differs from it in that one input and is allowed.

An installed side the platform reports as none is an empty flange: the data
plane checks it as a phantom hand and reports a match, so it is never refused.
The reverse (a hand where none is configured, or a hand other than the
OmniHand) still is.
"""

from __future__ import annotations

import json

from aimdk_robot import dataplane

MIRROR = "policy_stream"
BOTH_NIMBLE = [1, 1]
# A hand the geometry was not built for: what the data plane reports as a mismatch.
OTHER_RIGHT = [1, 3]


def report(**overrides):
    """The cbf section of a status from a data plane enforcing CBF on two OmniHands."""
    cbf = {
        "enabled": True,
        "mode": "enforce",
        "mirror_mode": MIRROR,
        "mapping_available": True,
        "hands_installed": [True, True],
        "hand_configuration_matches": True,
    }
    cbf.update(overrides)
    return cbf


def test_matching_hands_are_allowed():
    assert dataplane.cbf_hand_refusal(report(), MIRROR, BOTH_NIMBLE) == ""


def test_an_installed_side_reported_as_none_is_an_empty_flange_and_allowed():
    """Two hands configured, the right one not fitted yet: the data plane reports a match."""
    assert dataplane.cbf_hand_refusal(report(), MIRROR, [1, 0]) == ""
    assert dataplane.cbf_hand_refusal(report(), MIRROR, [0, 1]) == ""
    assert dataplane.cbf_hand_refusal(report(), MIRROR, [0, 0]) == ""
    # Control: the same hands under a mismatch report are refused, naming no installed side.
    refusal = dataplane.cbf_hand_refusal(report(hand_configuration_matches=False), MIRROR, [1, 0])
    assert refusal.startswith("CBF enforce hand mismatch: ")
    assert "expected" not in refusal


def test_another_hand_type_is_refused_with_side_expected_and_actual():
    refusal = dataplane.cbf_hand_refusal(report(hand_configuration_matches=False), MIRROR, OTHER_RIGHT)
    assert (
        "right hand expected 1 (nimble_hands) or 0 (none, an empty flange), platform reports 3 (leisai_nimble_hands)"
    ) in refusal
    assert "left" not in refusal
    # Control: the same report with both hands the OmniHand.
    assert dataplane.cbf_hand_refusal(report(), MIRROR, BOTH_NIMBLE) == ""


def test_a_side_with_no_reading_at_all_is_named():
    refusal = dataplane.cbf_hand_refusal(report(hand_configuration_matches=False), MIRROR, [])
    assert "left hand expected 1 (nimble_hands) or 0 (none, an empty flange), platform reports nothing" in refusal
    assert "right hand expected 1 (nimble_hands) or 0 (none, an empty flange), platform reports nothing" in refusal


def test_a_hand_on_a_side_configured_without_one_is_refused():
    cbf = report(hands_installed=[True, False], hand_configuration_matches=False)
    refusal = dataplane.cbf_hand_refusal(cbf, MIRROR, [1, 1])
    assert "right hand expected 0 (none), platform reports 1 (nimble_hands)" in refusal
    assert "no collision body on that side" in refusal
    assert "left" not in refusal
    # Control: the uninstalled side reports none.
    control = report(hands_installed=[True, False])
    assert dataplane.cbf_hand_refusal(control, MIRROR, [1, 0]) == ""


def test_a_mismatch_the_types_do_not_show_still_refuses():
    """Types agree but the hand state lists a hand on an uninstalled side."""
    cbf = report(hands_installed=[True, False], hand_configuration_matches=False)
    refusal = dataplane.cbf_hand_refusal(cbf, MIRROR, [1, 0])
    assert "configured without one" in refusal


def test_a_missing_mapping_is_refused():
    refusal = dataplane.cbf_hand_refusal(report(mapping_available=False), MIRROR, BOTH_NIMBLE)
    assert "thumb mapping" in refusal
    assert dataplane.cbf_hand_refusal(report(mapping_available=True), MIRROR, BOTH_NIMBLE) == ""


def test_a_cbf_that_is_off_or_observing_never_refuses():
    mismatch = {"hand_configuration_matches": False, "mapping_available": False}
    # Control: the same mismatch under enforce is refused.
    assert dataplane.cbf_hand_refusal(report(**mismatch), MIRROR, OTHER_RIGHT)
    assert dataplane.cbf_hand_refusal(report(mode="shadow", **mismatch), MIRROR, OTHER_RIGHT) == ""
    assert dataplane.cbf_hand_refusal({"enabled": False}, MIRROR, OTHER_RIGHT) == ""
    assert dataplane.cbf_hand_refusal({}, MIRROR, OTHER_RIGHT) == ""


def test_only_the_mode_the_cbf_protects_is_refused():
    mismatch = report(hand_configuration_matches=False)
    assert dataplane.cbf_hand_refusal(mismatch, MIRROR, OTHER_RIGHT)
    assert dataplane.cbf_hand_refusal(mismatch, "stream", OTHER_RIGHT) == ""


def test_a_data_plane_without_the_report_behaves_as_before():
    """An older data plane does not report the match: nothing is refused on its account."""
    cbf = report()
    del cbf["hand_configuration_matches"]
    assert dataplane.cbf_hand_refusal(cbf, MIRROR, OTHER_RIGHT) == ""
    # Control: with the report present the same hands are refused.
    assert dataplane.cbf_hand_refusal(report(hand_configuration_matches=False), MIRROR, OTHER_RIGHT)


def test_status_carries_the_cbf_report():
    text = json.dumps({"at": 1.0, "hand_types": OTHER_RIGHT, "cbf": report(hand_configuration_matches=False)})
    status = dataplane.Status.decode(text)
    assert status.cbf["hand_configuration_matches"] is False
    assert dataplane.cbf_hand_refusal(status.cbf, MIRROR, status.hand_types)
    # Control: a status without a cbf section decodes to an empty report.
    assert dataplane.Status.decode(json.dumps({"at": 1.0})).cbf == {}


def test_a_malformed_cbf_section_does_not_drop_the_status():
    for cbf in ("enforce", 5, [["mode", "enforce"]], [1, 2]):
        status = dataplane.Status.decode(json.dumps({"at": 1.0, "control_seq": 7, "hand_types": [1, 0], "cbf": cbf}))
        assert status.cbf == {}, cbf
        assert (status.control_seq, status.hand_types) == (7, [1, 0])
    # Control: a well-formed section in the same status is kept.
    status = dataplane.Status.decode(json.dumps({"at": 1.0, "control_seq": 7, "cbf": report()}))
    assert status.cbf == report()
