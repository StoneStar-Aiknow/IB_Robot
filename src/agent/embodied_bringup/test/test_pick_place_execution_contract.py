"""Text -> Agent -> Gateway -> pick -> custody -> place -> query/events/TTS."""

import json

import numpy as np
from std_msgs.msg import String

from ibrobot_msgs.srv import AuthorizeHeldReset

from .grasp_contract_rig import ContractRig, wait_until


def test_text_pick_place_completes_both_steps_and_releases_custody(tmp_path):
    rig = ContractRig(tmp_path, pick_place=True)
    try:
        request_id, envelope = rig.submit()
        terminal = rig.terminal(request_id)
        assert terminal["state"] == "SUCCEEDED", terminal
        # The planner sees only each skill's summary; the X2 variants are where
        # it learns that "put X in Y" is pick then place on the same target.
        planner_input = json.loads(rig.requests[0]["messages"][1]["content"])
        summaries = {skill["name"]: skill["summary"] for skill in planner_input["catalog"]["skills"]}
        assert "place_in_container" in summaries["pick_object"]
        assert "same target_name" in summaries["pick_object"]
        assert "pick_object" in summaries["place_in_container"]
        assert "same workflow" in summaries["place_in_container"]
        query = rig.query(request_id)
        assert query["state"] == "SUCCEEDED"
        assert query["current_skill"] == "place_in_container"
        assert query["workflow_step_index"] == 1
        assert query["execution_detail"]["phase"] == "verify_place"
        assert query["execution_detail"]["phase_state"] == "SUCCEEDED"
        assert rig.mock.queries == ["an apple"]
        assert rig.mock.container_queries == ["the blue bowl on the left"]
        assert rig.mock.closes == rig.mock.releases == rig.mock.verifications == 1
        assert rig.mock.moves == ["approach", "grasp", "lift", "transfer", "descend", "retreat", "home"]
        assert not rig.mock.gripped and np.allclose(rig.mock.position, 0)
        assert rig.pick._custody.custody.state == "released"
        assert not rig.pick._custody.custody.occupied
        response = rig.agent._bridge.get_execution_events(
            request_id=request_id, execution_id=query["execution_id"], after_sequence=0, timeout_sec=2.0
        )
        assert response["found"]
        events = response["events"]
        assert {event.current_skill for event in events if event.current_skill} == {"pick_object", "place_in_container"}
        assert {event.workflow_step_index for event in events if event.current_skill} == {0, 1}
        assert events[-1].sequence == query["last_event_sequence"]
        assert events[-1].execution_detail.phase == "verify_place"
        wait_until(lambda: "物体已放入指定容器并通过验证。" in rig.speech)
        wait_until(lambda: rig.played)
        rig.input.publish(String(data=json.dumps(envelope, ensure_ascii=False)))
        wait_until(lambda: len(rig.responses) >= 2)
        assert len(rig.requests) == 1 and rig.mock.closes == rig.mock.releases == 1
        print(
            json.dumps(
                {
                    "state": query["state"],
                    "last_skill": query["current_skill"],
                    "phase": query["execution_detail"]["phase"],
                    "moves": rig.mock.moves,
                    "custody": rig.pick._custody.custody.state,
                    "container": rig.mock.container_queries[0],
                    "releases": rig.mock.releases,
                }
            )
        )
    finally:
        rig.close()


def test_a_held_object_blocks_other_requests_until_it_is_put_back(tmp_path):
    """A failed place leaves the object held; nothing else may move the arm until pick puts it back."""
    rig = ContractRig(tmp_path, pick_place=True, place_scenario="container_missing", held_lease_s=4.0)
    try:
        first, _ = rig.submit()
        terminal = rig.terminal(first)
        assert terminal["state"] == "FAILED", terminal
        assert rig.query(first)["execution_detail"]["reason_code"] == "CONTAINER_NOT_FOUND"
        custody = rig.pick._custody.custody
        assert custody.state == "picked" and rig.mock.gripped
        wait_until(lambda: rig.gateway._held_object_custody()[1] == "picked")

        second, _ = rig.submit()
        terminal = rig.terminal(second)
        assert terminal["state"] == "FAILED", terminal
        assert rig.query(second)["error_code"] == "SKILL_BUSY", rig.query(second)
        assert rig.mock.closes == 1 and rig.mock.container_queries == ["the blue bowl on the left"]

        wait_until(lambda: custody.state == "released", 15)
        assert custody.reason == "LEASE_EXPIRED_PUT_BACK"
        assert not rig.mock.gripped and np.allclose(rig.mock.position, 0)
        wait_until(lambda: rig.gateway._held_object_custody() == ("", "", ""))
    finally:
        rig.close()


def test_stopped_place_failure_leaves_gateway_idle_and_manual_reset_can_clear_uncertain(tmp_path):
    rig = ContractRig(tmp_path, pick_place=True, place_scenario="transfer_failed")
    try:
        request_id, _ = rig.submit()
        terminal = rig.terminal(request_id)
        assert terminal["state"] == "FAILED", terminal
        query = rig.query(request_id)
        assert query["execution_detail"]["recovery_state"] == "FAILED"
        custody = rig.pick._custody.custody
        assert custody.state == "uncertain"
        wait_until(lambda: rig.gateway._held_object_custody()[1] == "uncertain")
        assert rig.gateway._gateway_lease.owner is None
        assert not rig.gateway._runtime_snapshot().is_busy
        assert not rig.gateway._execution_store.has_unresolved_unknown()

        # Simulate the operator removing the object. Authorization is read-only;
        # the test feedback still says the arm is at the lifted handover pose.
        rig.mock.gripped = False
        response = rig.gateway._authorize_held_reset(
            AuthorizeHeldReset.Request(
                schema_version=1,
                request_id="operator-check-1",
                custody_id=custody.held.custody_id,
                operator_confirmed_empty=True,
                operator_note="operator removed the object and checked the hand",
            ),
            AuthorizeHeldReset.Response(),
        )
        assert response.success, response
        assert rig.gateway._gateway_lease.owner is None
        assert rig.gateway._runtime_snapshot().held_object_manual_reset
        assert custody.state == "uncertain"
        # Feedback from the subsequent recovery pose, without physical hardware.
        rig.mock.position = np.asarray(custody.held.home_positions)
        wait_until(lambda: custody.state == "released")
        assert custody.reason == "MANUAL_RESET_COMPLETE"
        wait_until(lambda: rig.gateway._held_object_custody() == ("", "", ""))
    finally:
        rig.close()
