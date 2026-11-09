import pytest

from embodied_common.execution_detail import ExecutionDetail, execution_context, fill_execution_detail, short_message
from ibrobot_msgs.msg import ExecutionDetail as WireDetail


@pytest.mark.parametrize("length", [0, 299, 300, 301, 20000])
def test_summary_bounds_the_entire_unicode_message(length):
    message = short_message("苹" * length)
    assert len(message) == min(300, length)
    assert short_message(message) == message
    assert "\n" not in short_message("one\n two")


def test_ros_roundtrip_retains_all_facts():
    detail = ExecutionDetail(1, "recover", "SUCCEEDED", "close", "EMPTY_GRASP", "COMPLETED", 2, "WAIT_USER")
    wire = WireDetail()
    fill_execution_detail(wire, detail)
    assert ExecutionDetail.from_value(wire) == detail
    assert ExecutionDetail.from_value(detail.to_dict()) == detail


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1, "phase_state": "NEXT"},
        {"schema_version": 1, "retry_count": -1},
        {"schema_version": 1, "suggested_action": []},
        {"schema_version": 1, "phase": "x" * 65},
    ],
)
def test_invalid_presentation_detail_degrades_without_throwing(value):
    assert ExecutionDetail.from_value(value).schema_version == 0
    assert execution_context({"execution_detail": value})["execution_detail"]["schema_version"] == 0


def test_producer_rejects_invalid_contract():
    with pytest.raises(ValueError):
        ExecutionDetail(schema_version=1, retry_count=True)
    with pytest.raises(ValueError):
        ExecutionDetail(schema_version=3)


@pytest.mark.parametrize("missing", list(ExecutionDetail().to_dict())[:8])
def test_partial_detail_does_not_claim_version_one(missing):
    value = ExecutionDetail(1, "verify", "SUCCEEDED", "", "", "NOT_NEEDED", 0, "NONE").to_dict()
    del value[missing]
    assert ExecutionDetail.from_value(value).schema_version == 0


def test_v1_original_fields_and_positionals_preserve_generic_facts():
    detail = ExecutionDetail(1, "release", "SUCCEEDED", "", "", "NOT_NEEDED", 0, "NONE")
    old = dict(list(detail.to_dict().items())[:8])
    assert ExecutionDetail.from_value(old) == detail
    old.update(release_status="RELEASED", verification_status="SUCCESS", place_succeeded=True)
    assert ExecutionDetail.from_value(old) == detail
    assert detail.release_status == detail.verification_status == "UNKNOWN"


def test_v2_typed_serialization_roundtrip_preserves_context_and_facts():
    from rclpy.serialization import deserialize_message, serialize_message

    from embodied_common.execution_detail import fill_execution_context
    from ibrobot_msgs.action import ExecuteProgram, SkillCommand

    detail = ExecutionDetail(
        2,
        "cleanup",
        "FAILED",
        "cleanup",
        "CLEANUP_FAILED",
        "UNKNOWN",
        0,
        "MANUAL_CHECK",
        "RELEASED",
        "SUCCESS",
        False,
    )
    result = ExecuteProgram.Result()
    fill_execution_context(
        result,
        {
            "current_skill": "place_in_container",
            "workflow_step_index": 3,
            "execution_detail": detail.to_dict(),
        },
    )
    restored = deserialize_message(serialize_message(result), ExecuteProgram.Result)
    assert ExecutionDetail.from_value(restored.execution_detail) == detail
    assert restored.current_skill == "place_in_container" and restored.workflow_step_index == 3

    skill_result = SkillCommand.Result()
    fill_execution_detail(skill_result.execution_detail, detail)
    restored_skill = deserialize_message(serialize_message(skill_result), SkillCommand.Result)
    assert ExecutionDetail.from_value(restored_skill.execution_detail) == detail


@pytest.mark.parametrize("missing", list(ExecutionDetail().to_dict()))
def test_v2_requires_complete_facts(missing):
    value = ExecutionDetail(schema_version=2).to_dict()
    del value[missing]
    assert ExecutionDetail.from_value(value).schema_version == 0


def test_explicit_place_numeric_projection_matches_generated_action():
    from embodied_common.execution_detail import place_result_execution_detail
    from ibrobot_msgs.action import PlaceObject

    for release, expected in [
        (PlaceObject.Result.RELEASE_NOT_RELEASED, "NOT_RELEASED"),
        (PlaceObject.Result.RELEASE_RELEASED, "RELEASED"),
        (PlaceObject.Result.RELEASE_UNKNOWN, "UNKNOWN"),
        (255, "UNKNOWN"),
    ]:
        for verification, expected_verification in [
            (PlaceObject.Result.VERIFICATION_NOT_RUN, "NOT_RUN"),
            (PlaceObject.Result.VERIFICATION_SUCCESS, "SUCCESS"),
            (PlaceObject.Result.VERIFICATION_FAILED, "FAILED"),
            (PlaceObject.Result.VERIFICATION_UNCERTAIN, "UNCERTAIN"),
            (255, "UNKNOWN"),
        ]:
            result = PlaceObject.Result(release_status=release, verification_status=verification)
            fill_execution_detail(
                result.execution_detail,
                ExecutionDetail(
                    1,
                    "recover",
                    "FAILED",
                    "retreat",
                    "MOTION_FAILED",
                    "UNKNOWN",
                    2,
                    "MANUAL_CHECK",
                ),
            )
            detail = place_result_execution_detail(result)
            assert detail.release_status == expected
            assert detail.verification_status == expected_verification
            assert detail.recovery_state == "UNKNOWN" and detail.retry_count == 2
            assert detail.phase == "recover" and not detail.place_succeeded
