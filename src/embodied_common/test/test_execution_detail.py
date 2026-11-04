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
        ExecutionDetail(schema_version=2)


@pytest.mark.parametrize("missing", list(ExecutionDetail().to_dict()))
def test_partial_detail_does_not_claim_version_one(missing):
    value = ExecutionDetail(1, "verify", "SUCCEEDED", "", "", "NOT_NEEDED", 0, "NONE").to_dict()
    del value[missing]
    assert ExecutionDetail.from_value(value).schema_version == 0
