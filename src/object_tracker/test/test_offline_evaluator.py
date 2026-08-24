import pytest

from object_tracker.offline_evaluator import evaluate


def test_evaluate_reports_identity_rate_and_position_error():
    rows = [
        {
            "stamp_ns": "1000000000",
            "session_id": "session-1",
            "object_id": "object-1",
            "measured": "true",
            "x": "0.0",
            "y": "0.0",
            "gt_x": "0.0",
            "gt_y": "0.0",
        },
        {
            "stamp_ns": "2000000000",
            "session_id": "session-1",
            "object_id": "object-1",
            "measured": "false",
            "x": "1.2",
            "y": "0.0",
            "gt_x": "1.0",
            "gt_y": "0.0",
        },
    ]

    summary = evaluate(rows)

    assert summary["identity_retained"]
    assert summary["output_rate_hz"] == pytest.approx(1.0)
    assert summary["measured_ratio"] == pytest.approx(0.5)
    assert summary["position_rmse_m"] == pytest.approx(0.2 / 2**0.5)


def test_evaluate_rejects_out_of_order_rows():
    rows = [
        {"stamp_ns": "2", "session_id": "s", "object_id": "o"},
        {"stamp_ns": "1", "session_id": "s", "object_id": "o"},
    ]

    with pytest.raises(ValueError, match="ordered"):
        evaluate(rows)
