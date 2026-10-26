import json
from dataclasses import asdict, replace

import pytest

from ibrobot_tracing.clock_calibration import CalibrationPolicy, CaptureIdentity, ClockSample, PtpStatus
from ibrobot_tracing.rtp_metrics import analyze_rtp_metrics, read_metrics
from ibrobot_tracing.rtp_metrics import main as metrics_main


def _calibration(path, machine, *, offset=0, boot=None, fault=False):
    identity = CaptureIdentity("rtp", machine, boot or f"boot-{machine}", "eth0", "/dev/ptp0", "gm", 97)
    samples = []
    for index in range(12):
        mono = 10_000_000_000 + offset + index * 20_000_000
        phc = 1_700_000_000_000_000_000 + index * 20_000_000
        status = PtpStatus(
            mono - 1000, 97, "source", "gm", "SLAVE", "HARDWARE", "eth0", True, 0, 100, phc - 1_000_000, 0
        )
        if fault:
            status = replace(status, port_state="FAULTY")
        samples.append(ClockSample(mono, mono - 1, phc + 10_000_000_000, mono + 1, phc, 100, status))
    path.write_text(
        "\n".join(
            json.dumps(value)
            for value in [
                {
                    "kind": "clock_calibration",
                    "schema_version": 1,
                    "identity": asdict(identity),
                    "policy": asdict(CalibrationPolicy()),
                },
                *[{"kind": "sample", "sample": asdict(sample)} for sample in samples],
            ]
        )
        + "\n"
    )


def _write(path, machine, rows, *, footer=True):
    header = {
        "kind": "rtp_metrics",
        "schema_version": 1,
        "capture_session": "rtp",
        "machine_id": machine,
        "boot_id": f"boot-{machine}",
        "process_id": 1,
        "process_instance": machine + "-process",
        "clock": "monotonic_ns",
    }
    values = [header, *rows]
    if footer:
        values.append({"kind": "end", "dropped_records": 0})
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n")


def _row(kind, **overrides):
    values = {
        "kind": kind,
        "pipeline_id": "policy",
        "session_id": "policy-session",
        "session_generation": 1,
        "observation_key": "observation.images.top",
        "stream_id": "top",
        "ssrc": 9,
        "rtp_timestamp": 100,
        "last_sequence": 10,
        "first_sequence": 9,
        "packet_count": 2,
    }
    if kind == "send":
        values.update(
            send_start_ns=10_080_000_000,
            send_end_ns=10_081_000_000,
            capture_timestamp_ns=1234,
            encode_input_capture_ns=1234,
            encode_start_ns=10_078_000_000,
            encode_end_ns=10_079_000_000,
        )
    elif kind == "receive":
        values.update(receive_ns=20_082_000_000, reassembly_end_ns=20_085_000_000)
    else:
        values.update(
            decode_start_ns=20_085_000_000,
            decode_end_ns=20_086_000_000,
            input_capture_timestamp_ns=1234,
            output_capture_timestamps_ns=[1234],
        )
    return {**values, **overrides}


@pytest.fixture
def inputs(tmp_path):
    sender, receiver = tmp_path / "send.jsonl", tmp_path / "receive.jsonl"
    a, b = tmp_path / "a-cal.jsonl", tmp_path / "b-cal.jsonl"
    _calibration(a, "sender")
    _calibration(b, "receiver", offset=10_000_000_000)
    _write(sender, "sender", [_row("send")])
    _write(receiver, "receiver", [_row("receive"), _row("decode_call")])
    return [(sender, a), (receiver, b)]


def test_same_access_unit_is_mapped_and_local_stages_preserved(inputs):
    result = analyze_rtp_metrics(inputs)
    assert result["counts"] == {"valid": 1}
    frame = result["frames"][0]
    assert frame["frame_delivery_ns"] == 2_000_000  # clocks differ by ten seconds
    assert frame["uncertainty_ns"] < 100_000
    assert frame["send_call_ns"] == frame["encode_call_ns"] == frame["decode_call_ns"] == 1_000_000
    assert frame["receive_queue_and_reassembly_ns"] == 3_000_000
    assert "network_latency_ns" not in frame


@pytest.mark.parametrize(
    "key,value",
    [
        ("last_sequence", 11),
        ("rtp_timestamp", 101),
        ("session_generation", 2),
        ("session_id", "other"),
        ("ssrc", 10),
        ("stream_id", "wrist"),
    ],
)
def test_different_frames_epochs_and_sequences_never_pair(inputs, key, value):
    _write(inputs[1][0], "receiver", [_row("receive", **{key: value})])
    result = analyze_rtp_metrics(inputs)
    assert result["counts"] == {"unmatched_frame": 2}
    assert all("frame_delivery_ns" not in row for row in result["frames"])


def test_repeated_frame_is_ambiguous(inputs):
    _write(inputs[0][0], "sender", [_row("send"), _row("send")])
    assert analyze_rtp_metrics(inputs)["counts"] == {"ambiguous_frame": 1}


def test_truncated_access_unit_is_not_a_valid_delivery(inputs):
    _write(inputs[1][0], "receiver", [_row("receive", first_sequence=10, packet_count=1)])
    assert analyze_rtp_metrics(inputs)["counts"] == {"incomplete_access_unit": 1}


@pytest.mark.parametrize("mode", ["boot", "fault", "outside", "negative", "local_order"])
def test_invalid_measurement_is_not_emitted(inputs, mode):
    if mode == "boot":
        _calibration(inputs[1][1], "receiver", offset=10_000_000_000, boot="wrong")
    elif mode == "fault":
        _calibration(inputs[1][1], "receiver", offset=10_000_000_000, fault=True)
    elif mode == "outside":
        _write(inputs[1][0], "receiver", [_row("receive", receive_ns=30_000_000_000, reassembly_end_ns=30_001_000_000)])
    elif mode == "negative":
        _write(inputs[1][0], "receiver", [_row("receive", receive_ns=20_075_000_000)])
    else:
        _write(inputs[0][0], "sender", [_row("send", send_end_ns=10_079_000_000)])
    row = analyze_rtp_metrics(inputs)["frames"][0]
    assert row["status"] == "unavailable"
    assert "frame_delivery_ns" not in row


def test_decode_call_is_not_misrepresented_as_delayed_frame_latency(inputs):
    _write(inputs[1][0], "receiver", [_row("receive"), _row("decode_call", output_capture_timestamps_ns=[1000])])
    frame = analyze_rtp_metrics(inputs)["frames"][0]
    assert frame["decode_call_ns"] == 1_000_000
    assert "decode_latency_ns" not in frame
    assert frame["records"]["decode_call"][0]["record"]["output_capture_timestamps_ns"] == [1000]


def test_delayed_encoder_output_omits_unrelated_encode_duration(inputs):
    _write(inputs[0][0], "sender", [_row("send", encode_input_capture_ns=9999)])
    assert "encode_call_ns" not in analyze_rtp_metrics(inputs)["frames"][0]


def test_missing_footer_is_visible_and_input_validation(inputs):
    _write(inputs[0][0], "sender", [_row("send")], footer=False)
    assert analyze_rtp_metrics(inputs)["warnings"]
    with pytest.raises(ValueError, match="duplicate"):
        analyze_rtp_metrics([inputs[0], inputs[0]])
    _write(inputs[0][0], "sender", [_row("send", ssrc=True)])
    with pytest.raises(ValueError, match="invalid RTP field"):
        read_metrics(inputs[0][0])


def test_rtp_metrics_cli_and_no_overwrite(inputs, tmp_path):
    output = tmp_path / "report.json"
    args = ["--output", str(output)]
    for journal, calibration in inputs:
        args.extend(["--input", str(journal), str(calibration)])
    assert metrics_main(args) == 0
    assert json.loads(output.read_text())["counts"] == {"valid": 1}
    assert metrics_main(args) == 3


def test_truncated_and_invalid_headers_fail_closed(tmp_path):
    path = tmp_path / "bad.jsonl"
    for value in ("", "[]\n", "null\n", '{"kind": "rtp_metrics"}'):
        path.write_text(value)
        with pytest.raises(ValueError):
            read_metrics(path)
