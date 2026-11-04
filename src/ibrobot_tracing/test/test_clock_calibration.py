import json
from dataclasses import asdict, replace

import pytest

from ibrobot_tracing.clock_calibration import (
    CalibrationPolicy,
    CaptureIdentity,
    ClockCalibration,
    ClockSample,
    MappedTime,
    PtpStatus,
    load_calibration,
    reference_difference,
    sample_quality,
)
from ibrobot_tracing.clock_capture import parse_pmc

IDENTITY = CaptureIdentity("run-1", "host-1", "boot-1", "eth0", "/dev/ptp0", "aabbcc.fffe.ddeeff", 97)
POLICY = CalibrationPolicy()
EPOCH = 1_780_000_000_000_000_000


def samples(count=8):
    result = []
    for i in range(count):
        mono = 10_000_000_000 + i * 20_000_000
        phc = EPOCH + i * 20_000_000 + i * 100
        status = PtpStatus(
            mono - 1000,
            97,
            "112233.fffe.445566",
            IDENTITY.grandmaster_identity,
            "SLAVE",
            "HARDWARE",
            "eth0",
            True,
            -20,
            600,
            phc - 10_000_000,
            0,
        )
        result.append(ClockSample(mono, mono - 2000, phc + 37_000_000_000, mono + 900, phc, 1000, status))
    return result


def mapped(calibration, stamp, clock="monotonic_ns", **kwargs):
    return calibration.map_time(
        stamp, clock=clock, machine_id=kwargs.get("machine_id", "host-1"), boot_id=kwargs.get("boot_id", "boot-1")
    )


@pytest.mark.parametrize("clock", ["monotonic_ns", "monotonic_raw_ns", "realtime_ns"])
def test_integer_mapping_preserves_epoch_nanoseconds_and_drift(clock):
    data = samples()
    calibration = ClockCalibration(IDENTITY, data, POLICY)
    middle = (getattr(data[4], clock) + getattr(data[5], clock)) // 2
    result = mapped(calibration, middle, clock)
    assert result.valid
    assert result.timestamp_ns == (data[4].phc_ns + data[5].phc_ns) // 2
    assert 20_000 < result.uncertainty_ns < 50_000


def test_reused_poll_does_not_count_as_multiple_lock_updates():
    data = samples()
    data = [replace(sample, status=data[0].status) for sample in data]
    assert mapped(ClockCalibration(IDENTITY, data, POLICY), data[5].monotonic_ns).reason == "settling"


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"port_state": "LISTENING"}, "ptp_not_locked"),
        ({"timestamping": "SOFTWARE"}, "wrong_timestamp_source"),
        ({"grandmaster_identity": "another"}, "reference_changed"),
        ({"domain": 5}, "reference_changed"),
        ({"observed_monotonic_ns": 1}, "stale_ptp_status"),
        ({"ingress_time_ns": 1}, "stale_sync"),
        ({"mean_path_delay_ns": -1}, "negative_path_delay"),
        ({"master_offset_ns": 100_000}, "uncertainty_exceeded"),
    ],
)
def test_status_quality_rejects_unusable_snapshots(updates, reason):
    sample = samples()[4]
    sample = replace(sample, status=replace(sample.status, **updates))
    assert sample_quality(sample, IDENTITY, POLICY)[1] == reason


def test_fault_is_not_bridged_and_relock_requires_new_updates():
    data = samples(12)
    data[5] = replace(data[5], status=None)
    cal = ClockCalibration(IDENTITY, data, POLICY)
    assert mapped(cal, data[4].monotonic_ns + 10_000_000).reason == "missing_ptp_status"
    assert mapped(cal, data[6].monotonic_ns).reason == "settling"
    assert mapped(cal, data[9].monotonic_ns).valid


@pytest.mark.parametrize("field", ["phc_ns", "realtime_ns", "boottime_ns"])
def test_clock_steps_and_suspend_are_rejected(field):
    data = samples()
    for i in range(5, len(data)):
        data[i] = replace(data[i], **{field: getattr(data[i], field) + 1_000_000_000})
    clock = "realtime_ns" if field == "realtime_ns" else "monotonic_ns"
    result = mapped(ClockCalibration(IDENTITY, data, POLICY), getattr(data[4], clock) + 1, clock)
    assert not result.valid


def test_no_extrapolation_or_cross_host_reuse():
    data = samples()
    cal = ClockCalibration(IDENTITY, data, POLICY)
    assert mapped(cal, data[-1].monotonic_ns + 1).reason == "outside_capture"
    assert mapped(cal, data[0].monotonic_ns - 1).reason == "outside_capture"
    assert mapped(cal, data[4].monotonic_ns, boot_id="other").reason == "host_or_boot_mismatch"
    assert mapped(cal, data[4].monotonic_ns, machine_id="other").reason == "host_or_boot_mismatch"
    assert not mapped(ClockCalibration(IDENTITY, [], POLICY), 0).valid


def test_gap_reboot_and_ambiguous_realtime_epoch():
    data = samples()
    cal = ClockCalibration(IDENTITY, data[:4] + data[6:], POLICY)
    # Tighten max gap to demonstrate no interpolation over missing collection.
    cal = ClockCalibration(IDENTITY, cal.samples, replace(POLICY, max_gap_ns=30_000_000))
    assert mapped(cal, data[4].monotonic_ns).reason == "sample_gap"
    with pytest.raises(ValueError, match="strictly monotonic"):
        ClockCalibration(IDENTITY, [data[2], data[1]], POLICY)
    data[5] = replace(data[5], realtime_ns=data[3].realtime_ns)
    cal = ClockCalibration(IDENTITY, data, POLICY)
    assert mapped(cal, data[3].realtime_ns + 1, "realtime_ns").reason == "ambiguous_clock_epoch"


def test_reference_step_requires_a_new_capture_epoch():
    data = samples(12)
    for i in range(5, len(data)):
        data[i] = replace(data[i], phc_ns=data[i].phc_ns + 1_000_000)
    cal = ClockCalibration(IDENTITY, data, POLICY)
    assert mapped(cal, data[9].monotonic_ns).reason == "reference_discontinuity"


def test_combined_error_budget_and_reference_identity():
    cal = ClockCalibration(IDENTITY, samples(), POLICY)
    other = ClockCalibration(replace(IDENTITY, session_id="other"), samples(), POLICY)
    start, end = MappedTime(100, 30000), MappedTime(120, 30000)
    assert reference_difference(start, end, cal, cal) == MappedTime(20, 60000)
    assert reference_difference(start, end, cal, other).reason == "reference_mismatch"
    assert reference_difference(start, end, cal, cal, max_uncertainty_ns=50000).reason == "uncertainty_exceeded"
    assert reference_difference(MappedTime(None, None, "fault"), end, cal, cal).reason == "fault"


def test_capture_roundtrip_and_strict_timestamps(tmp_path):
    path = tmp_path / "capture.jsonl"
    header = {"kind": "clock_calibration", "schema_version": 1, "identity": asdict(IDENTITY), "policy": asdict(POLICY)}
    path.write_text(
        "\n".join(json.dumps(item) for item in [header, *[{"kind": "sample", "sample": asdict(s)} for s in samples()]])
    )
    assert load_calibration(path).summary()["intervals"]["valid"] > 0
    with pytest.raises(ValueError, match="int64"):
        replace(samples()[0], phc_ns=True)
    with pytest.raises(ValueError):
        replace(POLICY, max_clock_rate_ppm=0)


@pytest.mark.parametrize("contents", ["", "[]", "null", "{}", '{"kind":"clock_calibration","schema_version":1}'])
def test_invalid_capture_has_contextual_value_error(tmp_path, contents):
    path = tmp_path / "bad-clock.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match="invalid calibration capture.*bad-clock.jsonl:1"):
        load_calibration(path)


@pytest.mark.parametrize(
    "record", [{}, [], {"kind": "sample", "sample": {}}, {"kind": "sample", "sample": {"status": []}}]
)
def test_invalid_sample_has_line_number(tmp_path, record):
    path = tmp_path / "bad-sample.jsonl"
    header = {"kind": "clock_calibration", "schema_version": 1, "identity": asdict(IDENTITY), "policy": asdict(POLICY)}
    path.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="bad-sample.jsonl:2"):
        load_calibration(path)


def test_indexed_nonmonotonic_lookup_matches_original_closed_interval_scan():
    import random

    class LinearIntervals:
        def __init__(self, ticks):
            self.ticks = ticks

        def find(self, timestamp, matches):
            matches.extend(
                i
                for i, (a, b) in enumerate(zip(self.ticks, self.ticks[1:], strict=False))
                if min(a, b) <= timestamp <= max(a, b)
            )

    rng = random.Random(523)
    for _ in range(15):
        data = samples(100)
        ticks = [data[0].realtime_ns]
        for _sample in data[1:]:
            ticks.append(ticks[-1] + rng.choice([20_000_000, 20_000_000, 0, -40_000_000]))
        data = [replace(sample, realtime_ns=tick) for sample, tick in zip(data, ticks, strict=True)]
        indexed = ClockCalibration(IDENTITY, data, POLICY)
        linear = ClockCalibration(IDENTITY, data, POLICY)
        linear._interval_indexes["realtime_ns"] = LinearIntervals(ticks)
        for point in ticks + [tick + 1 for tick in ticks] + [tick - 1 for tick in ticks]:
            assert mapped(indexed, point, "realtime_ns") == mapped(linear, point, "realtime_ns")


def test_bad_calibration_cli_returns_error_not_traceback(tmp_path, capsys):
    from ibrobot_tracing.cli import main
    from ibrobot_tracing.rtp_metrics import main as rtp_main

    bad = tmp_path / "empty-clock.jsonl"
    bad.write_text("")
    inputs = ["--input", str(tmp_path / "unused"), str(bad)] * 2
    assert main(["merge", *inputs, "--output", str(tmp_path / "out.ibtrace")]) == 3
    assert rtp_main([*inputs, "--output", str(tmp_path / "out.json")]) == 3
    assert "invalid calibration capture" in capsys.readouterr().err


PMC = """
 abc-0 seq 0 RESPONSE MANAGEMENT DEFAULT_DATA_SET
 clockIdentity abc
 domainNumber 97
 abc-1 seq 1 RESPONSE MANAGEMENT PORT_DATA_SET
 portIdentity abc-1
 portState SLAVE
 abc-0 seq 2 RESPONSE MANAGEMENT TIME_STATUS_NP
 gmIdentity aabbcc.fffe.ddeeff
 gmPresent true
 master_offset -5
 ingress_time 1780000000000000000
 gmTimeBaseIndicator 0
 abc-0 seq 3 RESPONSE MANAGEMENT CURRENT_DATA_SET
 meanPathDelay 572.5
 abc-1 seq 4 RESPONSE MANAGEMENT PORT_PROPERTIES_NP
 portIdentity abc-1
 portState SLAVE
 timestamping HARDWARE
 interface eth0
"""


def test_pmc_parser_and_ambiguous_or_incomplete_replies():
    result = parse_pmc(PMC, 100)
    assert result.mean_path_delay_ns == 573 and result.master_offset_ns == -5
    with pytest.raises(ValueError):
        parse_pmc(PMC + PMC, 100)
    with pytest.raises(ValueError):
        parse_pmc("sending: GET TIME_STATUS_NP\n", 100)
    with pytest.raises(ValueError):
        parse_pmc(PMC.replace("572.5", "nan"), 100)
    with pytest.raises(ValueError, match="changed state"):
        parse_pmc(PMC.replace("portState SLAVE\n timestamping", "portState MASTER\n timestamping"), 100)
