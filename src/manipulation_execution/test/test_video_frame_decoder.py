"""Video backends preserve the upstream replaceable decoder contract."""

from types import SimpleNamespace

import pytest
from sensor_msgs.msg import CompressedImage

from manipulation_execution.frame_decoder import create_frame_decoder


@pytest.mark.parametrize("name", ["auto", "dvpp"])
@pytest.mark.parametrize("params", [{"device_id": -1}, {"device_id": True}, {"device_id": 0.5}, {"extra": 1}])
def test_hardware_decoder_rejects_invalid_parameters(name, params):
    with pytest.raises(ValueError):
        create_frame_decoder(name, params)


def test_unknown_decoder_still_fails_at_startup():
    with pytest.raises(ValueError, match="unknown decoder"):
        create_frame_decoder("unregistered", {})


def test_legacy_cpu_decoder_keeps_its_parameter_contract():
    decoder = create_frame_decoder("mock_cpu", {})
    assert decoder.decode(CompressedImage()) is None
    decoder.close()
    with pytest.raises(ValueError, match="takes no parameters"):
        create_frame_decoder("mock_cpu", {"device_id": 0})


def test_strict_dvpp_reports_bad_frame_without_violating_decoder_interface():
    decoder = create_frame_decoder("dvpp", {"device_id": 2})

    def fail_decode(_message):
        raise RuntimeError("device decode failed")

    closed = []
    decoder._transport = SimpleNamespace(
        decode=fail_decode, close=lambda: closed.append(True), last_error="", cpu_frames=0, dvpp_frames=0
    )
    assert decoder.decode(CompressedImage()) is None
    assert decoder.last_error == "RuntimeError: device decode failed"
    assert decoder.cpu_frames == decoder.dvpp_frames == 0
    decoder.close()
    assert closed == [True]


def test_auto_decoder_exposes_fallback_diagnostics_and_device():
    decoder = create_frame_decoder("auto", {"device_id": 1})
    assert decoder._transport.backend == "auto"
    assert decoder._transport.device_id == 1
    decoder._transport.cpu_frames = 3
    decoder._transport.dvpp_frames = 5
    decoder._transport.last_error = "fallback reason"
    assert decoder.cpu_frames == 3
    assert decoder.dvpp_frames == 5
    assert decoder.last_error == "fallback reason"
    decoder.close()
