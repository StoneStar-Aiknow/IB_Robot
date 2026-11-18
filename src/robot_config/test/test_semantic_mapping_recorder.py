import json
from types import SimpleNamespace

import pytest

from robot_config import semantic_mapping_recorder as recorder


def _arguments(tmp_path):
    return [
        "--session-id",
        "fixture",
        "--profile",
        "lekiwi_semantic_capture",
        "--robot-config",
        str(tmp_path / "robot.yaml"),
        "--session-root",
        str(tmp_path / "session"),
        "--state-file",
        str(tmp_path / "current.json"),
        "--mount-file",
        str(tmp_path / "mount.yaml"),
        "--camera-info-topic",
        "/camera/front/camera_info",
    ]


@pytest.mark.parametrize("message_type", [None, "livox_ros_driver2/msg/CustomMsg", "sensor_msgs/msg/PointCloud2"])
def test_recorder_persists_selected_lidar_type(tmp_path, monkeypatch, message_type):
    monkeypatch.setattr(recorder, "pin_calibration_sources", lambda *_args: {})
    monkeypatch.setattr(recorder.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        recorder.subprocess, "Popen", lambda *_args, **_kwargs: SimpleNamespace(pid=123, wait=lambda: 7)
    )
    arguments = _arguments(tmp_path)
    if message_type is not None:
        arguments.extend(["--lidar-message-type", message_type])
    assert recorder.main([*arguments, "--", "fixture-recorder"]) == 7
    state = json.loads((tmp_path / "current.json").read_text())
    assert state["lidar_message_type"] == (message_type or "livox_ros_driver2/msg/CustomMsg")


def test_recorder_rejects_unsupported_lidar_type_before_recording(tmp_path):
    with pytest.raises(SystemExit) as error:
        recorder.main([*_arguments(tmp_path), "--lidar-message-type", "sensor_msgs/msg/Image", "--", "recorder"])
    assert error.value.code == 2
    assert not (tmp_path / "current.json").exists()
