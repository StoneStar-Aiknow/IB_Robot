"""Explicit remote configuration keeps the original disabled/local paths intact."""

from pathlib import Path

import pytest
import yaml
from test_gesture_remote_runner import Transport
from test_hand_perception_gesture import Session, run_frames
from test_hand_perception_gesture import env as hand_environment

from manipulation_execution.hand_perception import remote_transport
from manipulation_execution.preset_motions.config import gesture_settings, launch_parameters


@pytest.fixture
def node_env(monkeypatch, tmp_path):
    yield from hand_environment.__wrapped__(monkeypatch, tmp_path)


def test_remote_mode_is_independent_of_local_om_and_mixed_backend(node_env, monkeypatch):
    transport = Transport()
    transport.available = False
    monkeypatch.setattr(remote_transport, "ModelServiceTransport", lambda *_: transport)
    node = node_env.make(
        gesture_mode="remote", gesture_model_path="", backend="inference_service", gesture_allow_mixed_backend=False
    )
    assert node._gesture is not None and node._gesture.is_remote
    run_frames(node, 8, wait_gesture=False)
    assert len(node.pub.msgs) == 8 and all(m.hands_in_frame == 2 for m in node.pub.msgs)
    assert not Session.instances and not transport.calls
    assert node._gesture is not None and not node._gesture.off


def test_explicit_off_never_constructs_a_local_or_remote_model(node_env):
    node = node_env.make(gesture_mode="off", gesture_enabled=True, gesture_model_path=node_env.model)
    assert node._gesture is None
    run_frames(node, 3)
    assert len(node.pub.msgs) == 3 and not Session.instances


def test_bad_remote_configuration_only_disables_gestures(node_env):
    node = node_env.make(gesture_mode="remote", gesture_remote_timeout_s=-1.0)
    assert node._gesture is None
    run_frames(node, 3)
    assert len(node.pub.msgs) == 3


def test_unexpected_remote_handoff_failure_leaves_hands_and_recovery_alive(node_env, monkeypatch):
    transport = Transport()
    transport.available = False
    monkeypatch.setattr(remote_transport, "ModelServiceTransport", lambda *_: transport)
    node = node_env.make(gesture_mode="remote")
    runner = node._gesture

    def bad_post(*_args, **_kwargs):
        raise RuntimeError("optional handoff failed")

    monkeypatch.setattr(runner, "post", bad_post)
    run_frames(node, 3, wait_gesture=False)
    assert len(node.pub.msgs) == 3
    assert node._gesture is runner and runner.stats["handoff_error"] == 3


def test_remote_source_config_adds_the_fence_only_when_explicitly_selected():
    stream = {"frame_topic": "/pear", "actor_state_topic": "/actor"}
    base = {"enabled": True, "motion_enabled": False, "gesture": {"enabled": True}}
    legacy = launch_parameters({"preset_motions": base}, stream)
    remote = launch_parameters({"preset_motions": base | {"gesture": base["gesture"] | {"source": "remote"}}}, stream)
    assert "gesture_remote_source" not in legacy
    assert remote["gesture_remote_source"] is True
    assert remote["gesture_source_topic"] == "/hri/hand/gesture_source_state"
    assert gesture_settings(base["gesture"])[0] is True


def test_bad_source_does_not_stop_the_original_observer():
    warnings = []
    result = launch_parameters(
        {"preset_motions": {"enabled": True, "gesture": {"enabled": True, "source": "remote", "source_topic": "bad"}}},
        {"frame_topic": "/pear", "actor_state_topic": "/actor"},
        warn=warnings.append,
    )
    assert result is not None and not result.get("gesture_enabled", False)
    assert warnings


def test_shipped_configuration_keeps_remote_features_off():
    path = Path(__file__).resolve().parents[2] / "robot_config/config/robots/aimdk_x2_voice_mirror_all.yaml"
    config = yaml.safe_load(path.read_text())["robot"]
    params = config["motion_imitation_stream"]["mock_playback"]["hands"]["perception"]["parameters"]
    assert params["gesture_mode"] == "" and params["gesture_enabled"] is False
    assert config["preset_motions"]["gesture"]["enabled"] is False
    assert config["preset_motions"]["gesture"]["source"] == "local"


@pytest.mark.parametrize("source", [None, True, "unknown"])
def test_source_selector_rejects_ambiguous_values(source):
    with pytest.raises(ValueError):
        gesture_settings({"source": source})
