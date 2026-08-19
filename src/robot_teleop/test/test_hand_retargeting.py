import math
import time
from types import SimpleNamespace

import pytest

from robot_teleop.devices.aero_compact_retarget import build_aero_compact_calibration
from robot_teleop.devices.glove_calibration import calibration_document, write_calibration_atomic
from robot_teleop.devices.hand_retarget import HandRetargetDevice
from robot_teleop.devices.mhandpro_source import GloveFrame, _replay_virtual_fingertips, replay_pose
from robot_teleop.hand_retargeting import AeroCompactRetargeter, HandObservation, SynergyMatrixRetargeter
from robot_teleop.hand_state import HUMAN_HAND_SCHEMA, detect_open_frames, extract_human_hand_geometry


def _frame(pose: str, sequence: int = 0) -> GloveFrame:
    positions = replay_pose(pose)
    return GloveFrame(
        positions,
        sequence,
        time.monotonic(),
        virtual_positions=_replay_virtual_fingertips(positions, thumb_curled=pose == "fist"),
    )


def _observation(frame: GloveFrame) -> HandObservation:
    geometry = extract_human_hand_geometry(frame.positions, frame.virtual_positions, "right")
    return HandObservation(
        source="mhandpro",
        schema=HUMAN_HAND_SCHEMA,
        side="right",
        sequence=frame.sequence,
        timestamp_ns=0,
        valid=True,
        status="ready",
        positions=frame.positions,
        quaternions_wxyz=[],
        virtual_positions=frame.virtual_positions,
        features=geometry.as_dict(),
    )


def _joint_limits(*names: str) -> dict[str, dict[str, float]]:
    return {name: {"min": 0.0, "max": 1.0} for name in names}


def _hand_state_message(frame: GloveFrame):
    geometry = extract_human_hand_geometry(frame.positions, frame.virtual_positions, "right")

    def point(values):
        return SimpleNamespace(x=values[0], y=values[1], z=values[2])

    orientation = SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)
    return SimpleNamespace(
        source="mhandpro",
        schema=HUMAN_HAND_SCHEMA,
        side="right",
        sequence=frame.sequence,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=2)),
        valid=True,
        status="ready",
        landmarks=[point(values) for values in frame.positions],
        orientations=[orientation for _ in range(20)],
        virtual_tips=[point(values) for values in frame.virtual_positions],
        feature_names=list(geometry.feature_names),
        features=list(geometry.features),
    )


def test_human_hand_geometry_is_scale_and_translation_invariant():
    frame = _frame("thumb_opp")
    transformed_positions = [
        [10.0 * value + offset for value, offset in zip(point, (4.0, -2.0, 7.0), strict=True)]
        for point in frame.positions
    ]
    transformed_tips = [
        [10.0 * value + offset for value, offset in zip(point, (4.0, -2.0, 7.0), strict=True)]
        for point in frame.virtual_positions
    ]

    original = extract_human_hand_geometry(frame.positions, frame.virtual_positions, "right")
    transformed = extract_human_hand_geometry(transformed_positions, transformed_tips, "right")

    assert transformed.feature_names == original.feature_names
    assert transformed.features == pytest.approx(original.features)
    assert transformed.openness_score == pytest.approx(original.openness_score)


def test_open_frames_are_detected_from_an_unlabeled_full_range_sweep():
    frames = [_frame(pose, sequence) for sequence, pose in enumerate(("fist", "thumb_abd", "open", "thumb_opp") * 30)]

    detected = detect_open_frames(frames, "right", minimum_frames=20)

    scores = [
        extract_human_hand_geometry(frame.positions, frame.virtual_positions, "right").openness_score
        for frame in detected
    ]
    assert len(detected) >= 20
    assert max(scores) < 0.01


def test_synergy_matrix_supports_arbitrary_three_channel_target():
    retargeter = SynergyMatrixRetargeter(
        {
            "input_features": ["index_mcp_flex", "thumb_root_yaw"],
            "joint_names": ["thumb", "index", "middle"],
            "matrix": [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
            "offsets": [0.1, 0.0, 0.0],
            "joint_limits": {
                "thumb": {"min": 0.0, "max": 0.5},
                "index": {"min": 0.0, "max": 1.0},
                "middle": {"min": 0.0, "max": 1.0},
            },
        }
    )
    observation = _observation(_frame("fist"))

    targets = retargeter.retarget(observation)

    assert list(targets) == ["thumb", "index", "middle"]
    assert targets["thumb"] == pytest.approx(0.5)
    assert targets["index"] == pytest.approx(min(1.0, observation.features["index_mcp_flex"]))
    assert targets["middle"] == pytest.approx(observation.features["thumb_root_yaw"])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("landmarks", [], "landmarks.*20"),
        ("orientations", [], "orientations.*20"),
        ("virtual_tips", [], "virtual_tips.*5"),
        ("feature_names", ["one"], "equal lengths"),
    ),
)
def test_hand_observation_rejects_malformed_message_at_decode_boundary(field, value, error):
    message = _hand_state_message(_frame("open"))
    setattr(message, field, value)

    with pytest.raises(ValueError, match=error):
        HandObservation.from_message(message)


def test_hand_observation_rejects_duplicate_or_nonfinite_features():
    duplicate = _hand_state_message(_frame("open"))
    duplicate.feature_names[1] = duplicate.feature_names[0]
    with pytest.raises(ValueError, match="feature_names must be unique"):
        HandObservation.from_message(duplicate)

    nonfinite = _hand_state_message(_frame("open"))
    nonfinite.features[0] = math.nan
    with pytest.raises(ValueError, match="must be finite"):
        HandObservation.from_message(nonfinite)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("source", "", "source must be non-empty"),
        ("schema", "legacy_hand_v0", "schema must be"),
        ("side", "center", "side must be"),
    ),
)
def test_hand_observation_rejects_wrong_contract_identity(field, value, error):
    message = _hand_state_message(_frame("open"))
    setattr(message, field, value)

    with pytest.raises(ValueError, match=error):
        HandObservation.from_message(message)


@pytest.mark.parametrize(
    "limits",
    (
        {},
        {"thumb": {"min": 0.0, "max": 1.0}},
        {
            "thumb": {"min": 0.0, "max": 1.0},
            "index": {"min": 1.0, "max": 1.0},
            "middle": {"min": 0.0, "max": 1.0},
        },
    ),
)
def test_synergy_matrix_requires_valid_limits_for_every_output(limits):
    with pytest.raises(ValueError, match="joint limits"):
        SynergyMatrixRetargeter(
            {
                "input_features": ["index_mcp_flex"],
                "joint_names": ["thumb", "index", "middle"],
                "matrix": [[1.0], [1.0], [1.0]],
                "joint_limits": limits,
            }
        )


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_synergy_matrix_rejects_nonfinite_configuration(invalid):
    with pytest.raises(ValueError, match="weights and offsets must be finite"):
        SynergyMatrixRetargeter(
            {
                "input_features": ["index_mcp_flex"],
                "joint_names": ["index"],
                "matrix": [[invalid]],
                "joint_limits": _joint_limits("index"),
            }
        )


def test_hand_retarget_device_reports_base_connection_state():
    class FakeNode:
        def create_subscription(self, *_args):
            return object()

        def get_logger(self):
            return None

    device = HandRetargetDevice(
        {
            "side": "right",
            "source_topic": "/hands/right/state",
            "stale_timeout": 0.2,
            "joint_names": ["thumb", "index", "middle"],
            "joint_limits": _joint_limits("thumb", "index", "middle"),
            "retargeter": {
                "type": "synergy_matrix",
                "input_features": ["index_mcp_flex"],
                "matrix": [[1.0], [1.0], [1.0]],
            },
        },
        node=FakeNode(),
    )

    assert not device.is_connected


def test_device_factory_uses_source_topic_retarget_adapter():
    from robot_teleop.device_factory import device_factory

    class FakeNode:
        def create_subscription(self, *_args):
            return object()

        def get_logger(self):
            return None

    device = device_factory(
        {
            "type": "hand_retarget",
            "side": "right",
            "source_topic": "/hands/right/state",
            "joint_names": ["thumb", "index", "middle"],
            "joint_limits": _joint_limits("thumb", "index", "middle"),
            "retargeter": {
                "type": "synergy_matrix",
                "input_features": ["index_mcp_flex"],
                "matrix": [[1.0], [1.0], [1.0]],
            },
        },
        node=FakeNode(),
    )

    assert isinstance(device, HandRetargetDevice)
    assert device.connect()
    assert device.is_connected
    device.disconnect()
    assert not device.is_connected


def test_hand_retarget_device_fails_closed_for_stale_and_invalid_state():
    class FakeLogger:
        def warning(self, _message):
            return None

    class FakeNode:
        def create_subscription(self, *_args):
            return object()

        def get_logger(self):
            return FakeLogger()

    device = HandRetargetDevice(
        {
            "side": "right",
            "source_topic": "/hands/right/state",
            "stale_timeout": 0.05,
            "joint_names": ["index"],
            "joint_limits": _joint_limits("index"),
            "retargeter": {
                "type": "synergy_matrix",
                "input_features": ["index_mcp_flex"],
                "matrix": [[1.0]],
            },
        },
        node=FakeNode(),
    )
    assert device.connect()
    valid = _observation(_frame("fist"))
    device._latest = valid
    device._received_at = time.monotonic() - 1.0
    assert device.get_joint_targets() == {}

    device._latest = HandObservation(
        source=valid.source,
        schema=valid.schema,
        side=valid.side,
        sequence=valid.sequence,
        timestamp_ns=valid.timestamp_ns,
        valid=False,
        status="waiting_p_pose",
        positions=valid.positions,
        quaternions_wxyz=valid.quaternions_wxyz,
        virtual_positions=valid.virtual_positions,
        features=valid.features,
    )
    device._received_at = time.monotonic()
    assert device.get_joint_targets() == {}

    device._latest = valid
    device._received_at = time.monotonic()
    assert device.get_joint_targets()["index"] > 0.0


def test_hand_retarget_device_rejects_wrong_source_and_side():
    warnings = []

    class FakeLogger:
        def warning(self, message):
            warnings.append(message)

    class FakeNode:
        def create_subscription(self, *_args):
            return object()

        def get_logger(self):
            return FakeLogger()

    device = HandRetargetDevice(
        {
            "source_name": "mhandpro",
            "side": "right",
            "source_topic": "/hands/right/state",
            "joint_names": ["index"],
            "joint_limits": _joint_limits("index"),
            "retargeter": {
                "type": "synergy_matrix",
                "input_features": ["index_mcp_flex"],
                "matrix": [[1.0]],
            },
        },
        node=FakeNode(),
    )
    device.connect()

    wrong_side = _hand_state_message(_frame("fist"))
    wrong_side.side = "left"
    device._state_callback(wrong_side)
    assert device.get_joint_targets() == {}

    wrong_source = _hand_state_message(_frame("fist"))
    wrong_source.source = "other_glove"
    device._state_callback(wrong_source)
    assert device.get_joint_targets() == {}
    assert any("Ignoring left hand state" in message for message in warnings)
    assert any("expected 'mhandpro'" in message for message in warnings)


def test_hand_retarget_estop_requires_a_new_post_release_frame():
    class FakeLogger:
        def warning(self, _message):
            return None

    class FakeNode:
        def create_subscription(self, *_args):
            return object()

        def get_logger(self):
            return FakeLogger()

    device = HandRetargetDevice(
        {
            "side": "right",
            "source_topic": "/hands/right/state",
            "joint_names": ["index"],
            "joint_limits": _joint_limits("index"),
            "retargeter": {
                "type": "synergy_matrix",
                "input_features": ["index_mcp_flex"],
                "matrix": [[1.0]],
            },
        },
        node=FakeNode(),
    )
    assert device.connect()
    observation = _observation(_frame("fist"))
    device._latest = observation
    device._received_at = time.monotonic()
    assert device.get_joint_targets()["index"] > 0.0

    device.emergency_stop()
    device._state_callback(SimpleNamespace())
    assert device.get_joint_targets() == {}

    device.emergency_stop_released()
    assert device.get_joint_targets() == {}
    device._latest = observation
    device._received_at = time.monotonic()
    assert device.get_joint_targets()["index"] > 0.0


def test_aero_plugin_preserves_verified_seven_channel_mapping(tmp_path):
    open_frames = [_frame("open", index) for index in range(20)]
    sweep_frames = [
        _frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    fitted = build_aero_compact_calibration(open_frames, sweep_frames, "right")
    calibration = write_calibration_atomic(
        tmp_path / "aero.json",
        calibration_document(
            "right",
            fitted["low"],
            fitted["high"],
            sdk_version="test",
            persistence_verified=False,
            feature_schema=fitted["feature_schema"],
            thumb_endpoints=fitted["thumb_endpoints"],
            finger_endpoints=fitted["finger_endpoints"],
        ),
    )
    joints = ["thumb_abd", "thumb_flex", "thumb_tendon", "index", "middle", "ring", "pinky"]
    limits = {name: {"min": 0.0, "max": math.pi / 2.0} for name in joints}
    for name in joints[:3]:
        limits[name]["min"] = 0.2
    retargeter = AeroCompactRetargeter(
        {
            "side": "right",
            "joint_names": joints,
            "joint_limits": limits,
            "calib_file": str(calibration),
        }
    )

    targets = retargeter.retarget(_observation(_frame("fist", 101)))

    assert list(targets) == joints
    assert targets["index"] == pytest.approx(math.pi / 2.0)
    assert targets["middle"] == pytest.approx(math.pi / 2.0)
    assert all(targets[name] >= 0.2 for name in joints[:3])
    assert all(
        math.isfinite(value) and limits[name]["min"] <= value <= math.pi / 2.0 for name, value in targets.items()
    )
