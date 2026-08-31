import json
import math
import threading
import time
from types import SimpleNamespace

import pytest

import robot_teleop.calibrate_glove as calibrate_glove
from robot_teleop.analyze_glove_capture import analyze_capture
from robot_teleop.analyze_glove_capture import main as analyze_main
from robot_teleop.devices.aero_compact_retarget import (
    THUMB_ROOT_SWEEP_LOWER_QUANTILE,
    THUMB_ROOT_SWEEP_UPPER_QUANTILE,
    AeroFingerModelConfig,
    AeroThumbModelConfig,
    _directional_normalize,
    _finger_contractions,
    _finger_shape_targets,
    _fit_endpoint,
    aero_compact_to_normalized,
    build_aero_compact_calibration,
)
from robot_teleop.devices.aero_hand_retarget import (
    AeroHandRetargeter,
    TaskSpaceRetargetConfig,
    extract_hand_shape_metrics,
    fit_task_space_thresholds,
    fit_thumb_quaternion_thresholds,
    shape_target_from_metrics,
    validate_fitted_task_space,
)
from robot_teleop.devices.glove_calibration import (
    calibration_document,
    load_calibration,
    load_raw_capture,
    raw_capture_document,
    write_calibration_atomic,
    write_raw_capture_atomic,
)
from robot_teleop.devices.mhandpro_sdk import (
    CONNECTED_BOTH_GLOVES,
    CONNECTED_LEFT_GLOVE,
    CONNECTED_NONE,
    CONNECTED_RIGHT_GLOVE,
    CS_SUCCEEDED,
    GloveMocapData,
    GloveMocapDataWithVirtual,
    connection_satisfies_policy,
    mocap_quaternions_to_list,
    mocap_sensor_states_to_list,
    mocap_vectors_to_list,
    mocap_virtual_positions_to_list,
)
from robot_teleop.devices.mhandpro_source import (
    GloveFrame,
    RealMHandProSource,
    ReplayGloveSource,
    _glove_frame_from_worker,
    replay_pose,
)
from robot_teleop.devices.mhandpro_worker_client import MHandProWorkerClient, connection_state_after_break
from robot_teleop.devices.mocap_retarget import (
    FEATURE_SCHEMA_AERO_COMPACT,
    FEATURE_SCHEMA_AERO_COMPACT_V1,
    FEATURE_SCHEMA_LEGACY,
    FEATURE_SCHEMA_SDK_VIRTUAL,
    build_calibration,
    build_sdk_skeleton_sweep_calibration,
    build_sweep_calibration,
    extract_features,
    extract_sdk_skeleton_features,
    extract_thumb_kinematics,
    positions_to_normalized,
    sdk_skeleton_to_normalized,
)
from robot_teleop.devices.retarget_utils import percentile
from robot_teleop.mhandpro_source_node import MHandProSourceNode

JOINT_NAMES = ["thumb_abd", "thumb_opp", "thumb_mcp", "index", "middle", "ring", "pinky"]


def _calibration(side="right"):
    return build_calibration(
        {pose: [replay_pose(pose, side)] for pose in ("open", "fist", "thumb_abd", "thumb_opp")},
        side,
    )


def _write_calibration(tmp_path, *, side="right", persistence=False):
    endpoints = _calibration(side)
    document = calibration_document(
        side,
        endpoints["low"],
        endpoints["high"],
        sdk_version="test",
        persistence_verified=persistence,
    )
    return write_calibration_atomic(tmp_path / "calibration.json", document)


def _planar_thumb_adduction_pose():
    return replay_pose("thumb_abd")


def _combined_thumb_cmc_pose():
    positions = replay_pose("open")
    root = positions[1]
    adduction = 1.15
    opposition = 0.65
    direction = [
        math.sin(adduction) * math.cos(opposition),
        math.cos(adduction) * math.cos(opposition),
        math.sin(opposition),
    ]
    positions[2] = [value + 0.42 * delta for value, delta in zip(root, direction, strict=True)]
    positions[3] = [value + 0.32 * delta for value, delta in zip(positions[2], direction, strict=True)]
    return positions


def _thumb_quaternions(angle=0.0):
    quaternions = [[1.0, 0.0, 0.0, 0.0] for _ in range(20)]
    quaternions[3] = [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]
    return quaternions


def _virtual_fingertips(positions):
    tips = []
    for previous, terminal in ((2, 3), (6, 7), (10, 11), (14, 15), (18, 19)):
        direction = [positions[terminal][axis] - positions[previous][axis] for axis in range(3)]
        tips.append([positions[terminal][axis] + 0.75 * direction[axis] for axis in range(3)])
    return tips


def _sdk_frame(pose, sequence=0):
    positions = replay_pose(pose)
    virtual_positions = _virtual_fingertips(positions)
    if pose == "fist":
        distal = [positions[3][axis] - positions[2][axis] for axis in range(3)]
        distal_length = math.sqrt(sum(value * value for value in distal))
        distal = [value / distal_length for value in distal]
        curled = [distal[0], -distal[2], distal[1]]
        virtual_positions[0] = [positions[3][axis] + 0.03 * curled[axis] for axis in range(3)]
    return GloveFrame(
        positions,
        sequence,
        time.monotonic(),
        _thumb_quaternions(),
        virtual_positions,
        [0] * 20,
    )


def _write_sdk_calibration(tmp_path, *, persistence=False):
    open_frames = [_sdk_frame("open", index) for index in range(20)]
    sweep_frames = [
        _sdk_frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    endpoints = build_sdk_skeleton_sweep_calibration(open_frames, sweep_frames, "right")
    document = calibration_document(
        "right",
        endpoints["low"],
        endpoints["high"],
        sdk_version="test",
        persistence_verified=persistence,
        feature_schema=FEATURE_SCHEMA_SDK_VIRTUAL,
    )
    return write_calibration_atomic(tmp_path / "sdk_calibration.json", document), endpoints


def _write_aero_compact_calibration(tmp_path, *, persistence=False):
    open_frames = [_sdk_frame("open", index) for index in range(20)]
    sweep_frames = [
        _sdk_frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    calibration = build_aero_compact_calibration(open_frames, sweep_frames, "right")
    document = calibration_document(
        "right",
        calibration["low"],
        calibration["high"],
        sdk_version="test",
        persistence_verified=persistence,
        feature_schema=FEATURE_SCHEMA_AERO_COMPACT,
        thumb_endpoints=calibration["thumb_endpoints"],
        finger_endpoints=calibration["finger_endpoints"],
    )
    return write_calibration_atomic(tmp_path / "aero_compact_calibration.json", document), calibration


def _root_pitch_only_frame(sequence=1):
    frame = _sdk_frame("open", sequence)
    positions = [list(point) for point in frame.positions]
    open_thumb = extract_thumb_kinematics(positions, frame.virtual_positions, "right")
    root_pitch = open_thumb.root_pitch + 0.5
    direction = [
        math.sin(open_thumb.root_yaw) * math.cos(root_pitch),
        math.cos(open_thumb.root_yaw) * math.cos(root_pitch),
        math.sin(root_pitch),
    ]
    root = positions[1]
    proximal_length = math.sqrt(sum((positions[2][axis] - root[axis]) ** 2 for axis in range(3)))
    distal_length = math.sqrt(sum((positions[3][axis] - positions[2][axis]) ** 2 for axis in range(3)))
    positions[2] = [root[axis] + proximal_length * direction[axis] for axis in range(3)]
    positions[3] = [positions[2][axis] + distal_length * direction[axis] for axis in range(3)]
    return GloveFrame(
        positions,
        sequence,
        time.monotonic(),
        frame.quaternions,
        _virtual_fingertips(positions),
        frame.sensor_states,
    )


def _root_yaw_only_frame(sequence=1):
    frame = _sdk_frame("open", sequence)
    positions = [list(point) for point in frame.positions]
    open_thumb = extract_thumb_kinematics(positions, frame.virtual_positions, "right")
    root_yaw = open_thumb.root_yaw - 0.3
    direction = [
        math.sin(root_yaw) * math.cos(open_thumb.root_pitch),
        math.cos(root_yaw) * math.cos(open_thumb.root_pitch),
        math.sin(open_thumb.root_pitch),
    ]
    root = positions[1]
    proximal_length = math.sqrt(sum((positions[2][axis] - root[axis]) ** 2 for axis in range(3)))
    distal_length = math.sqrt(sum((positions[3][axis] - positions[2][axis]) ** 2 for axis in range(3)))
    positions[2] = [root[axis] + proximal_length * direction[axis] for axis in range(3)]
    positions[3] = [positions[2][axis] + distal_length * direction[axis] for axis in range(3)]
    return GloveFrame(
        positions,
        sequence,
        time.monotonic(),
        frame.quaternions,
        _virtual_fingertips(positions),
        frame.sensor_states,
    )


def _unsmoothed_retargeter():
    return AeroHandRetargeter(TaskSpaceRetargetConfig(smoothness_weight=0.0, max_normalized_step=1.0, neutral_frames=3))


def _prime_open(retargeter):
    for _ in range(retargeter.config.neutral_frames):
        retargeter.retarget(replay_pose("open"))


def test_task_space_retarget_maps_planar_thumb_motion_to_abduction_only():
    positions = _planar_thumb_adduction_pose()
    config = TaskSpaceRetargetConfig(smoothness_weight=0.0, max_normalized_step=1.0, neutral_frames=3)
    neutral = extract_hand_shape_metrics(replay_pose("open"))
    target = shape_target_from_metrics(extract_hand_shape_metrics(positions), neutral, config)
    retargeter = AeroHandRetargeter(config)
    _prime_open(retargeter)
    normalized = retargeter.retarget(positions)

    assert target.cmc_abduction > 0.9
    assert normalized[0] > 0.9
    assert normalized[1] == pytest.approx(0.0, abs=1e-6)
    assert normalized[2:] == pytest.approx([0.0] * 5, abs=1e-6)


def test_task_space_retarget_fits_combined_thumb_cmc_motion_without_curling_fingers():
    config = TaskSpaceRetargetConfig(smoothness_weight=0.0, max_normalized_step=1.0, neutral_frames=3)
    retargeter = AeroHandRetargeter(config)
    _prime_open(retargeter)

    normalized = retargeter.retarget(_combined_thumb_cmc_pose())

    assert normalized[0] > 0.9
    assert normalized[1] > 0.9
    assert normalized[2:] == pytest.approx([0.0] * 5, abs=1e-6)


def test_task_space_quaternion_thumb_curve_is_invariant_to_cmc_motion():
    config = TaskSpaceRetargetConfig(
        smoothness_weight=0.0,
        max_normalized_step=1.0,
        neutral_frames=2,
        thumb_quaternion_deadband_rad=0.02,
        thumb_quaternion_range_rad=0.2,
    )
    retargeter = AeroHandRetargeter(config)
    retargeter.retarget(replay_pose("open"), _thumb_quaternions())
    retargeter.retarget(replay_pose("open"), _thumb_quaternions())

    straight_across = retargeter.retarget(_combined_thumb_cmc_pose(), _thumb_quaternions())
    curled = retargeter.retarget(_combined_thumb_cmc_pose(), _thumb_quaternions(0.2))

    assert straight_across[0] > 0.9 and straight_across[1] > 0.9
    assert straight_across[2] == pytest.approx(0.0, abs=1e-6)
    assert curled[2] > 0.9
    assert curled[3:] == pytest.approx([0.0] * 4, abs=1e-6)


def test_task_space_retarget_preserves_open_and_fist_shapes():
    retargeter = _unsmoothed_retargeter()
    _prime_open(retargeter)

    open_targets = retargeter.retarget(replay_pose("open"))
    retargeter.reset()
    _prime_open(retargeter)
    fist_targets = retargeter.retarget(replay_pose("fist"))

    assert open_targets == pytest.approx([0.0] * 7, abs=1e-6)
    assert min(fist_targets[3:]) > 0.8


def test_task_space_retarget_limits_per_frame_command_change():
    config = TaskSpaceRetargetConfig(max_normalized_step=0.1, neutral_frames=3)
    retargeter = AeroHandRetargeter(config)
    _prime_open(retargeter)
    initial = retargeter.retarget(replay_pose("open"))
    updated = retargeter.retarget(_combined_thumb_cmc_pose())

    assert max(abs(after - before) for before, after in zip(initial, updated, strict=True)) <= 0.100001


def test_task_space_thresholds_are_fitted_from_free_sweep():
    open_pose = replay_pose("open")
    sweep = [
        *[replay_pose("open") for _ in range(20)],
        *[replay_pose("fist") for _ in range(20)],
        *[replay_pose("thumb_abd") for _ in range(20)],
        *[replay_pose("thumb_opp") for _ in range(20)],
    ]

    thresholds = fit_task_space_thresholds([open_pose] * 20, sweep)
    config = TaskSpaceRetargetConfig.from_dict(thresholds)
    retargeter = AeroHandRetargeter(
        TaskSpaceRetargetConfig.from_dict(
            {
                **thresholds,
                "smoothness_weight": 0.0,
                "max_normalized_step": 1.0,
                "neutral_frames": 3,
            }
        )
    )
    _prime_open(retargeter)
    normalized = retargeter.retarget(_combined_thumb_cmc_pose())

    assert config.thumb_adduction_range_rad > 0.0
    assert min(config.finger_curve_range) > 0.0
    assert normalized[0] > 0.9 and normalized[1] > 0.9


def test_retarget_maps_calibration_poses_to_normalized_endpoints():
    calibration = _calibration()

    assert positions_to_normalized(replay_pose("open"), calibration, "right") == pytest.approx([0.0] * 7)
    assert positions_to_normalized(replay_pose("fist"), calibration, "right")[2:] == pytest.approx([1.0] * 5)
    assert positions_to_normalized(replay_pose("thumb_abd"), calibration, "right")[0] == pytest.approx(1.0)
    assert positions_to_normalized(replay_pose("thumb_opp"), calibration, "right")[1] == pytest.approx(1.0)


def test_retarget_rejects_overlapping_and_nonfinite_nodes():
    calibration = _calibration()
    with pytest.raises(ValueError, match="overlap"):
        positions_to_normalized([[0.0, 0.0, 0.0]] * 20, calibration, "right")

    invalid = replay_pose("open")
    invalid[3][0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        positions_to_normalized(invalid, calibration, "right")


def test_calibration_rejects_noise_sized_feature_spans():
    poses = {pose: [replay_pose(pose)] for pose in ("open", "fist", "thumb_abd", "thumb_opp")}
    poses["thumb_abd"] = [replay_pose("open")]

    with pytest.raises(ValueError, match="thumb_abd.*too close"):
        build_calibration(poses, "right")


def test_sweep_calibration_uses_open_reference_and_robust_extremes():
    open_pose = replay_pose("open")
    sweep = [replay_pose(pose) for pose in ("open", "fist", "thumb_abd", "thumb_opp") for _ in range(20)]

    calibration = build_sweep_calibration([open_pose] * 20, sweep, "right")

    assert positions_to_normalized(open_pose, calibration, "right") == pytest.approx([0.0] * 7)
    assert positions_to_normalized(replay_pose("fist"), calibration, "right")[2:] == pytest.approx([1.0] * 5)
    assert positions_to_normalized(replay_pose("thumb_abd"), calibration, "right")[0] == pytest.approx(1.0)
    assert positions_to_normalized(replay_pose("thumb_opp"), calibration, "right")[1] == pytest.approx(1.0)


def test_sdk_skeleton_sweep_calibration_maps_open_and_virtual_tip_curl():
    open_frames = [_sdk_frame("open", index) for index in range(20)]
    sweep_frames = [
        _sdk_frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]

    endpoints = build_sdk_skeleton_sweep_calibration(open_frames, sweep_frames, "right")
    calibration = {**endpoints, "feature_schema": FEATURE_SCHEMA_SDK_VIRTUAL}

    assert sdk_skeleton_to_normalized(
        open_frames[0].positions,
        open_frames[0].virtual_positions,
        calibration,
        "right",
    ) == pytest.approx([0.0] * 7)
    fist = _sdk_frame("fist")
    assert sdk_skeleton_to_normalized(
        fist.positions,
        fist.virtual_positions,
        calibration,
        "right",
    )[2:] == pytest.approx([1.0] * 5)


def test_aero_compact_maps_root_pitch_to_cmc_abduction():
    open_frame = _sdk_frame("open")
    pitch_frame = _root_pitch_only_frame()
    calibration = build_aero_compact_calibration(
        [open_frame] * 20,
        [
            *[_sdk_frame(pose) for pose in ("open", "fist", "thumb_abd", "thumb_opp") for _ in range(20)],
            *[pitch_frame] * 20,
        ],
        "right",
    )
    targets = aero_compact_to_normalized(
        pitch_frame.positions,
        pitch_frame.virtual_positions,
        calibration,
        "right",
        AeroThumbModelConfig.from_dict(
            {
                "root_output_scales": [1.0, 1.0],
                "root_deadband_rad": 0.0,
                "tendon_deadband_rad": 0.0,
                "tendon_output_scale": 1.0,
            }
        ),
    )

    assert targets[0] > 0.4
    assert targets[1] == pytest.approx(0.0, abs=1e-6)
    assert targets[2] == pytest.approx(0.0, abs=1e-6)


def test_aero_compact_maps_root_yaw_to_cmc_flexion():
    open_frame = _sdk_frame("open")
    yaw_frame = _root_yaw_only_frame()
    calibration = build_aero_compact_calibration(
        [open_frame] * 20,
        [
            *[_sdk_frame(pose) for pose in ("open", "fist", "thumb_abd", "thumb_opp") for _ in range(20)],
            *[yaw_frame] * 20,
        ],
        "right",
    )
    open_thumb = extract_thumb_kinematics(open_frame.positions, open_frame.virtual_positions, "right")
    yaw_thumb = extract_thumb_kinematics(yaw_frame.positions, yaw_frame.virtual_positions, "right")
    calibration["thumb_endpoints"]["root_yaw_rad"] = {
        "neutral": open_thumb.root_yaw,
        "active": yaw_thumb.root_yaw,
    }
    targets = aero_compact_to_normalized(
        yaw_frame.positions,
        yaw_frame.virtual_positions,
        calibration,
        "right",
        AeroThumbModelConfig.from_dict(
            {
                "root_output_scales": [1.0, 1.0],
                "root_deadband_rad": 0.0,
                "tendon_deadband_rad": 0.0,
                "tendon_output_scale": 1.0,
            }
        ),
    )

    assert targets[0] == pytest.approx(0.0, abs=1e-6)
    assert targets[1] > 0.4
    assert targets[2] == pytest.approx(0.0, abs=1e-6)


def test_aero_compact_rejects_cross_axis_root_matrix():
    with pytest.raises(ValueError, match="Unknown Aero thumb model settings: root_matrix"):
        AeroThumbModelConfig.from_dict({"root_matrix": [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]})


def test_aero_compact_rejects_nonphysical_mcp_ip_weights():
    with pytest.raises(ValueError, match="SDK tendon coefficient ratio"):
        AeroThumbModelConfig.from_dict({"mcp_ip_weights": [0.5, 0.5]})


def test_aero_compact_directional_endpoint_handles_thumb_index_open_close():
    endpoint = {"neutral": 1.013, "active": 0.594}

    assert _directional_normalize(1.013, endpoint, 0.0, "root_yaw_rad") == pytest.approx(0.0)
    assert _directional_normalize(0.8035, endpoint, 0.0, "root_yaw_rad") == pytest.approx(0.5)
    assert _directional_normalize(0.594, endpoint, 0.0, "root_yaw_rad") == pytest.approx(1.0)
    assert _directional_normalize(1.05, endpoint, 0.0, "root_yaw_rad") == pytest.approx(0.0)


def test_aero_compact_fits_thumb_endpoints_from_comfortable_sweep_quantiles():
    values = [float(value) for value in range(100)]
    values.extend((-1000.0, 1000.0))

    endpoint = _fit_endpoint(
        values,
        0.0,
        "root_yaw_rad",
        lower_quantile=THUMB_ROOT_SWEEP_LOWER_QUANTILE,
        upper_quantile=THUMB_ROOT_SWEEP_UPPER_QUANTILE,
    )

    assert endpoint["active"] == pytest.approx(percentile(values, 0.90))
    assert endpoint["active"] < 100.0


def test_aero_finger_model_maps_comfortable_shapes_without_extreme_calibration():
    config = AeroFingerModelConfig()
    open_frame = _sdk_frame("open")
    fist_frame = _sdk_frame("fist")

    open_targets = _finger_shape_targets(open_frame.positions, config)
    fist_targets = _finger_shape_targets(fist_frame.positions, config)

    assert open_targets == pytest.approx([0.0] * 4)
    assert fist_targets == pytest.approx([1.0] * 4)


def test_aero_finger_model_ignores_mcp_pose_when_pip_dip_shape_is_unchanged():
    config = AeroFingerModelConfig()
    open_frame = _sdk_frame("open")
    positions = [list(point) for point in open_frame.positions]
    root = positions[4]
    angle = math.radians(40.0)
    for node in (5, 6, 7):
        relative = [positions[node][axis] - root[axis] for axis in range(3)]
        positions[node] = [
            root[0] + relative[0],
            root[1] + relative[1] * math.cos(angle) - relative[2] * math.sin(angle),
            root[2] + relative[1] * math.sin(angle) + relative[2] * math.cos(angle),
        ]
    targets = _finger_shape_targets(positions, config)

    assert targets == pytest.approx([0.0] * 4)


def test_aero_finger_model_keeps_each_tendon_independent():
    config = AeroFingerModelConfig()
    open_frame = _sdk_frame("open")
    fist_frame = _sdk_frame("fist")
    positions = [list(point) for point in open_frame.positions]
    positions[4:8] = [list(point) for point in fist_frame.positions[4:8]]
    targets = _finger_shape_targets(positions, config)

    assert targets == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_aero_finger_model_uses_named_endpoints_not_legacy_low_high_arrays():
    open_frames = [_sdk_frame("open", index) for index in range(20)]
    sweep_frames = [
        _sdk_frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    calibration = build_aero_compact_calibration(open_frames, sweep_frames, "right")
    exaggerated = {
        **calibration,
        "low": [*calibration["low"][:3], -10.0, -20.0, -30.0, -40.0],
        "high": [*calibration["high"][:3], 10.0, 20.0, 30.0, 40.0],
    }
    frame = _sdk_frame("fist")
    thumb_model = AeroThumbModelConfig()
    finger_model = AeroFingerModelConfig()

    baseline = aero_compact_to_normalized(
        frame.positions,
        frame.virtual_positions,
        calibration,
        "right",
        thumb_model,
        finger_model,
    )
    changed = aero_compact_to_normalized(
        frame.positions,
        frame.virtual_positions,
        exaggerated,
        "right",
        thumb_model,
        finger_model,
    )

    assert changed == pytest.approx(baseline)


def test_aero_finger_model_trims_eight_percent_from_calibrated_active_end():
    config = AeroFingerModelConfig(open_threshold_rad=0.0, active_trim_fraction=0.08)
    fist = _sdk_frame("fist")
    contractions = _finger_contractions(fist.positions, config)
    endpoints = {
        name: {"neutral": 0.0, "active": contraction / 0.92}
        for name, contraction in zip(("index", "middle", "ring", "pinky"), contractions, strict=True)
    }

    targets = _finger_shape_targets(fist.positions, config, endpoints)

    assert targets == pytest.approx([1.0] * 4)


def test_aero_compact_directional_endpoint_trims_extreme_human_poses():
    endpoint = {"neutral": 1.0, "active": 0.0}

    assert _directional_normalize(
        0.8,
        endpoint,
        0.0,
        "root_yaw_rad",
        neutral_trim=0.2,
        active_trim=0.1,
    ) == pytest.approx(0.0)
    assert _directional_normalize(
        0.45,
        endpoint,
        0.0,
        "root_yaw_rad",
        neutral_trim=0.2,
        active_trim=0.1,
    ) == pytest.approx(0.5)
    assert _directional_normalize(
        0.1,
        endpoint,
        0.0,
        "root_yaw_rad",
        neutral_trim=0.2,
        active_trim=0.1,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "model",
    (
        {"root_neutral_trims": [-0.1, 0.0]},
        {"root_active_trims": [1.0, 0.0]},
        {"root_neutral_trims": [0.6, 0.0], "root_active_trims": [0.4, 0.0]},
    ),
)
def test_aero_compact_rejects_invalid_extreme_pose_trims(model):
    with pytest.raises(ValueError, match="root trims|non-empty input range"):
        AeroThumbModelConfig.from_dict(model)


def test_calibration_is_atomic_and_enforces_side_and_persistence(tmp_path):
    path = _write_calibration(tmp_path, persistence=False)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["channel_order"] == JOINT_NAMES
    assert load_calibration(path, "right", require_persistence=False) == {
        "low": payload["low"],
        "high": payload["high"],
        "feature_schema": FEATURE_SCHEMA_LEGACY,
    }
    with pytest.raises(ValueError, match="side"):
        load_calibration(path, "left", require_persistence=False)
    with pytest.raises(ValueError, match="not been verified"):
        load_calibration(path, "right", require_persistence=True)
    assert list(tmp_path.glob("*.tmp")) == []


def test_calibration_document_preserves_sweep_audit_metadata():
    endpoints = _calibration()
    acquisition = {
        "method": "guided_sweep",
        "open_reference_duration_s": 0.8,
        "sweep_duration_s": 15.0,
        "robust_quantiles": [0.02, 0.98],
        "reconnect_feature_deltas_rad": [0.001] * 7,
    }
    task_space = {
        "thumb_adduction_range_rad": 0.5,
        "finger_curve_range": [0.1, 0.2, 0.3, 0.4],
    }
    document = calibration_document(
        "right",
        endpoints["low"],
        endpoints["high"],
        sdk_version="test",
        persistence_verified=True,
        acquisition=acquisition,
        task_space=task_space,
    )

    assert document["acquisition"] == acquisition
    assert document["task_space"] == task_space


def test_real_source_rejects_nonpositive_startup_timeout():
    with pytest.raises(ValueError, match="startup_timeout"):
        RealMHandProSource("unused.so", "right", startup_timeout=0.0)


def test_sdk_quaternions_preserve_documented_wxyz_order():
    data = GloveMocapData()
    expected = [0.5, -0.1, 0.2, -0.3]
    for axis, value in enumerate(expected):
        data.quaternion[2][axis] = value

    assert mocap_quaternions_to_list(data)[2] == pytest.approx(expected)


def test_sdk_virtual_fingertips_and_sensor_states_preserve_vendor_order():
    data = GloveMocapDataWithVirtual()
    for index in range(5):
        for axis in range(3):
            data.positionVirtual[index][axis] = 10.0 * index + axis
    for index in range(20):
        data.sensorState[index] = index % 5

    assert mocap_virtual_positions_to_list(data)[3] == pytest.approx([30.0, 31.0, 32.0])
    assert mocap_sensor_states_to_list(data) == [index % 5 for index in range(20)]


def test_sdk_motion_vectors_and_worker_metadata_are_preserved():
    data = GloveMocapDataWithVirtual()
    for index in range(20):
        for axis in range(3):
            data.gyr[index][axis] = 100.0 * index + axis
    assert mocap_vectors_to_list(data, "gyr")[3] == pytest.approx([300.0, 301.0, 302.0])

    worker_frame = {
        "positions": [[0.0, 0.0, 0.0]] * 20,
        "sequence": 9,
        "timestamp": 12.5,
        "quaternions": [[1.0, 0.0, 0.0, 0.0]] * 20,
        "virtual_positions": [[0.0, 0.0, 0.0]] * 5,
        "sensor_states": list(range(20)),
        "sdk_frame_index": 77,
        "device_power": 0.82,
        "frequency": 120,
        "gyroscope": [[1.0, 2.0, 3.0]] * 20,
        "accelerations": [[4.0, 5.0, 6.0]] * 20,
        "velocities": [[7.0, 8.0, 9.0]] * 20,
    }
    frame = _glove_frame_from_worker(worker_frame, "left")

    assert frame.side == "left"
    assert frame.sdk_frame_index == 77
    assert frame.device_power == pytest.approx(0.82)
    assert frame.frequency == 120
    assert frame.gyroscope[0] == [1.0, 2.0, 3.0]


def test_worker_client_caches_left_and_right_frames_independently():
    client = MHandProWorkerClient("unused.so", "both")
    client._latest_frames = {
        "left": {"side": "left", "sequence": 1, "positions": [[1.0, 0.0, 0.0]]},
        "right": {"side": "right", "sequence": 2, "positions": [[2.0, 0.0, 0.0]]},
    }

    left = client.latest_frame("left")
    right = client.latest_frame("right")

    assert left["sequence"] == 1
    assert right["sequence"] == 2
    left["positions"][0][0] = 99.0
    assert client.latest_frame("left")["positions"][0][0] == 1.0


def test_worker_client_tracks_remaining_side_after_dual_glove_break():
    assert connection_state_after_break(CONNECTED_BOTH_GLOVES, CONNECTED_RIGHT_GLOVE) == CONNECTED_LEFT_GLOVE
    assert connection_state_after_break(CONNECTED_BOTH_GLOVES, CONNECTED_LEFT_GLOVE) == CONNECTED_RIGHT_GLOVE
    assert connection_state_after_break(CONNECTED_RIGHT_GLOVE, CONNECTED_RIGHT_GLOVE) == CONNECTED_NONE
    assert connection_state_after_break(CONNECTED_LEFT_GLOVE, CONNECTED_LEFT_GLOVE) == CONNECTED_NONE
    assert connection_state_after_break(CONNECTED_BOTH_GLOVES, CONNECTED_BOTH_GLOVES) == CONNECTED_NONE


def test_allow_available_accepts_one_connected_glove_but_require_all_does_not():
    sides = ("left", "right")

    assert connection_satisfies_policy(CONNECTED_RIGHT_GLOVE, sides, "allow_available") is True
    assert connection_satisfies_policy(CONNECTED_RIGHT_GLOVE, sides, "require_all") is False
    assert connection_satisfies_policy(CONNECTED_BOTH_GLOVES, sides, "require_all") is True


def test_allow_available_keeps_healthy_side_without_forcing_reconnect():
    source = SimpleNamespace(is_side_connected=lambda side: side == "left")
    node = SimpleNamespace(
        auto_reconnect=True,
        _shutdown=threading.Event(),
        sides=("left", "right"),
        failure_policy="allow_available",
        _reconnect_attempts=3,
        reconnect_initial_delay=1.0,
        _next_reconnect_at=0.0,
        reconnect_max_attempts=0,
        reconnect_max_delay=10.0,
        _last_reconnect_log=0.0,
        _reconnect_lock=threading.Lock(),
        _reconnect_thread=None,
        _source_for=lambda _side: source,
        get_logger=lambda: SimpleNamespace(error=lambda *_args: None),
    )

    MHandProSourceNode._maybe_reconnect(node)

    assert node._reconnect_thread is None
    assert node._reconnect_attempts == 0


def test_reconnect_attempt_runs_outside_ros_timer_callback():
    entered = threading.Event()
    release = threading.Event()
    source = SimpleNamespace(is_side_connected=lambda _side: False)

    def reconnect_once(_source):
        entered.set()
        release.wait(timeout=1.0)

    node = SimpleNamespace(
        auto_reconnect=True,
        _shutdown=threading.Event(),
        sides=("right",),
        failure_policy="require_all",
        _reconnect_attempts=0,
        reconnect_initial_delay=1.0,
        _next_reconnect_at=0.0,
        reconnect_max_attempts=0,
        reconnect_max_delay=10.0,
        _last_reconnect_log=0.0,
        _reconnect_lock=threading.Lock(),
        _reconnect_thread=None,
        _reconnecting=False,
        _ready=True,
        _ready_sides={"right"},
        source_name="test",
        _source_for=lambda _side: source,
        _reconnect_once=reconnect_once,
        get_logger=lambda: SimpleNamespace(error=lambda *_args: None),
    )

    started = time.monotonic()
    MHandProSourceNode._maybe_reconnect(node)
    elapsed = time.monotonic() - started

    assert entered.wait(timeout=0.2)
    assert elapsed < 0.1
    release.set()
    node._reconnect_thread.join(timeout=1.0)


def test_source_node_p_pose_helper_unlocks_only_after_quality_check():
    source = SimpleNamespace(
        calibrate_p_pose=lambda timeout: (CS_SUCCEEDED, timeout),
        is_side_connected=lambda side: side == "right",
    )
    checks = []
    node = SimpleNamespace(
        sides=("right",),
        calibration_timeout=30.0,
        failure_policy="require_all",
        _ready=True,
        _ready_sides={"right"},
        _source_for=lambda _side: source,
        _validate_post_calibration_frames=lambda sides: checks.append(sides) or "right flexion=0.100",
    )

    quality = MHandProSourceNode._run_p_pose_calibration(node)

    assert quality == "right flexion=0.100"
    assert checks == [{"right"}]
    assert node._ready is True
    assert node._ready_sides == {"right"}


def test_sdk_virtual_thumb_tip_decouples_straight_opposition_from_distal_curl():
    positions = _combined_thumb_cmc_pose()
    distal_direction = [0.15, 0.92, 0.36]
    length = math.sqrt(sum(value * value for value in distal_direction))
    distal_direction = [value / length for value in distal_direction]
    positions[3] = [positions[2][axis] + 0.32 * distal_direction[axis] for axis in range(3)]
    straight_virtual = _virtual_fingertips(positions)
    straight_virtual[0] = [positions[3][axis] + 0.03 * distal_direction[axis] for axis in range(3)]
    curled_virtual = [list(point) for point in straight_virtual]
    curled_direction = [distal_direction[0], -distal_direction[2], distal_direction[1]]
    curled_virtual[0] = [positions[3][axis] + 0.03 * curled_direction[axis] for axis in range(3)]

    legacy_curve = extract_features(positions, "right")[2]
    straight_curve = extract_sdk_skeleton_features(positions, straight_virtual, "right")[2]
    curled_curve = extract_sdk_skeleton_features(positions, curled_virtual, "right")[2]

    assert legacy_curve > 0.5
    assert straight_curve == pytest.approx(0.0, abs=1e-6)
    assert curled_curve > 0.5


def test_raw_capture_preserves_phase_and_quaternion_metadata(tmp_path):
    positions = replay_pose("open")
    quaternions = [[1.0, 0.0, 0.0, 0.0] for _ in range(20)]
    virtual_positions = _virtual_fingertips(positions)
    sensor_states = [index % 5 for index in range(20)]
    frame = GloveFrame(positions, 7, 12.5, quaternions, virtual_positions, sensor_states)
    document = raw_capture_document("right", "test", {"open": [frame], "sweep": [frame]})

    assert document["schema_version"] == 1
    assert document["quaternion_order"] == "wxyz"
    assert [item["phase"] for item in document["frames"]] == ["open", "sweep"]
    assert document["frames"][0]["quaternions_wxyz"][0] == [1.0, 0.0, 0.0, 0.0]
    assert document["frames"][0]["virtual_positions"] == virtual_positions
    assert document["frames"][0]["sensor_states"] == sensor_states

    path = write_raw_capture_atomic(tmp_path / "capture.json", "right", "test", {"open": [frame], "sweep": [frame]})
    assert json.loads(path.read_text(encoding="utf-8"))["frames"][0]["sequence"] == 7
    loaded = load_raw_capture(path, "right")
    assert loaded["open"][0].quaternions == quaternions
    assert loaded["open"][0].virtual_positions == virtual_positions
    assert loaded["open"][0].sensor_states == sensor_states


def test_raw_capture_can_be_analyzed_offline(tmp_path):
    quaternions = _thumb_quaternions()
    open_frames = [GloveFrame(replay_pose("open"), index, float(index), quaternions) for index in range(20)]
    sweep_frames = [
        GloveFrame(
            replay_pose(pose),
            20 + index,
            float(20 + index),
            _thumb_quaternions(0.25 if pose == "fist" else 0.0),
        )
        for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    path = write_raw_capture_atomic(
        tmp_path / "capture.json",
        "right",
        "test",
        {"open": open_frames, "sweep": sweep_frames},
    )

    result = analyze_capture(path, "right")

    assert result["frame_counts"] == {"open": 20, "sweep": 80}
    assert result["quaternion_frames"] == 100
    assert result["fitted_task_space"]["thumb_adduction_range_rad"] > 0.0
    assert result["fitted_task_space"]["thumb_quaternion_range_rad"] == pytest.approx(0.25)


def test_complete_sdk_capture_fits_reusable_virtual_tip_endpoints(tmp_path):
    open_frames = [_sdk_frame("open", index) for index in range(20)]
    sweep_frames = [
        _sdk_frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    path = write_raw_capture_atomic(
        tmp_path / "complete_capture.json",
        "right",
        "test",
        {"open": open_frames, "sweep": sweep_frames},
    )

    result = analyze_capture(path, "right")

    assert result["fitted_sdk_skeleton"]["low"][2] == pytest.approx(0.0)
    assert result["fitted_sdk_skeleton"]["high"][2] > math.radians(80.0)
    assert result["fitted_aero_compact"]["feature_schema"] == FEATURE_SCHEMA_AERO_COMPACT
    open_thumb = extract_thumb_kinematics(
        open_frames[0].positions,
        open_frames[0].virtual_positions,
        "right",
    )
    assert result["fitted_aero_compact"]["thumb_endpoints"]["mcp_flex_rad"]["neutral"] == pytest.approx(
        open_thumb.mcp_flex
    )
    assert "fitted_task_space" not in result
    assert "quaternion range" in result["task_space_error"]


def test_offline_analysis_upgrades_aero_compact_v1_without_new_capture(tmp_path, capsys):
    open_frames = [_sdk_frame("open", index) for index in range(20)]
    sweep_frames = [
        _sdk_frame(pose, 20 + index) for index, pose in enumerate(("open", "fist", "thumb_abd", "thumb_opp") * 20)
    ]
    capture_path = write_raw_capture_atomic(
        tmp_path / "complete_capture.json",
        "right",
        "test",
        {"open": open_frames, "sweep": sweep_frames},
    )
    v1 = build_aero_compact_calibration(open_frames, sweep_frames, "right")
    neutral = {
        name: v1["thumb_endpoints"][name]["neutral"]
        for name in ("root_yaw_rad", "root_pitch_rad", "mcp_flex_rad", "ip_flex_rad")
    }
    calibration_path = write_calibration_atomic(
        tmp_path / "calibration.json",
        calibration_document(
            "right",
            v1["low"],
            v1["high"],
            sdk_version="test",
            persistence_verified=True,
            feature_schema=FEATURE_SCHEMA_AERO_COMPACT_V1,
            thumb_neutral=neutral,
        ),
    )

    assert (
        analyze_main(
            [
                "--input",
                str(capture_path),
                "--side",
                "right",
                "--update-calibration",
                str(calibration_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    upgraded = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert upgraded["feature_schema"] == FEATURE_SCHEMA_AERO_COMPACT
    assert "thumb_endpoints" in upgraded
    assert "thumb_neutral" not in upgraded


def test_quaternion_threshold_fit_rejects_noise_only_capture():
    open_frames = [GloveFrame(replay_pose("open"), index, float(index), _thumb_quaternions()) for index in range(20)]
    thresholds = {
        **fit_task_space_thresholds(
            [replay_pose("open")] * 20,
            [replay_pose(pose) for pose in ("open", "fist", "thumb_abd", "thumb_opp") for _ in range(20)],
        ),
        **fit_thumb_quaternion_thresholds(open_frames, open_frames),
    }

    with pytest.raises(ValueError, match="quaternion (range|coverage)"):
        validate_fitted_task_space(thresholds)


def test_full_sweep_capture_uses_one_confirmation_and_detects_open_frames(monkeypatch):
    confirmations = []
    captures = []
    args = SimpleNamespace(
        single_pass=True,
        duration=0.8,
        minimum_frames=20,
        timeout=4.0,
        sweep_duration=15.0,
    )

    monkeypatch.setattr("builtins.input", lambda prompt: confirmations.append(prompt))

    def fake_collect(_source, side, **kwargs):
        captures.append((side, kwargs))
        return ["sweep"] * 300

    monkeypatch.setattr(calibrate_glove, "collect_glove_frames", fake_collect)
    monkeypatch.setattr(calibrate_glove, "detect_open_frames", lambda frames, side, minimum_frames: frames[:25])

    open_frames, sweep_frames = calibrate_glove.capture_open_and_sweep(object(), "right", args)

    assert len(confirmations) == 1
    assert open_frames == ["sweep"] * 25
    assert sweep_frames == ["sweep"] * 300
    assert [item[0] for item in captures] == ["right"]


def test_dual_calibration_cli_accepts_both_and_resolves_side_outputs(tmp_path):
    args = calibrate_glove.parse_args(
        [
            "--side",
            "both",
            "--lib-path",
            "mhandpro.so",
            "--output",
            str(tmp_path),
        ]
    )

    assert args.side == "both"
    assert calibrate_glove._side_paths(args.side, args.output) == {
        "left": tmp_path / "aero_hand_left_calibrate.json",
        "right": tmp_path / "aero_hand_right_calibrate.json",
    }


def test_dual_capture_uses_one_window_and_collects_both_sides(monkeypatch):
    confirmations = []
    captures = []
    args = SimpleNamespace(minimum_frames=20, timeout=4.0, sweep_duration=15.0)

    monkeypatch.setattr("builtins.input", lambda prompt: confirmations.append(prompt))

    def fake_collect(_source, sides, **kwargs):
        captures.append((tuple(sides), kwargs))
        return {side: [side] * 300 for side in sides}

    monkeypatch.setattr(calibrate_glove, "collect_glove_frames_multi", fake_collect)
    monkeypatch.setattr(calibrate_glove, "detect_open_frames", lambda frames, side, minimum_frames: frames[:25])

    result = calibrate_glove.capture_open_and_sweep_multi(object(), args)

    assert len(confirmations) == 1
    assert captures[0][0] == ("left", "right")
    assert set(result) == {"left", "right"}
    assert result["left"] == (["left"] * 25, ["left"] * 300)
    assert result["right"] == (["right"] * 25, ["right"] * 300)


def test_runtime_calibration_mode_does_not_require_library_path():
    args = calibrate_glove.parse_args(["--runtime-service", "/hand_sources/mhandpro/calibrate_p_pose"])

    assert args.runtime_service == "/hand_sources/mhandpro/calibrate_p_pose"
    assert args.lib_path is None


def test_persistence_features_include_virtual_thumb_tip():
    first = _sdk_frame("open", 1)
    second = _sdk_frame("open", 2)
    second.virtual_positions[0] = list(second.positions[3])
    second.virtual_positions[0][2] += 0.03

    first_features = calibrate_glove._persistence_features([first], "right")
    second_features = calibrate_glove._persistence_features([second], "right")

    assert first_features[:10] == pytest.approx(second_features[:10])
    assert abs(first_features[10] - second_features[10]) > math.radians(5.0)


def test_replay_source_advances_frames():
    source = ReplayGloveSource("right", rate_hz=100.0, segment_seconds=0.05)
    source.connect()
    try:
        first = source.latest_frame("right")
        time.sleep(0.04)
        second = source.latest_frame("right")
        assert first is not None and second is not None
        assert second.sequence > first.sequence
        assert second.positions != first.positions
        assert first.quaternions == [[1.0, 0.0, 0.0, 0.0]] * 20
    finally:
        source.disconnect()


def test_isolated_worker_reports_library_load_failure(tmp_path):
    client = MHandProWorkerClient(
        str(tmp_path / "missing.so"),
        "right",
        startup_timeout=2.0,
        runtime_dir=tmp_path / "runtime",
    )

    with pytest.raises(ConnectionError, match="library not found"):
        client.connect()

    assert not client.is_connected
    assert (tmp_path / "runtime" / "python3").is_file()
