import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from sensor_msgs.msg import Image, JointState

from ibrobot_msgs.action import PlaceObject
from ibrobot_msgs.msg import Detection2D, DetectionArray
from ibrobot_msgs.srv import GroundingDetect, SegmentDetections
from manipulation_execution.placement_executor_node import (
    PlacementExecutorNode,
    PlacementFlowError,
    PlacementState,
    PrimitiveFlowError,
    _load_exclusion_mask,
    _normalized_polygon_mask,
    evaluate_mask_arrays,
    evaluate_mask_containment,
)
from manipulation_execution.placement_replay import audit_placement_evidence


def _mask(array: np.ndarray) -> Image:
    message = Image()
    message.height, message.width = array.shape
    message.encoding = "mono8"
    message.step = int(array.shape[1])
    message.data = np.where(array, 255, 0).astype(np.uint8).tobytes()
    return message


def _binding(task_id: str):
    binding = PlaceObject.Goal().dispatch_binding
    binding.schema_version = 1
    binding.task_id = task_id
    binding.root_task_id = task_id
    binding.dispatch_nonce = "delegated-nonce"
    now = time.time()
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = int(now)
    binding.task_budget.started_at.nanosec = int((now - int(now)) * 1_000_000_000)
    deadline = now + 120.0
    binding.task_budget.deadline.sec = int(deadline)
    binding.task_budget.deadline.nanosec = int((deadline - int(deadline)) * 1_000_000_000)
    return binding


def _result_identity(node) -> None:
    node._executor_identity = {
        "name": "placement_pipeline",
        "contract_version": "1",
        "endpoint_kind": "ros_action",
        "endpoint_name": "/manipulation/execute_place",
        "configuration_digest": "0" * 64,
        "model_deployment_name": "",
        "model_fingerprint": "",
        "model_bundle_digest": "",
    }


def _container_and_target(*, target_inside: bool) -> tuple[Image, Image]:
    container = np.zeros((100, 120), dtype=bool)
    container[20:85, 15:105] = True
    target = np.zeros_like(container)
    if target_inside:
        target[45:65, 50:70] = True
    else:
        target[5:20, 50:70] = True
    return _mask(container), _mask(target)


def test_mask_containment_accepts_object_inside_filled_container_region():
    container, target = _container_and_target(target_inside=True)

    result = evaluate_mask_containment(
        container,
        target,
        min_target_pixels=100,
        min_inside_fraction=0.7,
        container_inset_ratio=0.05,
    )

    assert result.inside
    assert result.center_inside
    assert result.inside_fraction == 1.0


def test_mask_containment_rejects_visible_object_outside_container():
    container, target = _container_and_target(target_inside=False)

    result = evaluate_mask_containment(
        container,
        target,
        min_target_pixels=100,
        min_inside_fraction=0.7,
        container_inset_ratio=0.05,
    )

    assert not result.inside
    assert not result.center_inside
    assert result.inside_fraction == 0.0


def test_mask_containment_requires_enough_target_pixels():
    container, target = _container_and_target(target_inside=True)

    result = evaluate_mask_containment(
        container,
        target,
        min_target_pixels=1000,
        min_inside_fraction=0.7,
        container_inset_ratio=0.05,
    )

    assert not result.inside
    assert result.target_pixel_count == 400


def test_mask_array_replay_matches_ros_mask_containment():
    container, target = _container_and_target(target_inside=True)
    expected = evaluate_mask_containment(
        container,
        target,
        min_target_pixels=100,
        min_inside_fraction=0.7,
        container_inset_ratio=0.05,
    )
    actual = evaluate_mask_arrays(
        np.asarray(np.frombuffer(container.data, dtype=np.uint8).reshape((100, 120)) > 0),
        np.asarray(np.frombuffer(target.data, dtype=np.uint8).reshape((100, 120)) > 0),
        min_target_pixels=100,
        min_inside_fraction=0.7,
        container_inset_ratio=0.05,
    )
    assert actual == expected


def test_normalized_gripper_exclusion_mask_scales_to_wrist_frame():
    mask = _normalized_polygon_mask(
        (360, 640),
        [[[0.59, 0.70], [0.73, 0.70], [0.73, 1.0], [0.59, 1.0]]],
    )
    assert mask.shape == (360, 640)
    assert int(np.count_nonzero(mask)) > 8_000
    assert not mask[200, 300]
    assert mask[320, 430]


def test_target_exclusion_removes_gripper_detection_and_mask():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {"verification": {"target_exclusion": {"min_detection_overlap": 0.35}}}
    exclusion = np.zeros((10, 10), dtype=bool)
    exclusion[5:, 5:] = True
    gripper = Detection2D(bbox=[5.0, 5.0, 10.0, 10.0], mask=_mask(exclusion.copy()))
    object_mask = np.zeros((10, 10), dtype=bool)
    object_mask[1:4, 1:4] = True
    object_detection = Detection2D(bbox=[1.0, 1.0, 4.0, 4.0], mask=_mask(object_mask))

    result = node._filter_target_detections([gripper, object_detection], exclusion)

    assert result == [object_detection]


def test_external_sam_mask_is_loaded_and_resized(tmp_path):
    from PIL import Image as PILImage

    source = np.zeros((4, 6), dtype=np.uint8)
    source[1:3, 2:4] = 255
    path = tmp_path / "gripper_mask.png"
    PILImage.fromarray(source).save(path)

    result = _load_exclusion_mask(str(path), (8, 12))

    assert result.shape == (8, 12)
    assert result.dtype == bool
    assert int(np.count_nonzero(result)) == 16


def test_target_exclusion_mask_is_cached_per_frame_shape(monkeypatch):
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "verification": {
            "target_exclusion": {
                "enabled": True,
                "mask_path": "/fixed/gripper-mask.png",
                "min_detection_overlap": 0.35,
            }
        }
    }
    image = Image(height=360, width=640)
    calls = []
    expected = np.zeros((360, 640), dtype=bool)

    def load_mask(path, shape):
        calls.append((path, shape))
        return expected

    monkeypatch.setattr("manipulation_execution.placement_executor_node._load_exclusion_mask", load_mask)

    first = node._target_exclusion_mask(image)
    second = node._target_exclusion_mask(image)

    assert first is expected
    assert second is expected
    assert calls == [("/fixed/gripper-mask.png", (360, 640))]


def test_legacy_placement_record_is_explicitly_incompatible(tmp_path):
    (tmp_path / "placement_result.json").write_text("{}\n", encoding="utf-8")
    report = audit_placement_evidence(tmp_path)
    assert report["status"] == "legacy_incompatible"


def test_legacy_recovery_file_is_explicitly_incompatible(tmp_path):
    recovery = tmp_path / "placement_recovery.json"
    recovery.write_text("[]\n", encoding="utf-8")
    report = audit_placement_evidence(recovery)
    assert report["status"] == "legacy_incompatible"


def test_replay_fixture_verifies_two_consecutive_inside_samples(tmp_path):
    manifest = {
        "schema_version": 1,
        "pipeline": "placement_pipeline",
        "pipeline_version": 3,
        "task_id": "offline-place",
        "target_query": "marker",
        "container_query": "black bowl",
        "gripper": {"joint_name": "6", "open_position": 1.0, "position_tolerance": 0.05},
        "configuration": {"verification": {"required_confirmations": 2, "min_target_mask_pixels": 10}},
    }
    (tmp_path / "placement_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    container = np.zeros((20, 20), dtype=bool)
    container[2:18, 2:18] = True
    target = np.zeros_like(container)
    target[8:12, 8:12] = True
    for index in (1, 2):
        np.save(tmp_path / f"sample_{index:02d}_container_00_mask.npy", container, allow_pickle=False)
        np.save(tmp_path / f"sample_{index:02d}_target_00_mask.npy", target, allow_pickle=False)
        (tmp_path / f"sample_{index:02d}.json").write_text(
            json.dumps({"sample_index": index, "outcome": True}), encoding="utf-8"
        )
    (tmp_path / "open_gripper_joint_state.json").write_text(
        json.dumps({"name": ["6"], "position": [1.0]}), encoding="utf-8"
    )
    report = audit_placement_evidence(tmp_path)
    assert report["status"] == "verified"
    assert report["open_feedback_present"] is True
    assert report["open_feedback_verified"] is True
    assert report["vision_verified"] is True


def test_replay_selects_highest_confidence_masks(tmp_path):
    manifest = {
        "schema_version": 1,
        "pipeline": "placement_pipeline",
        "pipeline_version": 3,
        "task_id": "offline-place-multiple",
        "target_query": "red marker",
        "container_query": "black bowl",
        "gripper": {"joint_name": "6", "open_position": 1.0, "position_tolerance": 0.05},
        "configuration": {"verification": {"required_confirmations": 1, "min_target_mask_pixels": 10}},
    }
    (tmp_path / "placement_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    inside_container, inside_target = _container_and_target(target_inside=True)
    outside_container, outside_target = _container_and_target(target_inside=False)

    def mask_array(message):
        return np.frombuffer(message.data, dtype=np.uint8).reshape((message.height, message.width)) > 0

    np.save(tmp_path / "sample_01_container_00_mask.npy", mask_array(outside_container), allow_pickle=False)
    np.save(tmp_path / "sample_01_container_01_mask.npy", mask_array(inside_container), allow_pickle=False)
    np.save(tmp_path / "sample_01_target_00_mask.npy", mask_array(outside_target), allow_pickle=False)
    np.save(tmp_path / "sample_01_target_01_mask.npy", mask_array(inside_target), allow_pickle=False)
    (tmp_path / "sample_01.json").write_text(
        json.dumps(
            {
                "sample_index": 1,
                "container_detections": [{"confidence": 0.2}, {"confidence": 0.9}],
                "target_detections": [{"confidence": 0.3}, {"confidence": 0.8}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "open_gripper_joint_state.json").write_text(
        json.dumps({"name": ["6"], "position": [1.0]}), encoding="utf-8"
    )

    report = audit_placement_evidence(tmp_path)

    assert report["status"] == "verified"
    assert report["samples"][0]["outcome"] is True


def test_result_keeps_release_and_verification_independent():
    state = PlacementState()
    state.release_status = PlaceObject.Result.RELEASE_RELEASED
    state.verification_status = PlaceObject.Result.VERIFICATION_UNCERTAIN
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    _result_identity(node)

    result = PlacementExecutorNode._result(
        node,
        state,
        success=False,
        code="PLACE_VERIFICATION_UNCERTAIN",
        message="uncertain",
    )

    assert result.release_status == PlaceObject.Result.RELEASE_RELEASED
    assert result.verification_status == PlaceObject.Result.VERIFICATION_UNCERTAIN
    assert not result.place_succeeded


def test_open_feedback_must_arrive_after_open_completion():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._dispatch_binding = _binding("place-test")
    node._state_lock = threading.Lock()
    node._gripper_closed = 0.0
    node._gripper_open = 1.0
    node._gripper_tolerance = 0.05
    node._gripper_joint_name = "6"
    node._latest_joint_state = JointState(name=["6"], position=[1.0])
    node._latest_joint_receipt_monotonic = time.monotonic() - 1.0

    assert not node._gripper_open_feedback_is_fresh(newer_than_monotonic=time.monotonic())
    completion = time.monotonic()
    node._latest_joint_receipt_monotonic = completion + 0.01
    assert node._gripper_open_feedback_is_fresh(newer_than_monotonic=completion)


def test_preflight_only_waits_for_release_dependency():
    calls = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._rpc_timeout = 1.0
    node._primitive_client = type(
        "PrimitiveClient",
        (),
        {"wait_for_server": lambda _self, *, timeout_sec: calls.append(("primitive", timeout_sec)) or True},
    )()
    node._detect_client = type(
        "DetectClient",
        (),
        {"wait_for_service": lambda _self, *, timeout_sec: calls.append(("detect", timeout_sec)) or True},
    )()
    node._feedback = lambda *_args, **_kwargs: None

    node._preflight(object(), time.monotonic() + 2.0, PlacementState())

    assert calls[0][0] == "primitive"
    assert all(name != "detect" for name, _timeout in calls)


def test_joint_target_dispatches_guarded_primitive_fields_in_configured_order():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._dispatch_binding = _binding("place-test")
    sent_goals = []
    node._primitive_client = type(
        "PrimitiveClient",
        (),
        {"send_goal_async": lambda _self, goal: sent_goals.append(goal) or object()},
    )()
    handle = type("Handle", (), {"accepted": True, "get_result_async": lambda _self: object()})()
    wrapped = type("Wrapped", (), {"result": type("Result", (), {"success": True})()})()
    responses = iter([handle, wrapped])
    node._wait_future = lambda *_args, **_kwargs: next(responses)

    node._move_to_joint_positions(
        SimpleNamespace(goal_id=None),
        time.monotonic() + 60.0,
        "place-test",
        ["1", "2", "3", "4", "5"],
        [-0.047553, -0.073631, -0.840621, 1.497165, -1.570790],
        duration_sec=10.0,
        honor_cancel=False,
    )

    assert len(sent_goals) == 1
    goal = sent_goals[0]
    assert goal.dispatch_binding.task_id == "place-test"
    assert goal.dispatch_binding.dispatch_nonce == "delegated-nonce"
    assert goal.primitive_name == "move_to_joint_positions"
    assert list(goal.joint_names) == ["1", "2", "3", "4", "5"]
    assert list(goal.joint_positions) == pytest.approx([-0.047553, -0.073631, -0.840621, 1.497165, -1.570790])
    assert goal.primitive_duration_sec == pytest.approx(10.0)


def test_execute_place_moves_joint_3_to_verify_then_returns_to_release_pose():
    calls = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "motion": {
            "place_pose": "place_container",
            "place_joint_names": ["1", "2", "3", "4", "5"],
            "place_joint_positions": {
                "1": -0.047553,
                "2": -0.073631,
                "3": -0.840621,
                "4": 1.497165,
                "5": -1.570790,
            },
            "place_duration_sec": 10.0,
            "post_release": {
                "verify_joint_name": "3",
                "verify_joint_position": -0.687223,
                "verify_duration_sec": 2.0,
                "return_duration_sec": 2.0,
            },
        },
        "verification": {"post_release_wait_sec": 1.0},
    }
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._dispatch_binding = _binding("place-test")
    node._rpc_timeout = 1.0
    _result_identity(node)
    node._preflight = lambda *_args: calls.append(("preflight",))
    node._feedback = lambda _goal, _state, phase, _detail, **_kwargs: calls.append(("feedback", phase))
    node._move_to_joint_positions = lambda _goal, _deadline, _task_id, names, positions, **kwargs: calls.append(
        ("move_joints", names, positions, kwargs["duration_sec"], kwargs["honor_cancel"])
    )
    node._open_gripper = lambda *_args: calls.append(("open",)) or time.monotonic()
    node._wait_for_open_feedback = lambda *_args: True
    node._sleep_until = lambda *_args: calls.append(("settle",))
    node._verify_post_release = lambda *_args: calls.append(("verify",)) or PlaceObject.Result.VERIFICATION_SUCCESS
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123))
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            dispatch_binding=_binding("place-test"),
            target_query="marker",
            container_query="black bowl",
            timeout_sec=60.0,
        ),
        succeed=lambda: calls.append(("succeed",)),
        abort=lambda: calls.append(("abort",)),
        canceled=lambda: calls.append(("canceled",)),
    )

    result = PlacementExecutorNode._execute_place(node, goal_handle)

    assert result.success
    assert result.place_succeeded
    assert result.release_status == PlaceObject.Result.RELEASE_RELEASED
    assert result.verification_status == PlaceObject.Result.VERIFICATION_SUCCESS
    release_move = (
        "move_joints",
        ["1", "2", "3", "4", "5"],
        [-0.047553, -0.073631, -0.840621, 1.497165, -1.570790],
        10.0,
        True,
    )
    verify_move = (
        "move_joints",
        ["1", "2", "3", "4", "5"],
        [-0.047553, -0.073631, -0.687223, 1.497165, -1.570790],
        2.0,
        False,
    )
    return_move = (
        "move_joints",
        ["1", "2", "3", "4", "5"],
        [-0.047553, -0.073631, -0.840621, 1.497165, -1.570790],
        2.0,
        False,
    )
    assert calls.index(release_move) < calls.index(("open",))
    assert calls.index(("open",)) < calls.index(verify_move)
    assert calls.index(verify_move) < calls.index(("settle",))
    assert calls.index(("settle",)) < calls.index(("verify",))
    assert calls.index(("verify",)) < calls.index(return_move)
    assert calls.index(return_move) < calls.index(("succeed",))


def test_execute_place_stops_before_motion_when_shared_budget_expired():
    calls = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._dispatch_binding = _binding("place-expired")
    _result_identity(node)
    node._preflight = lambda *_args: calls.append("preflight")
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(time.time() * 1_000_000_000)))
    binding = _binding("place-expired")
    binding.task_budget.deadline.sec = binding.task_budget.started_at.sec
    binding.task_budget.deadline.nanosec = binding.task_budget.started_at.nanosec
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            dispatch_binding=binding,
            target_query="marker",
            container_query="black bowl",
            timeout_sec=60.0,
        ),
        succeed=lambda: calls.append("succeed"),
        abort=lambda: calls.append("abort"),
        canceled=lambda: calls.append("canceled"),
    )

    result = PlacementExecutorNode._execute_place(node, goal_handle)

    assert not result.success
    assert result.error_code == "TASK_TIMEOUT"
    assert result.release_status == PlaceObject.Result.RELEASE_NOT_RELEASED
    assert calls == ["abort"]


def test_primitive_cancel_cleanup_rejects_unknown_terminal_state():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._rpc_timeout = 1.0

    class DoneFuture:
        def __init__(self, value):
            self._value = value

        def done(self):
            return True

        def result(self):
            return self._value

    result = SimpleNamespace(
        error_code="CANCEL_CLEANUP_TIMEOUT",
        message="controller stop state unknown",
    )
    result_future = DoneFuture(SimpleNamespace(result=result))
    handle = SimpleNamespace(
        cancel_goal_async=lambda: DoneFuture(SimpleNamespace(goals_canceling=[object()])),
    )

    with pytest.raises(PlacementFlowError, match="controller stop state unknown") as raised:
        node._cancel_primitive_and_wait(handle, result_future, "move_to_joint_positions")

    assert raised.value.code == "CANCEL_CLEANUP_TIMEOUT"


def test_positioning_failure_does_not_open_gripper():
    calls = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "motion": {
            "place_pose": "place_container",
            "place_joint_names": ["1", "2", "3", "4", "5"],
            "place_joint_positions": {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
            "place_duration_sec": 10.0,
            "post_release": {
                "verify_joint_name": "3",
                "verify_joint_position": -0.687223,
                "verify_duration_sec": 2.0,
                "return_duration_sec": 2.0,
            },
        }
    }
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._dispatch_binding = _binding("place-test")
    node._rpc_timeout = 1.0
    _result_identity(node)
    node._preflight = lambda *_args: None
    node._feedback = lambda *_args, **_kwargs: None

    def fail_move(*_args, **_kwargs):
        calls.append("move")
        raise PlacementFlowError("PRIMITIVE_FAILED", "unknown named pose")

    node._move_to_joint_positions = fail_move
    node._open_gripper = lambda *_args: calls.append("open")
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(time.time() * 1_000_000_000)))
    node.get_logger = lambda: SimpleNamespace(exception=lambda *_args: None)
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            dispatch_binding=_binding("place-test"),
            target_query="marker",
            container_query="black bowl",
            timeout_sec=60.0,
        ),
        succeed=lambda: None,
        abort=lambda: calls.append("abort"),
        canceled=lambda: None,
    )

    result = PlacementExecutorNode._execute_place(node, goal_handle)

    assert not result.success
    assert result.error_code == "PLACE_POSITIONING_FAILED"
    assert result.release_status == PlaceObject.Result.RELEASE_NOT_RELEASED
    assert calls == ["move", "abort"]


def test_verification_failure_in_place_preserves_released_state():
    calls = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "motion": {
            "place_pose": "place_container",
            "place_joint_names": ["1", "2", "3", "4", "5"],
            "place_joint_positions": {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.0},
            "place_duration_sec": 10.0,
            "post_release": {
                "verify_joint_name": "3",
                "verify_joint_position": -0.687223,
                "verify_duration_sec": 2.0,
                "return_duration_sec": 2.0,
            },
        },
        "verification": {"post_release_wait_sec": 1.0},
    }
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._dispatch_binding = _binding("place-test")
    node._rpc_timeout = 1.0
    _result_identity(node)
    node._preflight = lambda *_args: None
    node._feedback = lambda *_args, **_kwargs: None

    node._move_to_joint_positions = lambda *_args, **_kwargs: calls.append(("move_joints",))
    node._open_gripper = lambda *_args: time.monotonic()
    node._wait_for_open_feedback = lambda *_args: True
    node._sleep_until = lambda *_args: None
    node._verify_post_release = lambda *_args: calls.append(("verify",)) or PlaceObject.Result.VERIFICATION_FAILED
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123))
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            dispatch_binding=_binding("place-test"),
            target_query="marker",
            container_query="black bowl",
            timeout_sec=60.0,
        ),
        succeed=lambda: None,
        abort=lambda: calls.append(("abort",)),
        canceled=lambda: None,
    )

    result = PlacementExecutorNode._execute_place(node, goal_handle)

    assert not result.success
    assert result.error_code == "PLACE_VERIFICATION_FAILED"
    assert result.release_status == PlaceObject.Result.RELEASE_RELEASED
    assert result.verification_status == PlaceObject.Result.VERIFICATION_FAILED
    assert ("verify",) in calls
    assert calls.count(("move_joints",)) == 3
    assert calls.index(("verify",)) < max(index for index, call in enumerate(calls) if call == ("move_joints",))


def test_verification_exception_still_returns_joint_3_to_release_position():
    moves = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "motion": {
            "place_pose": "place_container",
            "place_joint_names": ["1", "2", "3", "4", "5"],
            "place_joint_positions": {"1": 0.1, "2": 0.2, "3": -0.840621, "4": 0.4, "5": 0.5},
            "place_duration_sec": 10.0,
            "post_release": {
                "verify_joint_name": "3",
                "verify_joint_position": -0.687223,
                "verify_duration_sec": 2.0,
                "return_duration_sec": 2.0,
            },
        },
        "verification": {"post_release_wait_sec": 1.0},
    }
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._dispatch_binding = _binding("place-test")
    node._rpc_timeout = 1.0
    _result_identity(node)
    node._preflight = lambda *_args: None
    node._feedback = lambda *_args, **_kwargs: None
    node._move_to_joint_positions = lambda _goal, _deadline, _task_id, names, positions, **kwargs: moves.append(
        (list(names), list(positions), kwargs["duration_sec"])
    )
    node._open_gripper = lambda *_args: time.monotonic()
    node._wait_for_open_feedback = lambda *_args: True
    node._sleep_until = lambda *_args: None
    node._verify_post_release = lambda *_args: (_ for _ in ()).throw(
        PlacementFlowError("PLACE_VERIFICATION_UNCERTAIN", "camera unavailable")
    )
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123))
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            dispatch_binding=_binding("place-test"),
            target_query="marker",
            container_query="black bowl",
            timeout_sec=60.0,
        ),
        succeed=lambda: None,
        abort=lambda: None,
        canceled=lambda: None,
    )

    result = PlacementExecutorNode._execute_place(node, goal_handle)

    assert not result.success
    assert result.error_code == "PLACE_VERIFICATION_UNCERTAIN"
    assert len(moves) == 3
    assert moves[1][1] == pytest.approx([0.1, 0.2, -0.687223, 0.4, 0.5])
    assert moves[2][1] == pytest.approx([0.1, 0.2, -0.840621, 0.4, 0.5])


def test_return_failure_is_reported_after_successful_verification():
    calls = []
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "motion": {
            "place_pose": "place_container",
            "place_joint_names": ["1", "2", "3", "4", "5"],
            "place_joint_positions": {"1": 0.1, "2": 0.2, "3": -0.840621, "4": 0.4, "5": 0.5},
            "place_duration_sec": 10.0,
            "post_release": {
                "verify_joint_name": "3",
                "verify_joint_position": -0.687223,
                "verify_duration_sec": 2.0,
                "return_duration_sec": 2.0,
            },
        },
        "verification": {"post_release_wait_sec": 1.0},
    }
    node._goal_lock = threading.Lock()
    node._goal_active = True
    node._dispatch_binding = _binding("place-test")
    node._rpc_timeout = 1.0
    _result_identity(node)
    node._preflight = lambda *_args: None
    node._feedback = lambda *_args, **_kwargs: None

    def move(*_args, **_kwargs):
        calls.append("move")
        if len(calls) == 3:
            raise PlacementFlowError("PRIMITIVE_ARM_FAILED", "controller rejected return")

    node._move_to_joint_positions = move
    node._open_gripper = lambda *_args: time.monotonic()
    node._wait_for_open_feedback = lambda *_args: True
    node._sleep_until = lambda *_args: None
    node._verify_post_release = lambda *_args: PlaceObject.Result.VERIFICATION_SUCCESS
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123))
    goal_handle = SimpleNamespace(
        request=SimpleNamespace(
            dispatch_binding=_binding("place-test"),
            target_query="marker",
            container_query="black bowl",
            timeout_sec=60.0,
        ),
        succeed=lambda: None,
        abort=lambda: None,
        canceled=lambda: None,
    )

    result = PlacementExecutorNode._execute_place(node, goal_handle)

    assert not result.success
    assert result.place_succeeded
    assert result.verification_status == PlaceObject.Result.VERIFICATION_SUCCESS
    assert result.error_code == "PLACE_RETURN_FAILED"
    assert calls == ["move", "move", "move"]


def test_verification_requires_consecutive_confirmations():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {"verification": {"max_resamples": 3, "required_confirmations": 2}}
    node._rpc_timeout = 1.0
    node._detect_client = type("DetectClient", (), {"wait_for_service": lambda _self, *, timeout_sec: True})()
    node._segment_client = None
    outcomes = iter(
        [
            (True, "inside", (1, 1)),
            (False, "outside", (2, 2)),
            (True, "inside", (3, 3)),
            (True, "inside", (4, 4)),
        ]
    )
    node._sample_verification = lambda *_args, **_kwargs: next(outcomes)
    node._sleep_until = lambda *_args: None
    state = PlacementState()

    result = node._verify_post_release(object(), time.monotonic() + 2.0, "marker", "black bowl", state)

    assert result == PlaceObject.Result.VERIFICATION_SUCCESS
    assert len(state.diagnostic_details) == 4


def test_verification_does_not_count_the_same_image_twice():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {"verification": {"max_resamples": 2, "required_confirmations": 2}}
    node._rpc_timeout = 1.0
    node._detect_client = type("DetectClient", (), {"wait_for_service": lambda _self, *, timeout_sec: True})()
    node._segment_client = None
    outcomes = iter(
        [
            (True, "inside", (1, 1)),
            (True, "inside repeated", (1, 1)),
            (True, "inside new", (2, 2)),
        ]
    )
    node._sample_verification = lambda *_args, **_kwargs: next(outcomes)
    node._sleep_until = lambda *_args: None
    state = PlacementState()

    result = node._verify_post_release(object(), time.monotonic() + 2.0, "marker", "black bowl", state)

    assert result == PlaceObject.Result.VERIFICATION_SUCCESS
    assert any("repeated image stamps ignored" in detail for detail in state.diagnostic_details)


def test_verification_uses_runtime_container_query_without_static_config():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "verification": {
            "min_container_mask_pixels": 10,
            "min_target_mask_pixels": 10,
            "min_inside_mask_fraction": 0.7,
            "container_inset_ratio": 0.05,
        }
    }
    image = Image()
    image.header.stamp.nanosec = 1
    container, target = _container_and_target(target_inside=True)
    queries = []
    node._wait_for_rgb = lambda **_kwargs: image
    node._record_image_snapshot = lambda *_args: 0
    node._record_verification_sample = lambda **_kwargs: None

    def detect(_goal_handle, _deadline, _image, query):
        queries.append(query)
        mask = container if query == "black bowl" else target
        return [Detection2D(label=query, confidence=0.9, mask=mask)]

    node._detect = detect

    outcome, _detail, _sample_key = node._sample_verification(
        object(),
        time.monotonic() + 1.0,
        "marker",
        "black bowl",
        minimum_stamp_ns=0,
    )

    assert outcome is True
    assert queries == ["black bowl", "marker"]


def test_verification_selects_highest_confidence_container_and_target_with_multiple_candidates():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {
        "verification": {
            "min_container_mask_pixels": 10,
            "min_target_mask_pixels": 10,
            "min_inside_mask_fraction": 0.7,
            "container_inset_ratio": 0.05,
        }
    }
    image = Image()
    image.header.stamp.nanosec = 1
    inside_container, inside_target = _container_and_target(target_inside=True)
    outside_container, outside_target = _container_and_target(target_inside=False)
    queries = []
    node._wait_for_rgb = lambda **_kwargs: image
    node._record_image_snapshot = lambda *_args: 0
    node._record_verification_sample = lambda **_kwargs: None

    def detect(_goal_handle, _deadline, _image, query, **_kwargs):
        queries.append(query)
        if query == "black bowl":
            return [
                Detection2D(label=query, confidence=0.4, mask=outside_container),
                Detection2D(label=query, confidence=0.9, mask=inside_container),
            ]
        return [
            Detection2D(label=query, confidence=0.3, mask=outside_target),
            Detection2D(label=query, confidence=0.8, mask=inside_target),
        ]

    node._detect = detect

    outcome, detail, _sample_key = node._sample_verification(
        object(),
        time.monotonic() + 1.0,
        "red marker",
        "black bowl",
        minimum_stamp_ns=0,
    )

    assert outcome is True
    assert "containers=2 selected_container_confidence=0.900" in detail
    assert "targets=2 selected_target_confidence=0.800" in detail
    assert queries == ["black bowl", "red marker"]


def test_cancel_cleanup_timeout_maps_to_unknown_release_state():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._gripper_open = 1.0
    node._dispatch_binding = _binding("task")
    node._primitive_client = type(
        "PrimitiveClient",
        (),
        {"send_goal_async": lambda *_args, **_kwargs: object()},
    )()
    result = type(
        "Result",
        (),
        {"success": False, "error_code": "CANCEL_CLEANUP_TIMEOUT", "message": "cleanup unknown"},
    )()
    handle = type(
        "Handle",
        (),
        {"accepted": True, "get_result_async": lambda _self: object()},
    )()
    responses = iter([handle, type("Wrapped", (), {"result": result})()])
    node._wait_future = lambda *_args, **_kwargs: next(responses)
    goal_handle = type("Goal", (), {"goal_id": None})()

    try:
        node._open_gripper(goal_handle, time.monotonic() + 1.0, "task")
    except PrimitiveFlowError as exc:
        assert exc.code == "RELEASE_STATE_UNKNOWN"
        assert not exc.terminal_known
    else:
        raise AssertionError("expected unknown release state")


def test_accepted_open_failure_is_not_reported_as_definitely_not_released():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._gripper_open = 1.0
    node._dispatch_binding = _binding("task")
    node._primitive_client = type(
        "PrimitiveClient",
        (),
        {"send_goal_async": lambda *_args, **_kwargs: object()},
    )()
    result = type(
        "Result",
        (),
        {"success": False, "error_code": "PRIMITIVE_GRIPPER_FAILED", "message": "controller failed"},
    )()
    handle = type(
        "Handle",
        (),
        {"accepted": True, "get_result_async": lambda _self: object()},
    )()
    responses = iter([handle, type("Wrapped", (), {"result": result})()])
    node._wait_future = lambda *_args, **_kwargs: next(responses)
    goal_handle = type("Goal", (), {"goal_id": None})()

    try:
        node._open_gripper(goal_handle, time.monotonic() + 1.0, "task")
    except PrimitiveFlowError as exc:
        assert exc.code == "PRIMITIVE_GRIPPER_FAILED"
        assert not exc.terminal_known
    else:
        raise AssertionError("expected unknown release state")


def test_pc_detection_uses_masks_returned_by_grounding_service():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {"verification": {"confidence_threshold": 0.3}}
    detection = Detection2D(confidence=0.8, mask=_mask(np.ones((10, 10), dtype=bool)))
    response = GroundingDetect.Response(
        success=True,
        detections=DetectionArray(detections=[detection]),
    )
    node._detect_client = type("DetectClient", (), {"call_async": lambda _self, _request: response})()
    node._segment_client = None
    node._wait_future = lambda future, *_args, **_kwargs: future

    result = node._detect(object(), time.monotonic() + 1.0, Image(), "marker")

    assert result == [detection]


def test_target_detection_request_applies_static_gripper_exclusion_mask():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {"verification": {"confidence_threshold": 0.3}}
    detection = Detection2D(confidence=0.8, mask=_mask(np.ones((2, 2), dtype=bool)))
    response = GroundingDetect.Response(
        success=True,
        detections=DetectionArray(detections=[detection]),
    )
    requests = []
    node._detect_client = type(
        "DetectClient",
        (),
        {"call_async": lambda _self, request: requests.append(request) or response},
    )()
    node._segment_client = None
    node._wait_future = lambda future, *_args, **_kwargs: future
    image = Image(height=2, width=2, encoding="rgb8", step=6, data=bytes([10, 20, 30] * 4))
    exclusion = np.asarray([[False, True], [False, False]], dtype=bool)

    result = node._detect(object(), time.monotonic() + 1.0, image, "red marker", exclusion_mask=exclusion)

    assert result == [detection]
    assert len(requests) == 1
    masked = np.frombuffer(requests[0].image.data, dtype=np.uint8).reshape((2, 2, 3))
    assert masked[0, 0].tolist() == [10, 20, 30]
    assert masked[0, 1].tolist() == [127, 127, 127]


def test_310p_detection_segments_raw_boxes_before_verification():
    node = PlacementExecutorNode.__new__(PlacementExecutorNode)
    node._config = {"verification": {"confidence_threshold": 0.3}}
    raw = Detection2D(confidence=0.8)
    segmented = Detection2D(confidence=0.8, mask=_mask(np.ones((10, 10), dtype=bool)))
    detect_response = GroundingDetect.Response(
        success=True,
        detections=DetectionArray(detections=[raw]),
    )
    segment_response = SegmentDetections.Response(
        success=True,
        detections=DetectionArray(detections=[segmented]),
    )
    node._detect_client = type("DetectClient", (), {"call_async": lambda _self, _request: detect_response})()
    requests = []
    node._segment_client = type(
        "SegmentClient",
        (),
        {"call_async": lambda _self, request: requests.append(request) or segment_response},
    )()
    node._wait_future = lambda future, *_args, **_kwargs: future

    result = node._detect(object(), time.monotonic() + 1.0, Image(), "marker")

    assert result == [segmented]
    assert requests[0].detections.detections == [raw]
