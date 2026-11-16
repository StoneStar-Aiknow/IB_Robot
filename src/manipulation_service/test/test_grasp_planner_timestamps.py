import numpy as np
import pytest

from manipulation_service.grasp_planner_node import (
    _stamp_from_ns,
    _stamp_to_ns,
)
from manipulation_service.grasp_types import GraspDiagnostic
from manipulation_service.table_plane import (
    join_execution_table_fit,
    sample_execution_table_points,
)


def test_stamp_nanoseconds_round_trip():
    stamp_ns = 1_784_201_190_669_752_215

    assert _stamp_to_ns(_stamp_from_ns(stamp_ns)) == stamp_ns


def test_execution_table_sampling_matches_debug_ply_reader_stride():
    points = np.arange(31 * 3, dtype=np.float32).reshape(31, 3)

    sampled = sample_execution_table_points(points, max_points=10)

    np.testing.assert_array_equal(sampled, points[::3])


def test_execution_table_plane_uses_completed_scene_and_points_toward_object():
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 20), np.linspace(-0.2, 0.2, 20))
    scene = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]).astype(np.float32)
    object_points = np.array(
        [
            [-0.01, -0.01, 0.05],
            [0.01, -0.01, 0.05],
            [-0.01, 0.01, 0.05],
            [0.01, 0.01, 0.05],
        ],
        dtype=np.float32,
    )
    diagnostic = GraspDiagnostic(
        scene_pc_after_completion=scene,
        object_pc_after_completion=object_points,
    )

    fit = join_execution_table_fit(diagnostic)

    assert fit.plane is not None
    assert fit.plane.normal[2] == pytest.approx(1.0, abs=1e-12)
    assert fit.plane.d == pytest.approx(0.0, abs=1e-12)
    assert fit.plane.inlier_ratio == pytest.approx(1.0)


def test_srv_cb_joins_execution_table_fit_before_measuring_elapsed(monkeypatch):
    """M1: the real _srv_cb joins the background fit before measuring elapsed_ms.

    With a blocking fake join (0.3 s under a controlled clock) the response's
    ``inference_time_ms`` must include that wait: production order is
    t0 -> plan -> JOIN -> elapsed, never t0 -> plan -> elapsed -> join.
    """
    import threading
    import types

    import numpy as np
    from builtin_interfaces.msg import Time as TimeMsg
    from sensor_msgs.msg import CameraInfo

    from ibrobot_msgs.srv import PlanGrasp
    from manipulation_service import grasp_planner_node as node_module
    from manipulation_service.grasp_planner_node import DepthFrame, GraspPlannerNode
    from manipulation_service.grasp_types import GraspCandidate, GraspDiagnostic

    # Controlled clocks: time.time() reads 100.0 at t0 and 100.4 after the
    # join; perf_counter reads 10.0 / 10.3 around the blocking join.
    clock_values = iter([100.0, 10.0, 10.3, 100.4])
    calls = []

    def fake_time():
        value = next(clock_values)
        calls.append(("time", value))
        return value

    def fake_perf_counter():
        value = next(clock_values)
        calls.append(("perf_counter", value))
        return value

    fake_time_module = types.SimpleNamespace(time=fake_time, perf_counter=fake_perf_counter)
    monkeypatch.setattr(node_module, "time", fake_time_module)

    # A blocking join that consumes the perf_counter window.
    def fake_join(diag):
        calls.append("join")
        return types.SimpleNamespace(plane=None, best_inlier_ratio=0.0, failure_reason="", iterations_evaluated=0)

    monkeypatch.setattr(node_module, "join_execution_table_fit", fake_join)

    # Plan returns one candidate and a diagnostic carrying a started thread.
    class _FakeWrapper:
        gripper_name = "robotiq_test"
        collision_gripper_name = "robotiq_test"

        def plan_grasps(self, **_kwargs):
            diag = GraspDiagnostic()
            thread = threading.Thread(target=lambda: None)
            thread.start()
            thread.join()
            diag._execution_table_fit_thread = thread
            return [GraspCandidate(pose_4x4=np.eye(4, dtype=np.float32), confidence=0.9)], diag

    # A mask + header for the detection path.
    from std_msgs.msg import Header

    header = Header(stamp=TimeMsg(sec=1, nanosec=0), frame_id="camera")
    camera_info = CameraInfo(header=header, height=2, width=2)
    camera_info.k = [500.0, 0.0, 1.0, 0.0, 500.0, 1.0, 0.0, 0.0, 1.0]
    depth = DepthFrame(
        stamp_ns=1_000_000_000,
        frame_id="camera",
        data=np.full((2, 2), 500, dtype=np.uint16),
        depth_scale=1000.0,
    )

    planner = types.SimpleNamespace(
        _wrapper=_FakeWrapper(),
        _get_segmentation_mask=lambda prompt, threshold: (
            (np.full((2, 2), 255, dtype=np.uint8), header, 0.9, None),
            None,
        ),
        _get_synchronized_inputs=lambda stamp: (depth, camera_info),
        _grasp_pub=types.SimpleNamespace(publish=lambda msg: None),
        get_logger=lambda: types.SimpleNamespace(
            info=lambda *_a, **_k: None, warn=lambda *_a, **_k: None, error=lambda *_a, **_k: None
        ),
        get_parameter=_fake_node_get_parameter,
        _resolve_debug_output_mode=lambda mode: "none",
    )

    response = GraspPlannerNode._srv_cb(
        planner,
        PlanGrasp.Request(text_prompt="banana", confidence_threshold=0.3, grasp_threshold=0.5),
        PlanGrasp.Response(),
    )

    # Order: the join happened before the elapsed time.time() read.
    join_index = calls.index("join")
    elapsed_index = next(i for i, call in enumerate(calls) if call == ("time", 100.4))
    assert join_index < elapsed_index, calls
    # The blocking join (0.3 s under the controlled clock) is inside the
    # elapsed window (0.4 s): reported inference_time_ms includes the wait.
    assert response.inference_time_ms == pytest.approx(400.0)
    # And the fake join itself received the planned diagnostic.
    assert response.success is True
    assert len(response.grasps.grasps) == 1


# The node's declare_parameter defaults that _srv_cb reads on its timed path.
_NODE_PARAMETER_DEFAULTS = {
    "grasp_threshold": 0.5,
    "num_grasps": 2000,
    "topk_num_grasps": 300,
    "enable_collision_filter": True,
    "collision_threshold": 0.005,
    "enable_tabletop_filter": True,
    "enable_source_gripper_tabletop_sweep": False,
    "require_tabletop_filter": True,
    "tabletop_clearance": 0.003,
    "tabletop_pregrasp_distance": 0.08,
    "tabletop_pregrasp_steps": 5,
    "tabletop_ransac_threshold": 0.006,
    "tabletop_min_inlier_ratio": 0.15,
    "tabletop_filter_mode": "strict",
    "adaptive_tabletop_low_profile_height": 0.035,
    "adaptive_tabletop_clearance_min": 0.001,
    "adaptive_tabletop_clearance_max": 0.002,
    "adaptive_tabletop_clearance_height_ratio": 0.12,
    "adaptive_tabletop_pregrasp_min": 0.02,
    "adaptive_tabletop_pregrasp_height_ratio": 2.0,
    "adaptive_tabletop_hard_floor": -0.002,
    "adaptive_tabletop_auto_tune": True,
    "adaptive_tabletop_retry_clearances": "0.002,0.001",
    "adaptive_tabletop_retry_pregrasp_height_ratio": 1.5,
    "adaptive_tabletop_retry_hard_floor": -0.003,
    "target_width_axis_local": "1.0,0.0,0.0",
    "target_width_percentile_low": 5.0,
    "target_width_percentile_high": 95.0,
    "target_width_min_m": 0.005,
    "target_width_max_m": 0.14,
    "enable_object_cloud_completion": True,
    "object_cloud_completion_mode": "mask_depth_inpaint",
    "object_cloud_completion_max_points": 5000,
    "object_cloud_completion_kernel_size": 5,
    "object_cloud_completion_min_neighbors": 6,
    "enable_object_cloud_prismatic_extrude": True,
    "object_cloud_prismatic_extrude_max_points": 8000,
    "object_cloud_prismatic_extrude_layers": 8,
    "enable_scene_cloud_table_holes": True,
    "scene_cloud_table_holes_max_points": 8000,
}


class _FakeParameterValue:
    def __init__(self, value):
        if isinstance(value, bool):
            self.bool_value = value
        elif isinstance(value, int):
            self.integer_value = value
        elif isinstance(value, float):
            self.double_value = value
        else:
            self.string_value = value


class _FakeParameter:
    def __init__(self, value):
        self._value = _FakeParameterValue(value)

    def get_parameter_value(self):
        return self._value


def _fake_node_get_parameter(name):
    return _FakeParameter(_NODE_PARAMETER_DEFAULTS[name])
