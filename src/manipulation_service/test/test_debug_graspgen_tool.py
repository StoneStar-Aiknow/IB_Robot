"""Committed tests for the installed debug_graspgen tool (issue #176, D6.5).

Covers the D4 JSON serialisation (a diagnostic carrying a started background
table-fit thread, a table-plane normal, an object-top point and point clouds),
the missing-Open3D export policy (inference stubbed, ``open3d`` import forced
to fail -> JSON written, ``[skipped] open3d`` line, success), and the
incomplete-fixture exit codes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from manipulation_service import debug_graspgen as tool
from manipulation_service.grasp_types import GraspCandidate, GraspDiagnostic
from manipulation_service.table_plane import start_execution_table_fit


def _threaded_diagnostic():
    """A diagnostic carrying a started background fit thread and all arrays."""
    rng = np.random.default_rng(5)
    diag = GraspDiagnostic()
    diag.object_pc_raw = rng.uniform(-0.02, 0.02, size=(60, 3)).astype(np.float32) + np.array(
        [0.0, 0.0, 0.5], np.float32
    )
    diag.object_pc_after_completion = rng.uniform(-0.02, 0.02, size=(80, 3)).astype(np.float32) + np.array(
        [0.0, 0.0, 0.5], np.float32
    )
    diag.object_pc_inference_input = diag.object_pc_after_completion[::2].copy()
    scene = np.tile([0.0, 0.0, 0.5], (600, 1)).astype(np.float32)
    scene += rng.normal(0.0, 0.0005, scene.shape).astype(np.float32)
    diag.scene_pc_after_completion = scene
    object_pc = np.tile([0.0, 0.0, 0.52], (200, 1)).astype(np.float32)
    diag.object_pc_after_completion = object_pc
    diag.tabletop_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    diag.tabletop_plane_d = -0.5
    diag.tabletop_object_top_xyz = np.array([0.001, -0.002, 0.51], dtype=np.float64)
    diag.tabletop_inlier_ratio = np.float64(0.9)
    diag.tabletop_ransac_iterations = np.int64(64)
    start_execution_table_fit(diag)
    return diag


def test_result_json_serialises_threaded_diagnostic_without_deep_copy(tmp_path):
    diag = _threaded_diagnostic()
    candidates = [
        GraspCandidate(pose_4x4=np.eye(4, dtype=np.float32), confidence=0.9),
    ]

    out_path = tmp_path / "grasp_result.json"
    tool._save_json(candidates, 1.234, out_path, "robotiq", diag)
    diag._execution_table_fit_thread.join()

    data = json.loads(out_path.read_text(encoding="utf-8"))

    # Point clouds stay as shape/dtype summaries; underscore internals absent.
    diagnostic = data["diagnostic"]
    assert diagnostic["object_pc_raw_shape"] == [60, 3]
    assert diagnostic["object_pc_raw_dtype"] == "float32"
    assert "object_pc_raw" not in diagnostic
    assert "_execution_table_fit_thread" not in diagnostic
    assert "_execution_table_fit_holder" not in diagnostic
    # Other ndarrays become JSON lists; NumPy scalars become Python scalars.
    assert diagnostic["tabletop_plane_normal"] == [0.0, 0.0, 1.0]
    assert diagnostic["tabletop_object_top_xyz"] == [0.001, -0.002, 0.51]
    assert diagnostic["tabletop_inlier_ratio"] == 0.9
    assert diagnostic["tabletop_ransac_iterations"] == 64
    assert data["num_grasps"] == 1
    assert data["gripper"] == "robotiq"
    assert data["grasps"][0]["pose_4x4_rowmajor"] == np.eye(4, dtype=np.float32).flatten().tolist()


def _make_fixture(tmp_path, *, with_depth=True, with_mask=True):
    fixture = tmp_path / "fixture"
    fixture.mkdir(exist_ok=True)
    if with_depth:
        np.save(fixture / "depth_raw.npy", np.full((96, 128), 500, dtype=np.uint16))
    if with_mask:
        import cv2

        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[32:64, 48:88] = 255
        cv2.imwrite(str(fixture / "mask_object.png"), mask)
    return fixture


def test_missing_depth_file_exits_false_without_traceback(tmp_path, capsys):
    fixture = _make_fixture(tmp_path, with_depth=False)

    assert tool.run_on_fixture(str(fixture)) is False
    captured = capsys.readouterr()
    assert f"ERROR: {fixture / 'depth_raw.npy'} not found" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_missing_mask_file_exits_false_without_traceback(tmp_path, capsys):
    fixture = _make_fixture(tmp_path, with_mask=False)

    assert tool.run_on_fixture(str(fixture)) is False
    captured = capsys.readouterr()
    assert f"ERROR: no mask_*.png found in {fixture}" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_missing_open3d_policy_names_skipped_exports(capsys):
    tool._print_open3d_skipped(show=False)
    captured = capsys.readouterr()
    assert captured.out.startswith("[skipped] open3d not installed: ")
    for name in (
        "object_cloud.ply",
        "scene_cloud.ply",
        "grasp_cloud.ply",
        "grasp_grippers.ply",
        "grasp_lines.ply",
        "grasp_preview.png",
    ):
        assert name in captured.out
    assert "interactive Open3D viewer" not in captured.out

    tool._print_open3d_skipped(show=True)
    assert "interactive Open3D viewer" in capsys.readouterr().out


def test_missing_open3d_still_writes_json_and_succeeds(tmp_path, monkeypatch, capsys):
    """D4 export policy: planning succeeds, open3d import fails, exit 0.

    Inference is stubbed at the wrapper level (fixed poses/confidences), the
    real runtime composition builds a registry but no model is loaded, and
    ``import open3d`` is forced to fail: the tool must still write
    ``grasp_result.json`` and report success.
    """
    import torch

    import manipulation_service.graspgen_wrapper as wrapper_module

    fixture = _make_fixture(tmp_path)
    (fixture / "result.json").write_text(
        json.dumps({"camera_info": {"k": [500.0, 0, 64, 0, 500.0, 48, 0, 0, 1]}}),
        encoding="utf-8",
    )

    monkeypatch.setitem(sys.modules, "open3d", None)

    rng = np.random.default_rng(3)
    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    poses[:, 2, 3] = [0.505, 0.506, 0.507]
    confidences = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    def _stub_run_batched_inference(self, object_pc, **_kwargs):
        return torch.from_numpy(poses.copy()), torch.from_numpy(confidences.copy())

    monkeypatch.setattr(
        wrapper_module.GraspGenWrapper,
        "_run_batched_inference",
        _stub_run_batched_inference,
    )

    def _stub_ensure_collision_geometry(self):
        self._gripper_info = "stub"
        self._collision_mesh = "stub-mesh"
        self._collision_vertices = rng.uniform(-0.01, 0.01, size=(30, 3)).astype(np.float64)

    monkeypatch.setattr(
        wrapper_module.GraspGenWrapper,
        "_ensure_collision_geometry",
        _stub_ensure_collision_geometry,
    )

    # The fixture run must skip the real backend construction (no manifest on
    # this test host): stub the wrapper class itself.
    class _NoBackendWrapper:
        def __init__(self, **_kwargs):
            self.gripper_name = "robotiq_test"
            self.collision_gripper_name = "robotiq_test"

        def plan_grasps(self, **kwargs):
            wrapper = object.__new__(wrapper_module.GraspGenWrapper)
            wrapper._inference_point_count = 2048
            wrapper._collision_vertices = rng.uniform(-0.01, 0.01, size=(30, 3)).astype(np.float64)
            wrapper._collision_mesh = "stub-mesh"
            wrapper._filter_tabletop_clearance = lambda *a, **k: (a[0], a[1], np.zeros(0))
            return wrapper_module.GraspGenWrapper.plan_grasps(wrapper, **kwargs)

    monkeypatch.setattr(tool, "GraspGenWrapper", _NoBackendWrapper)

    # The collision stage imports grasp_gen.utils.point_cloud_utils lazily;
    # provide a keep-all stub (this host has no grasp_gen runtime here).
    fake_pkg = type(sys)("grasp_gen")
    fake_utils = type(sys)("grasp_gen.utils")
    fake_pcu = type(sys)("grasp_gen.utils.point_cloud_utils")
    fake_pcu.filter_colliding_grasps = lambda scene_pc, grasp_poses, gripper_collision_mesh, collision_threshold: (
        np.array([True] * len(grasp_poses))
    )
    fake_utils.point_cloud_utils = fake_pcu
    fake_pkg.utils = fake_utils
    monkeypatch.setitem(sys.modules, "grasp_gen", fake_pkg)
    monkeypatch.setitem(sys.modules, "grasp_gen.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "grasp_gen.utils.point_cloud_utils", fake_pcu)

    # The tool composes the runtime dependencies like the node; stub the
    # composition so no real model session is needed.
    class _Dependencies:
        class _Providers:
            @staticmethod
            def close():
                return None

        providers = _Providers()
        registry_set = None

    import inference_service.runtime_composition as runtime_composition

    monkeypatch.setattr(
        runtime_composition,
        "build_model_service_runtime_dependencies",
        lambda **_kwargs: _Dependencies(),
    )

    success = tool.run_on_fixture(str(fixture), enable_tabletop_filter=False)

    assert success is True
    out_json = fixture / "graspgen_output" / "grasp_result.json"
    assert out_json.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["num_grasps"] == 3
    captured = capsys.readouterr()
    assert "[skipped] open3d not installed:" in captured.out
    assert "=== TEST PASSED" in captured.out


class _BrokenOpen3dLoader:
    """A loader whose open3d exec raises ModuleNotFoundError(broken_name) (M2)."""

    def __init__(self, broken_name):
        self._broken_name = broken_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named '{self._broken_name}'", name=self._broken_name)


class _BrokenOpen3dFinder:
    """Answers only for top-level ``open3d``; the load raises the given name."""

    def __init__(self, broken_name):
        self._broken_name = broken_name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "open3d":
            import importlib.util

            return importlib.util.spec_from_loader("open3d", _BrokenOpen3dLoader(self._broken_name))
        return None


def _install_common_fixture_stubs(monkeypatch, fixture, rng):
    """The inference/wrapper stubs shared by the export-policy tests."""
    import sys

    import torch

    import manipulation_service.graspgen_wrapper as wrapper_module

    poses = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
    poses[:, 2, 3] = [0.505, 0.506, 0.507]
    confidences = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    def _stub_run_batched_inference(self, object_pc, **_kwargs):
        return torch.from_numpy(poses.copy()), torch.from_numpy(confidences.copy())

    monkeypatch.setattr(
        wrapper_module.GraspGenWrapper,
        "_run_batched_inference",
        _stub_run_batched_inference,
    )

    class _NoBackendWrapper:
        def __init__(self, **_kwargs):
            self.gripper_name = "robotiq_test"
            self.collision_gripper_name = "robotiq_test"

        def plan_grasps(self, **kwargs):
            wrapper = object.__new__(wrapper_module.GraspGenWrapper)
            wrapper._inference_point_count = 2048
            wrapper._collision_vertices = rng.uniform(-0.01, 0.01, size=(30, 3)).astype(np.float64)
            wrapper._collision_mesh = "stub-mesh"
            wrapper._filter_tabletop_clearance = lambda *a, **k: (a[0], a[1], np.zeros(0))
            return wrapper_module.GraspGenWrapper.plan_grasps(wrapper, **kwargs)

    monkeypatch.setattr(tool, "GraspGenWrapper", _NoBackendWrapper)

    # The collision stage imports grasp_gen.utils.point_cloud_utils lazily;
    # provide a keep-all stub (this host has no grasp_gen runtime here).
    fake_pkg = type(sys)("grasp_gen")
    fake_utils = type(sys)("grasp_gen.utils")
    fake_pcu = type(sys)("grasp_gen.utils.point_cloud_utils")
    fake_pcu.filter_colliding_grasps = lambda scene_pc, grasp_poses, gripper_collision_mesh, collision_threshold: (
        np.array([True] * len(grasp_poses))
    )
    fake_utils.point_cloud_utils = fake_pcu
    fake_pkg.utils = fake_utils
    monkeypatch.setitem(sys.modules, "grasp_gen", fake_pkg)
    monkeypatch.setitem(sys.modules, "grasp_gen.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "grasp_gen.utils.point_cloud_utils", fake_pcu)

    class _Dependencies:
        class _Providers:
            @staticmethod
            def close():
                return None

        providers = _Providers()
        registry_set = None

    import inference_service.runtime_composition as runtime_composition

    monkeypatch.setattr(
        runtime_composition,
        "build_model_service_runtime_dependencies",
        lambda **_kwargs: _Dependencies(),
    )
    return sys


@pytest.mark.parametrize(
    "broken_name",
    ["open3d._missing_internal", "some_open3d_dependency"],
)
def test_open3d_internal_dependency_failure_propagates(broken_name, tmp_path, monkeypatch, capsys):
    """M2: a broken open3d install (internal import missing) fails loudly.

    Only ``ModuleNotFoundError`` with ``exc.name`` exactly ``"open3d"`` takes
    the ``[skipped]`` branch; a missing submodule (``open3d._missing_internal``)
    or a missing transitive dependency (``some_open3d_dependency``) means the
    install is broken and must propagate.
    """
    rng = np.random.default_rng(3)
    fixture = _make_fixture(tmp_path)
    (fixture / "result.json").write_text(
        json.dumps({"camera_info": {"k": [500.0, 0, 64, 0, 500.0, 48, 0, 0, 1]}}),
        encoding="utf-8",
    )
    sys = _install_common_fixture_stubs(monkeypatch, fixture, rng)

    sys.modules.pop("open3d", None)
    monkeypatch.setattr(sys, "meta_path", [_BrokenOpen3dFinder(broken_name), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError) as excinfo:
        tool.run_on_fixture(str(fixture), enable_tabletop_filter=False)
    assert excinfo.value.name == broken_name
    captured = capsys.readouterr()
    assert "[skipped] open3d" not in captured.out


def test_open3d_absent_takes_the_skipped_branch(tmp_path, monkeypatch, capsys):
    """M2: exactly a missing open3d itself skips the exports and succeeds."""
    rng = np.random.default_rng(3)
    fixture = _make_fixture(tmp_path)
    (fixture / "result.json").write_text(
        json.dumps({"camera_info": {"k": [500.0, 0, 64, 0, 500.0, 48, 0, 0, 1]}}),
        encoding="utf-8",
    )
    _install_common_fixture_stubs(monkeypatch, fixture, rng)

    import sys

    monkeypatch.setitem(sys.modules, "open3d", None)

    assert tool.run_on_fixture(str(fixture), enable_tabletop_filter=False) is True
    captured = capsys.readouterr()
    assert "[skipped] open3d not installed:" in captured.out


def test_model_dir_resolution_three_cases(monkeypatch):
    """M3: explicit --model-dir wins, then $WORKSPACE, then the package root."""
    from manipulation_service.graspgen_backend import WORKSPACE_ROOT

    # 1. Explicit CLI value wins even with WORKSPACE set.
    monkeypatch.setenv("WORKSPACE", "/some/workspace")
    assert tool._resolve_model_dir("/explicit/bundle") == "/explicit/bundle"

    # 2. WORKSPACE set: $WORKSPACE/models/graspgen.
    assert tool._resolve_model_dir(None) == str(Path("/some/workspace") / "models" / "graspgen")

    # 3. WORKSPACE unset: the package workspace root's models/graspgen
    #    (NOT the wrapper's legacy models/grasp default).
    monkeypatch.delenv("WORKSPACE", raising=False)
    assert tool._resolve_model_dir(None) == str(WORKSPACE_ROOT / "models" / "graspgen")
    assert tool._resolve_model_dir(None).endswith("models/graspgen")


def test_model_dir_manifest_follows_final_dir_and_backend_routing(monkeypatch, tmp_path):
    """M3: the resolved manifest follows the final model dir and routes per backend."""
    monkeypatch.setattr(tool, "_print_cuda_preflight", lambda: None)
    monkeypatch.setattr(
        tool,
        "GraspGenWrapper",
        lambda **kwargs: kwargs,
    )
    import inference_service.runtime_composition as runtime_composition

    class _Providers:
        @staticmethod
        def close():
            return None

    class _Dependencies:
        providers = _Providers()
        registry_set = None

    monkeypatch.setattr(
        runtime_composition,
        "build_model_service_runtime_dependencies",
        lambda **_kwargs: _Dependencies(),
    )

    fixture = _make_fixture(tmp_path)
    (fixture / "result.json").write_text(
        json.dumps({"camera_info": {"k": [500.0, 0, 64, 0, 500.0, 48, 0, 0, 1]}}),
        encoding="utf-8",
    )
    # Short-circuit the plan: wrapper stub returns directly.
    monkeypatch.setattr(tool, "_save_json", lambda *a, **k: None)

    captured = {}

    def fake_save_json(candidates, elapsed, out_path, gripper_name, diagnostic):
        captured["diagnostic"] = diagnostic
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("{}", encoding="utf-8")

    monkeypatch.setattr(tool, "_save_json", fake_save_json)

    class _StubCandidates(dict):
        def __init__(self):
            super().__init__(gripper_name="robotiq_test")

    # run_on_fixture with 0 candidates returns False after _save_json; the
    # wrapper kwargs capture is what matters here.
    class _CaptureWrapper(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)
            raise SystemExit  # stop right after construction

    monkeypatch.setattr(tool, "GraspGenWrapper", _CaptureWrapper)
    with pytest.raises(SystemExit):
        tool.run_on_fixture(str(fixture), model_dir="/my/bundle", manifest_path=None)

    expected_manifest = str(Path("/my/bundle/inference_manifest.json").resolve())
    assert captured["local_manifest_path"] == expected_manifest
    assert captured["ascend_local_manifest_path"] == ""
    assert captured["model_dir"] == "/my/bundle"

    # ascend_local backend routes the same manifest to the ascend path.
    captured.clear()

    class _CaptureAscendWrapper(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured.update(kwargs)
            raise SystemExit

    monkeypatch.setattr(tool, "GraspGenWrapper", _CaptureAscendWrapper)
    with pytest.raises(SystemExit):
        tool.run_on_fixture(
            str(fixture),
            inference_backend="ascend_local",
            model_dir="/my/bundle",
            manifest_path=None,
        )
    assert captured["ascend_local_manifest_path"] == expected_manifest
    assert captured["local_manifest_path"] == ""


class _FakeInfo:
    collision_mesh = "fake-template"


class SimpleNamespaceish:
    points = []
    lines = []
    colors = None
    collision_mesh = "fake-template"


def test_shared_gripper_writers_apply_the_count_rule(tmp_path, monkeypatch):
    """M5: the mesh/line writers own the count rule and skip on zero."""
    from manipulation_service import debug_viz

    calls = []

    monkeypatch.setattr(debug_viz, "make_gripper_mesh_visual", lambda *args: calls.append(("mesh", args[0])) or "m")
    fake_open3d = type(sys)("open3d")
    fake_geometry = type(sys)("open3d.geometry")
    fake_io = type(sys)("open3d.io")

    class _Combined:
        vertices = []
        triangles = []

        def __iadd__(self, other):
            return self

        def compute_vertex_normals(self):
            return None

    fake_geometry.TriangleMesh = _Combined
    fake_open3d.geometry = fake_geometry
    fake_open3d.io = fake_io
    fake_utility = type(sys)("open3d.utility")
    fake_utility.Vector3dVector = lambda values: list(values)
    fake_utility.Vector2iVector = lambda values: list(values)
    fake_utility.Vector3iVector = lambda values: list(values)
    fake_open3d.utility = fake_utility
    monkeypatch.setitem(sys.modules, "open3d.utility", fake_utility)

    fake_robot = type(sys)("grasp_gen.robot")
    fake_pkg = type(sys)("grasp_gen")
    fake_pkg.robot = fake_robot
    fake_robot.get_gripper_info = lambda name: _FakeInfo()
    monkeypatch.setitem(sys.modules, "grasp_gen", fake_pkg)
    monkeypatch.setitem(sys.modules, "grasp_gen.robot", fake_robot)

    written = []

    fake_io.write_triangle_mesh = lambda path, mesh: written.append(("mesh", str(path)))
    fake_io.write_line_set = lambda path, line_set: written.append(("lines", str(path)))
    fake_geometry.LineSet = lambda *a, **k: SimpleNamespaceish()
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)
    monkeypatch.setitem(sys.modules, "open3d.geometry", fake_geometry)
    monkeypatch.setitem(sys.modules, "open3d.io", fake_io)

    class Candidate:
        def __init__(self, i):
            self.pose_4x4 = np.eye(4, dtype=np.float32)
            self.confidence = 0.5 + 0.1 * i

    candidates = [Candidate(i) for i in range(4)]

    # max_grasps 2 -> count 2, writes and returns 2.
    count = debug_viz.save_gripper_meshes(candidates, "robotiq", tmp_path / "m.ply", 2)
    assert count == 2
    assert written == [("mesh", str(tmp_path / "m.ply"))]

    # max_grasps 0 -> count 0, nothing written (skip_empty default).
    written.clear()
    count = debug_viz.save_gripper_meshes(candidates, "robotiq", tmp_path / "m.ply", 0)
    assert count == 0
    assert written == []

    # skip_empty=False writes even for an empty candidate list.
    count = debug_viz.save_gripper_meshes([], "robotiq", tmp_path / "m.ply", 0, skip_empty=False)
    assert count == 0
    assert written == [("mesh", str(tmp_path / "m.ply"))]

    # Same rule for the line writer.
    monkeypatch.setattr(
        debug_viz,
        "build_gripper_lines",
        lambda pose, gripper: [np.zeros((3, 3), dtype=np.float32)],
    )
    count = debug_viz.save_gripper_lines(candidates, "robotiq", tmp_path / "l.ply", 2)
    assert count == 2
    assert written[-1] == ("lines", str(tmp_path / "l.ply"))

    written.clear()
    assert debug_viz.save_gripper_lines(candidates, "robotiq", tmp_path / "l.ply", 0) == 0
    assert written == []
