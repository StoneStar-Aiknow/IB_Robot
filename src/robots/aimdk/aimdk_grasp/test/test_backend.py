"""The X2 grasp backends: their options, the gripper they plan for, and their assembly."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from aimdk_grasp.backend import (
    COMMAND_INTERFACES,
    _options,
    _require_declared,
    _ResumingPlanner,
    gripper_directory,
)

WORKSPACE = Path(__file__).resolve().parents[5]
BUNDLE = WORKSPACE / "models" / "graspgenx_omnihand_left_x2"
PROFILE = Path(__file__).resolve().parents[2] / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"


def test_options_default_to_the_x2_deployment_and_refuse_what_they_do_not_know():
    options = _options({})
    assert options["service_timeout_s"] == 4.0 and options["watch"] == "stop" and options["segment_service"] == ""
    with pytest.raises(ValueError, match="unknown keys \\['detect'\\]"):
        _options({"backend_options": {"detect": "/x"}})
    for bad in ("4", float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="service_timeout_s must be a finite number"):
            _options({"backend_options": {"service_timeout_s": bad}})
    with pytest.raises(ValueError, match="watch must be stop, log or off"):
        _options({"backend_options": {"watch": "on"}})
    assert (options["watch_frame_timeout_s"], options["watch_unknown_timeout_s"], options["watch_image_age_s"]) == (
        1.0,
        4.0,
        0.5,
    )
    with pytest.raises(ValueError, match="watch_unknown_timeout_s must be a finite number of seconds"):
        _options({"backend_options": {"watch_unknown_timeout_s": 0.0}})
    with pytest.raises(ValueError, match="watch_frame_timeout_s must be a finite number"):
        _options({"backend_options": {"watch_frame_timeout_s": float("inf")}})
    with pytest.raises(ValueError, match="status_timeout_s must be a finite number"):
        _options({"backend_options": {"head": {"status_timeout_s": float("nan")}}})
    with pytest.raises(ValueError, match="head has unknown keys \\['nodes'\\]"):
        _options({"backend_options": {"head": {"nodes": "4-7"}}})
    assert options["arm_side"] == "left" and _options({"backend_options": {"arm_side": "right"}})["arm_side"] == "right"
    with pytest.raises(ValueError, match="arm_side must be left or right"):
        _options({"backend_options": {"arm_side": "both"}})


def test_the_x2_backends_refuse_a_configuration_that_names_no_model_deployment():
    """The shared aimdk_x2_grasp names none; only a host deployment (CUDA or 310P) does."""
    from aimdk_grasp.backend import x2_robot

    with pytest.raises(ValueError, match="aimdk_x2_grasp_cuda or aimdk_x2_grasp_310p"):
        x2_robot({"command_interfaces": list(COMMAND_INTERFACES), "model_bundle_path": "/nowhere"})


def test_the_pipeline_side_of_the_split_needs_no_cuda_stack():
    """The robot-side host (the 310P) runs the remote backends without torch, Warp or cuRobo."""
    import subprocess
    import sys

    code = (
        "import sys, aimdk_grasp.remote_head, aimdk_grasp.split; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'torch', 'warp', 'curobo'}))"
    )
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == "[]"


def test_the_configuration_must_declare_the_streams_the_backends_command():
    _require_declared({"command_interfaces": list(COMMAND_INTERFACES)})
    with pytest.raises(ValueError, match="missing \\['joint.hand_stream'\\]"):
        _require_declared({"command_interfaces": ["joint.arm_stream"]})


class RecordingImu:
    def __init__(self):
        self.calls = []

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")


def test_the_imu_runs_on_after_a_plan_and_stays_paused_when_planning_fails():
    imu = RecordingImu()
    planner = _ResumingPlanner(SimpleNamespace(plan=lambda observation, proposals, **budget: "plan"), imu)
    assert planner.plan(None, None) == "plan" and imu.calls == ["resume"]

    def unplannable(observation, proposals, **budget):
        raise RuntimeError("no plan")

    planner = _ResumingPlanner(SimpleNamespace(plan=unplannable), imu)
    with pytest.raises(RuntimeError, match="no plan"):
        planner.plan(None, None)
    assert imu.calls == ["resume", "pause"]


def make_bundle(root: Path, **assets) -> Path:
    (root / "assets").mkdir(parents=True)
    content = {"model_type": "graspgenx", "gripper_assets_dir": "assets", "gripper_name": "omnihand_left_x2", **assets}
    (root / "assets" / "adapter.json").write_text(json.dumps(content), encoding="utf-8")
    return root


def test_the_gripper_is_the_one_the_deployed_bundle_generates_for(tmp_path, monkeypatch):
    make_bundle(tmp_path / "bundle")
    monkeypatch.setenv("TEST_MODELS", str(tmp_path))
    found = gripper_directory({"model_bundle_path": "$(env TEST_MODELS)/bundle"})
    assert found == tmp_path / "bundle" / "assets" / "x_grippers" / "omnihand_left_x2"
    make_bundle(tmp_path / "other", model_type="grounded_sam2")
    with pytest.raises(ValueError, match="not a GraspGen-X bundle"):
        gripper_directory({"model_bundle_path": str(tmp_path / "other")})
    with pytest.raises(ValueError, match="model_bundle_path"):
        gripper_directory({})
    monkeypatch.delenv("TEST_MODELS")
    with pytest.raises(ValueError, match="environment variable is not set: TEST_MODELS"):
        gripper_directory({"model_bundle_path": "$(env TEST_MODELS)/bundle"})


@pytest.mark.skipif(not (BUNDLE / "assets" / "adapter.json").is_file(), reason="GraspGen-X bundle not installed")
def test_the_backends_assemble_on_the_x2_description_and_wait_for_their_warmup():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("warp")
    from aimdk_grasp.backend import x2_grasp_backends
    from aimdk_grasp.execution_watch import X2TargetWatch
    from aimdk_grasp.grasp_request import X2GraspPlanner
    from aimdk_grasp.omnihand import OmniHandEffector
    from aimdk_grasp.proposer import X2Proposer

    from manipulation_execution.grasp_pipeline.streaming import StreamExecutor
    from robot_runtime.interface_description import build_description
    from robot_runtime.profile import load_profile

    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    node = rclpy.create_node("x2_backend_test")
    grasp_execution = {
        "model_bundle_path": os.fspath(BUNDLE),
        "model_deployment": "torch_cuda",
        "command_interfaces": list(COMMAND_INTERFACES),
        "backend_options": {"service_timeout_s": 0.2, "registration_device": "cpu"},
    }
    split = {**grasp_execution, "backend_options": {**grasp_execution["backend_options"], "head": {"namespace": "/h"}}}
    with pytest.raises(ValueError, match="single-process X2 backends run none"):
        x2_grasp_backends(node=node, grasp_execution=split, description=description)
    backends = x2_grasp_backends(node=node, grasp_execution=grasp_execution, description=description)
    try:
        assert isinstance(backends.proposer, X2Proposer) and isinstance(backends.executor, StreamExecutor)
        assert isinstance(backends.planner._inner, X2GraspPlanner)  # resumes the paused IMU after planning
        assert isinstance(backends.effector, OmniHandEffector) and isinstance(backends.watch, X2TargetWatch)
        # No camera on this graph: the warmup fails and is retried by the node.
        with pytest.raises(RuntimeError, match="warmup on 'a bottle .'.*RGB-D"):
            backends.prepare()
        assert backends.watch.imu.paused  # nothing executes after a warmup
    finally:
        backends.close()
        node.destroy_node()
        if owned and rclpy.ok():
            rclpy.shutdown()


@pytest.mark.skipif(not (BUNDLE / "assets" / "adapter.json").is_file(), reason="GraspGen-X bundle not installed")
def test_arm_side_right_switches_every_part_to_the_right_arm_and_hand():
    from aimdk_grasp.backend import x2_robot
    from aimdk_grasp.joints import RIGHT_ARM
    from aimdk_grasp.robot_model import arm_limits, config_directory

    grasp_execution = {
        "model_bundle_path": os.fspath(BUNDLE),
        "model_deployment": "torch_cuda",
        "command_interfaces": list(COMMAND_INTERFACES),
    }
    left = x2_robot(grasp_execution)
    assert (left.side, left.hand.side, left.gravity.side, left.perception.scene.arm_side) == ("left",) * 4
    right = x2_robot({**grasp_execution, "backend_options": {"arm_side": "right"}})
    assert (right.side, right.hand.side, right.gravity.side, right.perception.scene.arm_side) == ("right",) * 4
    assert right.limits == arm_limits(config_directory("right")) and set(right.limits) == set(RIGHT_ARM)
    # One gripper description serves both hands: the same alias presets, the right hand's native values.
    for name in ("open", "close"):
        assert right.hand.presets[name] == left.hand.presets[name]
