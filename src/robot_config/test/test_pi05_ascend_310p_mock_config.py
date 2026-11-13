"""The PI0.5 Ascend310P validation configuration runs on the SO-101 runtime.

It is an overlay of so101_single_arm: the policy observes the runtime's top and
wrist cameras and joint state, and commands the runtime's arm and gripper
streams, on the SDK simulation platform (``use_sim:=true``).
"""

from pathlib import Path

import pytest

from robot_config.inference_config import parse_inference_config
from robot_config.loader import load_robot_config_dict
from robot_config.testing.provider_binding import bind_offline

ROBOTS = Path(__file__).resolve().parents[1] / "config" / "robots"
CONFIG_PATH = ROBOTS / "so101_pi05_ascend_310p_mock.yaml"
SO101_PROFILE = (
    Path(__file__).resolve().parents[2] / "robots" / "so101" / "so101_robot" / "profiles" / "so101_single_arm.yaml"
)


def test_pi05_mock_binds_the_so101_runtime_cameras_and_streams() -> None:
    deferred = load_robot_config_dict(CONFIG_PATH, defer_interface_binding=True, require_local_model_artifacts=False)
    assert deferred["runtime"]["provider"] == "so101_robot"
    assert deferred["simulation"]["platform"] == "sdk"

    bound = bind_offline(deferred, SO101_PROFILE)

    assert {obs["key"]: obs["topic"] for obs in bound["contract"]["observations"]} == {
        "observation.images.top": "/camera/top/image_raw",
        "observation.images.wrist": "/camera/wrist/image_raw",
        "observation.state": "/joint_states",
    }
    assert [action["publish"]["topic"] for action in bound["contract"]["actions"]] == [
        "/arm_position_controller/commands",
        "/gripper_position_controller/commands",
    ]


def test_pi05_ascend_310p_mock_pipeline(monkeypatch) -> None:
    workspace = CONFIG_PATH.parents[4]
    bundle = workspace / "models" / "pi05" / "pi05-doublecam-fp32" / "019200-torch-npu"
    if not bundle.is_dir():
        pytest.skip(f"PI0.5 Torch-NPU bundle is not installed at {bundle}")
    monkeypatch.setenv("WORKSPACE", str(workspace))
    config = load_robot_config_dict(CONFIG_PATH, defer_interface_binding=True)
    inference = parse_inference_config(config, "model_inference")
    pipeline = inference.pipelines["policy"]

    assert pipeline.deployment == "torch-npu-fast"
    assert pipeline.execution_mode == "monolithic"
    assert pipeline.runtime_options == {"model_dtype": "fp16"}
    assert pipeline.request_timeout == 1800.0
    assert pipeline.validated_manifest.manifest.model.model_type == "pi05"
