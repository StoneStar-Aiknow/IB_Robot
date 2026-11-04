"""The X2 grasp catalog compiles offline and binds the grasp pipeline to the model each host deployment serves."""

import pytest

from robot_config.config_path import resolve_robot_config_path
from robot_config.loader import load_robot_config_dict


def _snapshot(monkeypatch, name):
    from pathlib import Path

    from robot_skill_cli.catalog import compile_local_snapshot

    # The executor identity reads the GraspGen-X bundle's manifest under the workspace.
    workspace = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    monkeypatch.setenv("WORKSPACE", str(workspace))
    config_path = resolve_robot_config_path(config_name=name)
    config = load_robot_config_dict(config_path, defer_interface_binding=True)
    return compile_local_snapshot(config, config_path)


# Each configuration resolves its perception bundles and the executor identity
# reads the GraspGen-X manifest: skip, rather than fail, where they are absent.
@pytest.mark.parametrize(
    ("name", "deployment"),
    [
        pytest.param(
            "aimdk_x2_grasp_cuda",
            "torch_cuda",
            marks=pytest.mark.model_bundle("grounded_sam2_swint_ogc", "graspgenx_omnihand_left_x2"),
        ),
        pytest.param(
            "aimdk_x2_grasp_310p",
            "torch_npu",
            marks=pytest.mark.model_bundle(
                "grounding_dino_swint_seq8_1280x720", "sam2.1_hiera_tiny_prompt_ascend", "graspgenx_omnihand_left_x2"
            ),
        ),
    ],
)
def test_x2_grasp_exposes_pick_object_through_the_grasp_pipeline(monkeypatch, name, deployment):
    snapshot = _snapshot(monkeypatch, name)
    assert snapshot.enabled_skill_names == ("pick_object",)
    assert snapshot.robot_context.required_control_mode == "moveit_planning"
    template = snapshot.templates["pick_object"]
    assert template["executor"] == "grasp_pipeline" and template["timeout_sec"] == 120.0
    executor = snapshot.delegated_executors["grasp_pipeline"]
    assert executor.endpoint_name == "/manipulation/execute_pick"
    # The deployment the host's GraspGen-X service runs, not the other host's.
    assert executor.model_deployment_name == deployment and executor.model_fingerprint
