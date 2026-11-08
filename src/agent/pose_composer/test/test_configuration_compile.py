"""The offline command consumes the deployment configuration without a prior artifact."""

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mujoco")

from pose_composer import compile_library as compiler  # noqa: E402
from robot_config.loader import _load_robot_section  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="session", autouse=True)
def configured():
    # These command tests mock compilation, not the model or ROS transport.
    return None


@pytest.mark.parametrize("sides,arm_side", [(["right"], "right"), (["left"], "left"), (["left", "right"], "auto")])
@pytest.mark.parametrize("hand_policy", ["online", "off"])
def test_compiler_uses_the_deployed_mounting_and_task_pose_snapshot(
    tmp_path, monkeypatch, sides, arm_side, hand_policy
):
    source = ROOT / "src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
    _, config = _load_robot_section(source)
    config["task_poses"]["file"] = str(source.parent / config["task_poses"]["file"])
    config["grasp_execution"]["backend_options"].update(installed_hands=sides, arm_side=arm_side)
    config["pose_composer"]["path_validation"]["hand"] = hand_policy
    output = tmp_path / "assets/fixed_poses.json"
    config["pose_composer"]["fixed_library"] = str(output)
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump({"robot": config}))
    observed = {}

    def compile_capture(mjcf, profile, limits, seeds, selected, *, task_poses, hand_path_check):
        observed.update(profile=profile, limits=limits, snapshot=task_poses, hand_path_check=hand_path_check)
        assert Path(mjcf).name == "scene.xml" and Path(seeds).is_file()
        assert selected == ["arms_up"]
        return {"poses": {"arms_up": {}}, "digest": "command-test"}

    monkeypatch.setattr(compiler, "compile_library", compile_capture)
    compiler.main(["--robot-config", str(path), "--pose", "arms_up"])
    assert observed["limits"] == config.get("arm_limits")
    assert Path(observed["profile"]).is_file()
    assert observed["hand_path_check"] == hand_policy
    for pose in observed["snapshot"]["poses"].values():
        assert pose["hands"] == dict.fromkeys(sides, "open")
        assert pose["hardware_approved"] is True
    assert json.loads(output.read_text())["digest"] == "command-test"


@pytest.mark.parametrize("override", ["--profile", "--limits", "--task-poses"])
def test_robot_config_compile_refuses_conflicting_execution_inputs(override):
    with pytest.raises(SystemExit) as error:
        compiler.main(["--robot-config", "unused.yaml", "--pose", "arms_up", override, "unused.yaml"])
    assert error.value.code == 2


def test_legacy_explicit_compile_inputs_remain_available(tmp_path, monkeypatch):
    limits = tmp_path / "limits.yaml"
    limits.write_text("left_elbow_joint: [-2.0, 0.0]\n")
    output = tmp_path / "artifact.json"
    observed = {}

    def compile_capture(mjcf, profile, bounds, seeds, selected, *, task_poses, hand_path_check):
        observed.update(mjcf=mjcf, profile=profile, bounds=bounds, seeds=seeds, selected=selected)
        assert task_poses is None and hand_path_check == "online"
        return {"poses": {"arms_up": {}}, "digest": "legacy-command"}

    monkeypatch.setattr(compiler, "compile_library", compile_capture)
    compiler.main(
        [
            "--mjcf",
            "scene.xml",
            "--profile",
            "profile.yaml",
            "--limits",
            str(limits),
            "--seeds",
            "seeds.yaml",
            "--pose",
            "arms_up",
            "--output",
            str(output),
        ]
    )
    assert observed == {
        "mjcf": "scene.xml",
        "profile": "profile.yaml",
        "bounds": {"left_elbow_joint": [-2.0, 0.0]},
        "seeds": "seeds.yaml",
        "selected": ["arms_up"],
    }
    assert json.loads(output.read_text())["digest"] == "legacy-command"
