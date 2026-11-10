"""The recovered entry keeps its hardware assets and lifecycle guards usable."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from robot_config.loader import _load_robot_section
from robot_runtime.profile import load_profile

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"


def test_left_hand_entry_exposes_the_commissioned_skill_families():
    _, config = _load_robot_section(CONFIG)
    options = config["grasp_execution"]["backend_options"]
    assert options["installed_hands"] == ["left"] and options["arm_side"] == "left"
    assert config["pose_composer"]["enabled"] and not config["pose_composer"]["blocked_task_poses"]
    catalog = yaml.safe_load(
        (ROOT / "src/agent/skill_catalog/config/profiles/aimdk_x2_voice_mirror_grasp.yaml").read_text()
    )
    names = {item["name"] for item in catalog["enabled_skills"]}
    assert names == set(config["embodied"]["agent"]["test_allowlist"])
    assert "handshake" not in names
    assert {"wave_hand", "free_pose", "return_to_task_pose", "pick_object", "place_in_container"} <= names
    assert catalog["robot_name"] == config["name"]


def test_runtime_profile_reuses_packaged_trajectories_and_external_audio():
    _, config = _load_robot_section(CONFIG)
    path = ROOT / config["runtime"]["profile"]
    profile = load_profile(path)
    assert profile["capabilities"]["interaction.audio_capture"]["channels"] == 2
    for trajectory in profile["vendor"]["named_motion"]["trajectories"].values():
        resolved = (path.parent / trajectory).resolve()
        assert resolved.is_relative_to(ROOT / "src/robots/aimdk/aimdk_robot/profiles/trajectories")
        assert resolved.is_file()


def test_recovered_fixed_library_has_intact_targets_and_left_hand_anchors():
    library = json.loads((ROOT / "commissioning/fixed_poses_voice_mirror_grasp.json").read_text())
    expected = library.pop("digest")
    encoded = json.dumps(library, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert len(library["poses"]) == 12
    assert {"arms_down", "arms_up", "arms_out", "salute"} <= set(library["poses"])
    for pose in library["task_poses"]["poses"].values():
        assert pose["hands"] == {"left": "open"}


def test_failed_microphone_preflight_never_starts_the_pipeline(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in {
        "python3": "#!/bin/bash\nexit 17\n",
        "ros2": f'#!/bin/bash\ntouch "{tmp_path}/unexpected_launch"\n',
    }.items():
        path = binaries / name
        path.write_text(body)
        path.chmod(0o755)
    environment = tmp_path / "site.sh"
    environment.write_text(f'export WORKSPACE="{ROOT}"\n')
    state = tmp_path / "state"
    env = dict(
        os.environ,
        PATH=f"{binaries}:{os.environ['PATH']}",
        IBROBOT_INTERACTION_ENV=str(environment),
        IBROBOT_INTERACTION_STATE=str(state),
        IBROBOT_MANAGE_GRASP_HEAD="0",
    )
    result = subprocess.run(["bash", str(ROOT / "commissioning/start_interaction.sh")], env=env, timeout=10)
    assert result.returncode == 17
    assert not (tmp_path / "unexpected_launch").exists()
    assert not (state / "pipeline.pgid").exists()


def test_stop_refuses_a_pid_owned_by_another_checkout(tmp_path):
    # Own the target process so even a broken stop implementation cannot hit another task.
    target = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        binaries = tmp_path / "bin"
        binaries.mkdir()
        ps = binaries / "ps"
        ps.write_text("#!/bin/bash\necho unrelated-worker\n")
        ps.chmod(0o755)
        state = tmp_path / "state"
        state.mkdir()
        (state / "pipeline.pgid").write_text(str(target.pid))
        env = dict(
            os.environ,
            PATH=f"{binaries}:{os.environ['PATH']}",
            IBROBOT_INTERACTION_STATE=str(state),
            IBROBOT_MANAGE_GRASP_HEAD="0",
        )
        result = subprocess.run(
            ["bash", str(ROOT / "commissioning/stop_interaction.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1 and "ownership mismatch" in result.stderr
        assert target.poll() is None
    finally:
        if target.poll() is None:
            target.terminate()
        target.wait(timeout=5)
