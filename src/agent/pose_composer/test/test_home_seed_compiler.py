"""HOME-bound seeds are compiled and validated without hardware or a ROS graph."""

import copy
from pathlib import Path

import pytest
import yaml
from aimdk_grasp.hand_restore_geometry import DualHandCollision
from aimdk_robot.pose_backend import model

from pose_composer.compile_library import compile_library, update_fixed_catalog
from pose_composer.fixed_backend import FixedBackend, digest
from pose_composer.fixed_library import FixedComposer

SEEDS = Path(__file__).resolve().parents[1] / "seeds/x2_poses.yaml"
PROFILE = Path(__file__).resolve().parents[3] / "robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"
MJCF = Path.home() / ".cache/ibrobot/agibot_x2_urdf/575cc6b988f976c23550e0db85aa1e5475d3652d/X2_URDF-v1.3.0/scene.xml"


@pytest.fixture
def anchors():
    return yaml.safe_load((Path(__file__).parent / "fixtures/x2_home_task_poses.yaml").read_text())


@pytest.fixture
def compile_home(anchors):
    previous = model.load()

    def compile(*, seeds=SEEDS, selected=("arms_down",), task_poses=anchors, **options):
        return compile_library(MJCF, PROFILE, None, seeds, selected, task_poses=task_poses, **options)

    try:
        yield compile
    finally:
        model.configure(MJCF, PROFILE, previous.hard_limits, task_poses=previous.task_poses)


@pytest.fixture
def compiled_home(compile_home, monkeypatch):
    # Snapshot/alias integrity remains testable for an artifact compiled under
    # the previous 0.025 rad envelope. This is not a proof for today's 0.1 rad.
    check = DualHandCollision.check_arm_path
    monkeypatch.setattr(
        DualHandCollision, "check_arm_path", lambda self, *a, **kw: check(self, *a, **{**kw, "finger_radius": 0.025})
    )
    return compile_home()


def test_current_compiler_rejects_home_with_insufficient_widened_finger_clearance(compile_home):
    with pytest.raises(ValueError, match="CPU path rejected.*insufficient hand clearance"):
        compile_home()


def test_explicit_hand_check_off_records_scope_and_keeps_home(compile_home, anchors):
    library = compile_home(hand_path_check="off")
    assert library["compile_validation"] == {"hand": "off", "body": "online"}
    assert library["poses"]["arms_down"]["target"] == anchors["poses"]["grasp_ready"]["joint_positions"]


def test_compiled_down_uses_exact_snapshot_and_the_frozen_backend(compiled_home, anchors):
    backend = FixedBackend(PROFILE, library=compiled_home, task_poses=anchors)
    target = anchors["poses"]["grasp_ready"]["joint_positions"]
    assert compiled_home["poses"]["arms_down"]["target"] == target
    assert backend.task_poses["digest"] == compiled_home["task_poses"]["digest"]
    composer = FixedComposer(backend.library)
    prepared = composer.compose("双臂放下", dict(backend.rest))
    assert prepared.composition.frames == [target]
    assert prepared.pose_id == "arms_down"


@pytest.mark.parametrize("fault", ["changed_home", "missing_home", "altered_target"])
def test_runtime_cannot_load_an_old_or_falsified_home_anchor(compiled_home, anchors, fault):
    data = copy.deepcopy(compiled_home)
    runtime = copy.deepcopy(anchors)
    if fault == "changed_home":
        runtime["poses"]["grasp_ready"]["joint_positions"]["left_shoulder_pitch_joint"] += 0.02
    elif fault == "missing_home":
        runtime = {}
    else:
        data["poses"]["arms_down"]["target"]["left_shoulder_pitch_joint"] += 0.02
        data["digest"] = digest({k: v for k, v in data.items() if k != "digest"})
    with pytest.raises(ValueError, match="snapshot|anchors"):
        FixedBackend(PROFILE, library=data, task_poses=runtime)


def test_compiler_rejects_missing_anchor_instead_of_using_hardcoded_home(compile_home):
    with pytest.raises(ValueError, match="not configured"):
        compile_home(task_poses=None)


def test_old_keep_is_still_rejected_and_not_silently_reinterpreted(tmp_path, compile_home):
    data = yaml.safe_load(SEEDS.read_text())
    seed = next(r for r in data["poses"] if r["id"] == "right_forward")
    seed["description"]["left"] = {"keep": True}
    path = tmp_path / "seeds.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="keep-current"):
        compile_home(seeds=path, selected=["right_forward"])


def test_candidate_catalog_contains_only_compiled_actions_and_no_ambiguous_open_alias(tmp_path, compiled_home):
    source = Path(__file__).resolve().parents[2] / "skill_catalog/config/skills/free_pose/manifest.yaml"
    destination = tmp_path / "config/skills/free_pose/manifest.yaml"
    destination.parent.mkdir(parents=True)
    destination.write_text(source.read_text())
    update_fixed_catalog(compiled_home, tmp_path)
    result = yaml.safe_load(destination.read_text())
    targets = result["capability"]["parameters"]["properties"]["target_name"]["enum"]
    assert "arms_down" in targets and "双臂放下" in targets
    assert "hands_on_hips" not in targets and "张开双手" not in targets
    assert "freeform" not in result["capability"]["parameters"]["properties"]["target_name"]
