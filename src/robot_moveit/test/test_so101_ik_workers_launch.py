import importlib.util
from pathlib import Path


def _load_launch_module():
    path = Path(__file__).parents[1] / "launch" / "so101_ik_workers.launch.py"
    spec = importlib.util.spec_from_file_location("so101_ik_workers_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_launch_only_leaves_kinematics_capability_enabled():
    module = _load_launch_module()
    disabled = set(module._DISABLED_CAPABILITIES.split())

    assert "move_group/MoveGroupKinematicsService" not in disabled
    assert "move_group/MoveGroupExecuteService" in disabled
    assert "move_group/MoveGroupExecuteTrajectoryAction" in disabled
    assert "move_group/MoveGroupMoveAction" in disabled
    assert "move_group/MoveGroupPlanService" in disabled
    assert "move_group/ApplyPlanningSceneService" in disabled
    assert "move_group/TfPublisher" in disabled
