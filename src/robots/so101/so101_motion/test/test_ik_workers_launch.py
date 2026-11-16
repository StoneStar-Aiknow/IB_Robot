import importlib.util
from pathlib import Path

from launch import LaunchContext


def _load_launch_module():
    path = Path(__file__).parents[1] / "launch" / "ik_workers.launch.py"
    spec = importlib.util.spec_from_file_location("ik_workers_launch", path)
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


def _launch_text(value):
    if isinstance(value, tuple):
        return "".join(_launch_text(item) for item in value)
    return getattr(value, "text", str(value))


def test_worker_move_group_uses_configured_arm_only_joint_states():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "worker_count": "1",
            "namespace_prefix": "ik_worker",
            "use_sim_time": "false",
            "log_level": "warn",
            "joint_state_topic": "/arm_joint_state_broadcaster/joint_states",
        }
    )

    worker = module._launch_setup(context)[0]
    remappings = {_launch_text(source): _launch_text(target) for source, target in worker._Node__remappings}

    assert remappings["joint_states"] == "/arm_joint_state_broadcaster/joint_states"


def _parameters_text(worker) -> str:
    """
    Flatten the worker's full parameter structure to text.

    Keys and values; substitution objects contribute their best-effort text.
    """
    chunks: list[str] = []

    def walk(obj):
        if hasattr(obj, "perform") and not isinstance(obj, str | int | float | bool):
            try:
                chunks.append(str(obj.perform(LaunchContext())))
            except Exception:  # noqa: BLE001
                chunks.append(type(obj).__name__)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(key)
                walk(value)
        elif isinstance(obj, list | tuple):
            for item in obj:
                walk(item)
        else:
            chunks.append(str(obj))

    walk(worker._Node__parameters)
    return "\n".join(chunks)


def _launch_text_pairs(worker):
    pairs = []
    for layer in worker._Node__parameters:
        if isinstance(layer, dict):
            for key, value in layer.items():
                text_value = value if not isinstance(value, tuple) else _launch_text(value)
                pairs.append((_launch_text(key), text_value))
    return pairs


def _worker_parameters(worker):
    parameters: dict = {}
    for layer in worker._Node__parameters:
        if isinstance(layer, dict):
            for key, value in layer.items():
                text = _launch_text(key)
                parameters[text] = value if not isinstance(value, tuple) else _launch_text(value)
    return parameters


def test_worker_move_group_uses_the_passed_runtime_description():
    """
    Pin the revision-2 E2E bring-up fix: the workers must plan in the SAME frame.

    The includer passes the effective runtime URDF (lekiwi renames the arm
    root to arm_root; the default-args render's root differs and the workers
    TF-failed on every request).
    """
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "worker_count": "2",
            "namespace_prefix": "ik_worker",
            "use_sim_time": "false",
            "log_level": "warn",
            "joint_state_topic": "/joint_states",
            "robot_description": "<robot name='effective'><link name='arm_root'/></robot>",
        }
    )

    workers = module._launch_setup(context)
    assert len(workers) == 2
    for worker in workers:
        text = _parameters_text(worker)
        assert "<robot name='effective'><link name='arm_root'/></robot>" in text


def test_worker_move_group_renders_the_standalone_description_when_unset():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "worker_count": "1",
            "namespace_prefix": "ik_worker",
            "use_sim_time": "false",
            "log_level": "warn",
            "joint_state_topic": "/joint_states",
        }
    )

    workers = module._launch_setup(context)
    text = _parameters_text(workers[0])
    assert "<robot" in text and "world" in text  # the standalone default render


def test_motion_launch_passes_its_robot_description_to_the_workers():
    """
    Assert the includer's effective runtime URDF reaches the workers.

    The same selector invariant the binding test pins for the primary.
    """
    motion_path = Path(__file__).parents[1] / "launch" / "motion.launch.py"
    source = motion_path.read_text(encoding="utf-8")
    include_marker = "ik_workers = IncludeLaunchDescription("
    include_start = source.index(include_marker)
    include_block = source[include_start:]
    assert '"robot_description": LaunchConfiguration("robot_description")' in include_block
