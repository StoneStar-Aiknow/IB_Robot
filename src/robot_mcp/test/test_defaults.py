import ast
import inspect
from pathlib import Path

from robot_mcp.ros_bridge import RosBridge


def test_build_skill_goal_defaults_to_thirty_seconds():
    timeout_parameter = inspect.signature(RosBridge.build_skill_goal).parameters["timeout_sec"]

    assert timeout_parameter.default == 30.0


def test_execute_skill_uses_catalog_timeout_by_default():
    server_path = Path(__file__).parents[1] / "robot_mcp" / "server.py"
    module = ast.parse(server_path.read_text(encoding="utf-8"))
    execute_skill = next(
        node for node in module.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_skill"
    )
    timeout_index = [argument.arg for argument in execute_skill.args.args].index("timeout_sec")
    first_default_index = len(execute_skill.args.args) - len(execute_skill.args.defaults)
    timeout_default = execute_skill.args.defaults[timeout_index - first_default_index]

    assert ast.literal_eval(timeout_default) == 0.0
