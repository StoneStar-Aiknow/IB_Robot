"""Task registry for sim scene tasks.

Usage::

    from sim_models.tasks import get_task
    task = get_task("pick_banana", ros2_node)
    task.randomize()
"""

from sim_models.tasks.base import EvalResult, SceneTask
from sim_models.tasks.pick_banana import PickBananaTask

_REGISTRY: dict[str, type[SceneTask]] = {
    PickBananaTask.scene_name: PickBananaTask,
}


def get_task(scene_name: str, node) -> SceneTask:
    """Instantiate the task registered for *scene_name*.

    Args:
        scene_name: Scene name (e.g. ``"pick_banana"``).
        node:       Live ``rclpy.node.Node`` passed to the task.

    Raises:
        KeyError: If no task is registered for *scene_name*.
    """
    if scene_name not in _REGISTRY:
        raise KeyError(f"No task registered for scene '{scene_name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[scene_name](node)


__all__ = ["SceneTask", "EvalResult", "get_task"]
