"""MuJoCo simulation backend adapter (world-only, so101-mujoco-runtime-platform D7).

MuJoCo is served through a runtime provider: the single simulator +
controller-manager process (``mujoco_ros2_control ros2_control_node``) is the
provider's embodiment, and generic launch owns only the world-side values.
``start_world_only`` resolves the scene name and the robot spawn pose
(``simulation.robot_spawn`` > scene layout > zero) and starts nothing; the
provider receives both by name as launch arguments and composes its robot
MJCF with the scene through ``sim_models.scene_compiler.compose_mujoco_model``
(D3). The provider-less full mode this adapter once carried (robot MJCF
generation hard-wired to SO-101 paths, camera remappings, a generic
``ros2_control_node``) was removed: it was SO-101-specific dead code and a
second MuJoCo implementation to maintain.
"""

import copy

from robot_config.logger_utils import get_colored_logger

from .base_adapter import SimBackendAdapter

logger = get_colored_logger("robot_config.sim_backend.mujoco")


class MujocoAdapter(SimBackendAdapter):
    """MuJoCo world-only backend: generic launch hands the scene to the provider."""

    provides_clock = True
    needs_ros2_control = True

    def start_world_only(self, robot_config: dict) -> tuple[list, str, tuple[float, float, float, float]]:
        """Resolve the scene name and spawn pose for the provider (D7).

        Starts no process: the provider owns the single MuJoCo simulator.
        Returns ``(actions, scene_name, spawn_pose)`` — the actions are always
        empty, the scene name is ``simulation.scene`` (empty means the robot
        model runs alone) and the spawn pose (x, y, z, yaw) follows the
        shared precedence ``simulation.robot_spawn`` > scene layout
        ``robot_spawn`` > zero.
        """
        simulation = robot_config.get("simulation") or {}
        scene_name = str(simulation.get("scene", "") or "").strip()
        spawn: dict = copy.deepcopy(simulation.get("robot_spawn", {}) or {})
        if scene_name and not spawn:
            try:
                from sim_models.scene_compiler import get_scene_layout

                layout = get_scene_layout(scene_name)
                spawn = copy.deepcopy(layout.get("robot_spawn", {}) or {})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"could not load layout for scene '{scene_name}': {e}")
        spawn_pose = (
            float(spawn.get("x", 0.0)),
            float(spawn.get("y", 0.0)),
            float(spawn.get("z", 0.0)),
            float(spawn.get("yaw", 0.0)),
        )
        logger.info(f"MuJoCo world-only hand-off: scene={scene_name!r}, spawn_pose={spawn_pose}")
        return [], scene_name, spawn_pose

    def start_backend(self, robot_config: dict) -> tuple:
        """Provider-less MuJoCo bring-up is not offered (D1/D7).

        The adapter stays concrete (``SimBackendAdapter`` is abstract and
        ``get_sim_backend`` instantiates the class); MuJoCo is served through
        a runtime provider, so this always raises. Loader validation and
        launch preparation reject a provider-less ``simulation.platform:
        mujoco`` earlier through ``providerless_sim_platform_error``; this is
        the belt-and-braces gate for direct registry use.
        """
        raise RuntimeError(
            "simulation.platform=mujoco without runtime.provider is not supported: "
            "MuJoCo is served through a runtime provider (select a runtime.provider "
            "whose profile declares the mujoco platform)"
        )

    def load_scene(self, scene_file_path: str) -> list:
        """The provider composes the scene; nothing to load here."""
        return []

    def ensure_controller_manager(self, robot_config: dict) -> list:
        """The provider's simulator embeds the controller manager."""
        return []

    def spawn_peripheral_bridges(self, peripherals: list) -> list:
        """The provider owns the camera sensors and relays; no bridges here."""
        return []

    def update_object_pose(self, object_name: str, pose) -> None:
        """Reserved for episode parametrization."""
        pass
