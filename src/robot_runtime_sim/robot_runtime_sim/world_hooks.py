"""The generic-launch world hooks (openspec sim-platform-registry D5).

Generic launch owns the world configuration on a physics platform: it
resolves the scene and the spawn pose once (D7) and hands them to the
platform's ``start_world`` hook, referenced from the platform spec
(``SimPlatform.world``). The hook is the simulator process host where one
exists:

- Gazebo starts the scene world (through the shared launcher in ``world.py``:
  the ``gz_sim.launch.py`` include plus the ``/clock`` bridge — the same
  actions the provider hand-off has always started) and returns the gz world
  instance name the provider creates its entity in. A scene that fails to
  load falls back to the ``empty`` scene with a warning; the hook reports
  the final scene through ``SimWorldStartup.scene`` (``""`` after a
  fallback) so the world owner re-resolves the scene identity and the spawn
  pose (D7: ``robot_spawn`` else the origin — the failed scene's layout is
  never consulted).
- MuJoCo starts nothing: one process is both simulator and controller
  manager, and the provider's embodiment starts it with the composed
  robot+scene model. The hook returns ``world=scene`` so the provider and
  the scene task share the scene identity, and ``scene=None`` (the scene
  it was asked for is the world identity; no fallback can happen here —
  the embodiment's composition owns the scene).
"""

from __future__ import annotations

import logging

from launch.actions import SetEnvironmentVariable

from robot_runtime.sim_platforms import SimWorldStartup
from robot_runtime_sim.world import gazebo_world_actions

logger = logging.getLogger(__name__)

#: The gz world instance name of the sim_models scene worlds (and the
#: provider-alone empty world): the ``<world name="...">`` the provider's
#: ``create`` must target.
_GAZEBO_WORLD_NAME = "demo"


def _gazebo_gui_env_actions(robot_config: dict) -> list:
    """Optional GUI / EGL workaround env actions (shared by both Gazebo modes)."""
    actions = []
    gz_gui = robot_config.get("gazebo_gui") or {}
    qt_platform = gz_gui.get("qt_platform") or robot_config.get("gazebo_qt_platform")
    if qt_platform:
        actions.append(SetEnvironmentVariable("QT_QPA_PLATFORM", str(qt_platform)))
        logger.info(f"Gazebo GUI: QT_QPA_PLATFORM={qt_platform}")
    libgl_sw = gz_gui.get("libgl_always_software", robot_config.get("gazebo_libgl_always_software"))
    if libgl_sw:
        actions.append(SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"))
        logger.info(
            "Gazebo GUI: LIBGL_ALWAYS_SOFTWARE=1 "
            "(software GL; set gazebo_gui.libgl_always_software: false if GPU view works)"
        )
    return actions


def gazebo_start_world(robot_config: dict, *, scene: str, spawn_pose) -> SimWorldStartup:
    """The Gazebo world hook (D5): start the scene world, return the world name.

    Reuses the provider hand-off's world start verbatim — the shared launcher
    (D6) with the scene-resolved world file (a scene that fails to load falls
    back to the ``empty`` scene with a warning) — after the optional GUI/EGL
    env workarounds. The gz world instance name (``demo``) is the hand-off
    identifier the provider receives as ``sim_world`` and creates its entity
    in. No ``create`` and no joint-state or camera bridges: the embodiment is
    the provider's.

    The final scene is reported back through ``startup.scene`` (the
    requested scene when it came up, ``""`` after the fallback): the world
    owner, not this hook, owns the scene identity and the spawn pose, so
    it — not this hook — re-resolves them when the requested scene failed
    here.
    """
    from sim_models.scene_compiler import get_gazebo_world_path

    actions: list = []
    actions.extend(_gazebo_gui_env_actions(robot_config))

    scene_name = str(scene or "").strip()
    world_path = ""
    if scene_name:
        try:
            world_path = str(get_gazebo_world_path(scene_name))
            logger.info(f"sim_models scene '{scene_name}' → {world_path}")
        except Exception as e:  # noqa: BLE001 - a broken scene falls back to the empty world, never fails the launch
            logger.warning(f"scene '{scene_name}' load failed: {e}; falling back to the empty scene world")
            scene_name = ""
            world_path = ""
    if not world_path:
        # No scene configured (or the configured scene failed to load): the
        # generic fallback world is the sim_models ``empty`` scene.
        world_path = str(get_gazebo_world_path("empty"))
        logger.info(f"sim_models scene 'empty' → {world_path}")

    gazebo_world_name = str(robot_config.get("gazebo_world_name", _GAZEBO_WORLD_NAME) or _GAZEBO_WORLD_NAME)
    logger.info(f"Gazebo world name (generic world hook): {gazebo_world_name}")

    actions.extend(gazebo_world_actions(world_path, gazebo_world_name))
    # ``scene_name`` now holds the FINAL scene (unchanged when the requested
    # one came up, "" after a fallback): report it so simulation.py re-resolves
    # scene_name + spawn against the world that is actually up.
    return SimWorldStartup(actions=actions, world=gazebo_world_name, scene=scene_name)


def mujoco_start_world(robot_config: dict, *, scene: str, spawn_pose) -> SimWorldStartup:
    """The MuJoCo world hook (D5): no process; the world identity is the scene.

    The provider's embodiment starts the single simulator process with the
    composed robot+scene model, so the hook starts nothing and returns the
    scene name as the hand-off identifier (``sim_world``) — the provider
    composes with it and the scene task shares the identity.
    """
    scene_name = str(scene or "").strip()
    logger.info(
        f"MuJoCo world hook: scene={scene_name!r}, spawn_pose={tuple(spawn_pose)} (the provider starts the simulator)"
    )
    return SimWorldStartup(actions=[], world=scene_name)
