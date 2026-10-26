"""Simulation launch builders — thin dispatch layer.

The actual simulation logic lives in sim_backend/:
  sim_backend/gazebo_adapter.py  — Gazebo Ignition implementation
  sim_backend/mujoco_adapter.py  — MuJoCo implementation
  sim_backend/mock_adapter.py    — hardware_mock contract backend

robot.launch.py uses get_sim_backend() directly, reading
'simulation.platform' from the robot YAML.

generate_gazebo_nodes() is kept as a convenience helper for
debugging and unit testing; it always uses the Gazebo backend
regardless of YAML platform.
"""

from typing import Any

from robot_config.launch_registry.builder_support import runtime_provider_name
from robot_config.launch_registry.registry import (
    BuilderSpec,
    BuildResult,
    LaunchBuildContext,
    SimulationStartup,
    isolated_config,
)
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.launch_builders.simulation")


def generate_gazebo_nodes(robot_config: dict) -> list:
    """Debug/test helper: directly invoke GazeboAdapter.

    Always uses the Gazebo backend. In normal operation, robot.launch.py
    calls get_sim_backend(platform) instead of this function so that the
    YAML simulation.platform field is respected.

    Args:
        robot_config: Full robot configuration dict from YAML.

    Returns:
        List of launch actions for Gazebo simulation.
    """
    from robot_config.launch_builders import sim_backend as sim_backend_module

    adapter = sim_backend_module.get_sim_backend("gazebo")
    actions, _ = adapter.start_backend(robot_config)
    actions += adapter.spawn_peripheral_bridges(robot_config.get("peripherals", []))
    return actions


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration.
# ---------------------------------------------------------------------------


def build_simulation(context: LaunchBuildContext) -> BuildResult:
    """Design §2 simulation entry: sim backend + peripheral bridges + scene task.

    Always participates (infrastructure entries are never enable-gated
    at the registry level); the build body decides whether backend
    processes are actually created. Hardware launches and benchmark
    targets never enter the sim backend registry — benchmark even though
    its effective use_sim is true ([IBROBOT_BENCHMARK][SKIP]
    simulation_backend) — and a runtime provider owns its own transport;
    all three yield the explicit empty :class:`SimulationStartup`. Mock is
    NOT skipped: the mock contract backend starts through this section
    (legacy behavior).

    Uses a private deep copy of the context config: the Gazebo adapter
    may write ``initial_pose_x/y/z`` from the scene layout / robot_spawn
    into it for the create-entity arguments; those local writes never
    leak back to the shared context or later entries.
    """
    if context.simulation_startup is not None:
        # Pre-seeded by the orchestrator's pre-binding world bootstrap
        # (runtime-simulation-transport D3b): the world is already starting
        # ahead of the provider readiness wait, so this second plan pass
        # contributes no actions and re-uses the resolved startup verbatim.
        return BuildResult(simulation_startup=context.simulation_startup)
    if not context.use_sim:
        # Hardware launch: no simulation backend processes (legacy section 5
        # ran only for use_sim and not benchmark and not provider).
        return BuildResult(simulation_startup=SimulationStartup(entity_action=None))
    if context.is_benchmark:
        # simulation_backend skipped by benchmark target; the
        # [IBROBOT_BENCHMARK][SKIP] simulation_backend marker is already
        # printed at runtime target resolution. Benchmark must never enter
        # the SimBackendAdapter registry even though use_sim=true.
        logger.info("benchmark target: skipping simulation backend (SimBackendAdapter)")
        return BuildResult(simulation_startup=SimulationStartup(entity_action=None))
    if runtime_provider_name(context):
        if context.sim_platform == "gazebo":
            # D4: generic launch owns the world only; the provider owns the
            # embodiment (description, entity, controllers, sensors, facade).
            # get_sim_backend resolves at call time (monkeypatch seam).
            from robot_config.launch_builders import sim_backend as sim_backend_module

            robot_config = context.robot_config
            private_config = isolated_config(robot_config)
            logger.info("========== Generating Gazebo World (provider hand-off) ==========")
            sim_adapter = sim_backend_module.get_sim_backend("gazebo")
            actions, world_name, spawn_pose = sim_adapter.start_world_only(private_config)
            logger.info(f"World-only Gazebo: world={world_name!r}, spawn_pose={spawn_pose}")
            return BuildResult(
                actions=actions,
                simulation_startup=SimulationStartup(entity_action=None, world_name=world_name, spawn_pose=spawn_pose),
            )
        if context.sim_platform == "mujoco":
            # so101-mujoco-runtime-platform D7: generic launch resolves the
            # scene and spawn pose only; the provider starts the single
            # MuJoCo simulator and composes its robot MJCF with the scene.
            from robot_config.launch_builders import sim_backend as sim_backend_module

            robot_config = context.robot_config
            private_config = isolated_config(robot_config)
            logger.info("========== Resolving MuJoCo Scene (provider hand-off) ==========")
            sim_adapter = sim_backend_module.get_sim_backend("mujoco")
            actions, scene_name, spawn_pose = sim_adapter.start_world_only(private_config)
            logger.info(f"World-only MuJoCo: scene={scene_name!r}, spawn_pose={spawn_pose}")
            return BuildResult(
                actions=actions,
                simulation_startup=SimulationStartup(entity_action=None, scene_name=scene_name, spawn_pose=spawn_pose),
            )
        logger.info("runtime.provider active: simulation transport owned by the robot runtime")
        return BuildResult(simulation_startup=SimulationStartup(entity_action=None))

    robot_config = context.robot_config
    sim_platform = context.sim_platform
    private_config = isolated_config(robot_config)
    actions: list[Any] = []
    create_entity = None

    logger.info("========== Generating Simulation Nodes ==========")
    logger.info(f"Sim platform: {sim_platform}")
    try:
        # Call-time resolution keeps the monkeypatch seam (tests patch
        # robot_config.launch_builders.sim_backend.get_sim_backend).
        from robot_config.launch_builders import sim_backend as sim_backend_module

        sim_adapter = sim_backend_module.get_sim_backend(sim_platform)
        sim_nodes, create_entity = sim_adapter.start_backend(private_config)
        sim_nodes += sim_adapter.spawn_peripheral_bridges(private_config.get("peripherals", []))
        actions.extend(sim_nodes)
        logger.info(f"Added {len(sim_nodes)} simulation nodes ({sim_platform})")
    except NotImplementedError:
        logger.warning(
            f"sim platform '{sim_platform}' not implemented yet, "
            f"skipping simulation nodes (set simulation.platform: gazebo to use Gazebo)"
        )
    except Exception as e:
        logger.error(f"generating simulation nodes: {e}")
        raise

    return BuildResult(
        actions=actions,
        simulation_startup=SimulationStartup(entity_action=create_entity),
    )


SIMULATION_BUILDER = BuilderSpec(
    id="simulation",
    factory=f"{__name__}:build_simulation",
    provides=("simulation_startup",),
    # Infrastructure entries always participate (design §2):
    # the build body decides whether backend processes are
    # actually created and returns an explicit empty
    # SimulationStartup otherwise, so controller_startup can
    # always rely on the typed output.
)
