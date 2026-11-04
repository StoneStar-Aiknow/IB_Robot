"""Simulation launch builders — thin dispatch layer.

The actual simulation logic lives in sim_backend/:
  sim_backend/gazebo_adapter.py  — Gazebo Ignition implementation
  sim_backend/mock_adapter.py    — hardware_mock contract backend

robot.launch.py uses get_sim_backend() directly, reading
'simulation.platform' from the robot YAML. A runtime provider launch never
enters the backend registry: the world side is resolved through the
platform's world hook (sim-platform-registry D5).
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


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration.
# ---------------------------------------------------------------------------


def _resolve_provider_world(context: LaunchBuildContext) -> BuildResult:
    """The registry-driven provider world hand-off (sim-platform-registry D5).

    Generic launch owns the world side only; the provider owns the
    embodiment. The scene (``simulation.scene``) and the spawn pose (D7:
    ``simulation.robot_spawn`` > scene layout > origin) are resolved once,
    before any world hook, so every platform shares one spawn rule; the
    platform's ``SimPlatform.world`` hook then starts whatever world-side
    processes the platform needs (Gazebo: the gz server + ``/clock``
    bridge; MuJoCo: nothing, the provider starts the simulator). A platform
    with ``world=None`` gets the default resolution: ``world=scene`` and no
    actions. ``scene_name`` records the scene asset identity on every
    platform — ``""`` when the configured scene failed to load, in which
    case every platform falls back to the ``empty`` scene and the layout is
    never consulted (D7).

    A hook that fails its OWN scene load (Gazebo's world-file generation,
    which the precheck cannot fully cover: the layout parses and the
    template exists, but the generated world file read/write fails) reports
    the final scene through ``SimWorldStartup.scene``; the scene identity
    and the spawn pose are then re-resolved from that verdict here, because
    this builder — not the hook — owns them. A re-resolved scene of ``""``
    records the fallback, drops the failed scene's layout spawn and keeps
    rule 1 (``robot_spawn``, else the origin), exactly like a precheck
    failure.
    """
    from robot_runtime.sim_platforms import SimPlatformError, get_sim_platform

    robot_config = context.robot_config
    simulation = robot_config.get("simulation") or {}
    platform = str(context.sim_platform or "")
    if platform == "sdk" or not platform:
        # sdk delegates the whole transport to the provider (no registry
        # access): generic launch owns no world side on it.
        logger.info("runtime.provider active on sdk: simulation transport owned by the robot runtime")
        return BuildResult(simulation_startup=SimulationStartup(entity_action=None))
    try:
        spec = get_sim_platform(platform)
    except SimPlatformError:
        # The preparation phase already rejected unregistered selections;
        # reaching here means a direct plan execution bypassed it.
        raise
    scene = str(simulation.get("scene", "") or "").strip()
    if scene and not _scene_loads(scene, platform):
        logger.warning(
            f"scene '{scene}' does not load; falling back to the empty scene (spawn keeps rule 1: robot_spawn, else the origin)"
        )
        scene = ""
    spawn_pose = _resolve_spawn_pose(simulation, scene)

    world_actions: list[Any] = []
    world = scene
    if spec.world:
        from robot_config.launch_registry.registry import resolve_builder_factory

        startup = resolve_builder_factory(spec.world)(robot_config, scene=scene, spawn_pose=spawn_pose)
        world_actions = list(startup.actions)
        world = str(startup.world or "")
        final_scene = getattr(startup, "scene", None)
        if final_scene is not None and str(final_scene) != scene:
            # The hook's own scene attempt failed and it fell back (D5): the
            # final scene — "" for the fallback-to-empty case — is the
            # authoritative scene identity, and the spawn pose re-resolves
            # from it (D7: robot_spawn, else the origin; the failed scene's
            # layout is never consulted).
            scene = str(final_scene)
            spawn_pose = _resolve_spawn_pose(simulation, scene)
    return BuildResult(
        actions=world_actions,
        simulation_startup=SimulationStartup(
            entity_action=None,
            world=world,
            spawn_pose=spawn_pose,
            scene_name=scene,
        ),
    )


def _resolve_spawn_pose(simulation: dict, scene: str) -> tuple[float, float, float, float]:
    """The one spawn-pose rule (D7), resolved through sim_models."""
    from sim_models.scene_compiler import resolve_robot_spawn

    return resolve_robot_spawn(simulation, scene)


def _scene_loads(scene: str, platform: str) -> bool:
    """Whether the scene loads for ``platform`` (D7 fallback rule).

    A scene that fails to load keeps today's Gazebo semantics on every
    platform: the world falls back to the ``empty`` scene with a warning,
    ``SimulationStartup.scene_name`` becomes ``""`` and the spawn is rule 1
    (``robot_spawn``), else the origin — the layout is never consulted. The
    scene's layout must load; a platform sim_models ships templates for must
    also find its template (other platforms bring their own scene assets).

    This is the precheck — it proves the scene resolves. A platform whose
    hook still fails to build its own world from a precheck-passing scene
    (Gazebo's world-file generation) reports the fallback through
    ``SimWorldStartup.scene`` and ``_resolve_provider_world`` re-resolves.
    """
    from sim_models.scene_compiler import get_scene_file, get_scene_layout, scene_template_platforms

    try:
        get_scene_layout(scene)
        if platform in scene_template_platforms():
            get_scene_file(scene, platform)
    except Exception:  # noqa: BLE001 - any load failure means the scene cannot load
        return False
    return True


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
        # sim-platform-registry D5: the registry drives the provider world
        # hand-off — the selected platform's ``world`` hook starts the
        # world-side processes (none for ``world=None``), and generic launch
        # resolved the scene and spawn pose once, before calling it (D7).
        logger.info(f"========== Resolving Provider World ({context.sim_platform}) ==========")
        result = _resolve_provider_world(context)
        logger.info(
            f"Provider world hand-off: platform={context.sim_platform!r}, "
            f"world={result.simulation_startup.world!r}, "
            f"scene={result.simulation_startup.scene_name!r}, "
            f"spawn_pose={result.simulation_startup.spawn_pose}"
        )
        return result

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
