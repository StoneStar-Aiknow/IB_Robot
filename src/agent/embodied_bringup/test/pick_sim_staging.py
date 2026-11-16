"""The pick-sim staging support (lekiwi-nav-grasp revision 2, R1/R2/R8, test-only).

The pick-specific staging the production-pipeline E2E needs, lived here in
embodied_bringup's test package (review r1 item 12: robot_config.testing
keeps the GENERIC navigation staging only): the pick-sim overlay staged
against the shipped base configuration, the GENERATED pick-sim Nav2
parameters (the shipped file plus the single final-approach lever) and the
assembled development staging catalog (the committed test profile + the
enabled skill packages copied from the shipped ``skill_catalog`` share).

The generic navigation staging (base + overlay, the platform rewrite, the
``NAV2_TEST_PARAMS_FILE`` mechanism) stays in
``robot_config.testing.navigation_staging`` — this module builds on it and
resolves the shipped resources through the ament share directories.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

#: The pick-sim overlay of the hybrid-stage E2E (revision 2, R1/R2/R8),
#: staged by stage_pick_sim_config against the shipped base configuration.
PICK_SIM_OVERLAY_NAME = "lekiwi_nav_grasp_pick_sim.yaml"

#: The pick-sim Nav2 params file the stager GENERATES (never committed): the
#: shipped ``nav2_params_navgrasp_sim.yaml`` plus only the final-approach
#: overrides below.
PICK_SIM_NAV2_PARAMS_NAME = "nav2_params_navgrasp_pick_sim.yaml"

#: The final-approach lever (maintainer decision, design.md "Revision-2 fix
#: during E2E bring-up"; lever = the inflation radius, test params only).
#: Measured on the lowered-table scene (2026-11-02, runs/e2e/mujoco-09*): at
#: the shipped 0.40 m the approach stalls ~0.35 m from the table edge (every
#: candidate fails preparation IK); at 0.20 m the inflated ring still swallows
#: the staging goal, so MPPI stalls at the cost cliff at 0.67-0.70 m (edge
#: distance 0.30-0.33 m, cube pan radius 0.32-0.36 m) with ~3 cm run-to-run
#: variance, and candidate preparation flips at pan ~0.35 — some runs pick,
#: some fail all 80. 0.12 m clears the staging goal (x 0.785) from the
#: inflated region entirely, so the stop lands at the goal +/- the 0.05 goal
#: tolerance (pan 0.19-0.29 m — the real robot's validated pick band). Both
#: costmaps.
PICK_SIM_INFLATION_RADIUS_M = 0.12

#: The header written on top of the generated params file (the measurement
#: the overrides encode).
_PICK_SIM_NAV2_PARAMS_HEADER = """\
# GENERATED at staging time by embodied_bringup.test.pick_sim_staging — do
# not edit or commit: the shipped nav2_params_navgrasp_sim.yaml plus ONLY the
# pick-E2E final-approach override (inflation_radius 0.40 -> 0.12 on both
# costmaps). Measured on the lowered-table scene (E2E runs 2026-11-02,
# runs/e2e/mujoco-09*): with 0.40 the base stops ~0.35 m from the table edge
# (every GraspGen candidate fails preparation IK); 0.20 still stalls MPPI at
# the inflated ring's cost cliff (stops 0.67-0.70 m, cube pan 0.32-0.36 m,
# +-3 cm run to run, candidate preparation flips at pan ~0.35); 0.12 clears
# the staging goal (x 0.785) from the inflated region, so the stop lands at
# the goal +/- 0.05 (cube pan 0.19-0.29 m, the real robot's validated pick
# band). Everything else is byte-identical in value: collision monitor,
# footprint/robot_radius, controller motion bounds, goal checker.
"""


def _share_root(package: str) -> Path:
    """The ament share directory of ``package`` (lazy: no ROS import at load)."""
    from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

    return Path(get_package_share_directory(package))


def stage_pick_sim_nav2_params(destination_dir: Path) -> Path:
    """Generate the pick-sim Nav2 params (the shipped file + the one lever).

    Reads the SHIPPED ``robot_navigation`` ``nav2_params_navgrasp_sim.yaml``
    through the ament index, overrides ``inflation_radius`` on both costmaps
    (:data:`PICK_SIM_INFLATION_RADIUS_M` — the only changed value) and writes
    the result next to the staged overlay. The generated file is staging
    output, never committed.
    """
    document = yaml.safe_load(
        (_share_root("robot_navigation") / "config" / "nav2_params_navgrasp_sim.yaml").read_text(encoding="utf-8")
    )
    for costmap in ("local_costmap", "global_costmap"):
        inflation = document[costmap][costmap]["ros__parameters"]["inflation_layer"]
        if float(inflation["inflation_radius"]) == PICK_SIM_INFLATION_RADIUS_M:
            continue  # the shipped value already matches: nothing to override
        inflation["inflation_radius"] = PICK_SIM_INFLATION_RADIUS_M
    path = destination_dir / PICK_SIM_NAV2_PARAMS_NAME
    path.write_text(_PICK_SIM_NAV2_PARAMS_HEADER + yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _stage_development_catalog(staged_catalog: Path, catalog_source: Path) -> None:
    """Assemble the development staging catalog at staging time.

    Only the test PROFILE is committed (``catalog_source`` holds
    ``config/profiles``); the skill packages it enables are COPIED from the
    shipped ``skill_catalog`` share at staging time — the E2E must not carry
    a second, drifting copy of production skill packages.
    """
    profile_dir = catalog_source / "config" / "profiles"
    profiles = sorted(profile_dir.glob("*.yaml"))
    if len(profiles) != 1:
        raise FileNotFoundError(f"the staging catalog must hold exactly one test profile: {profile_dir}")
    profile = yaml.safe_load(profiles[0].read_text(encoding="utf-8"))
    (staged_catalog / "config" / "profiles").mkdir(parents=True, exist_ok=True)
    shutil.copy2(profiles[0], staged_catalog / "config" / "profiles" / profiles[0].name)
    shipped_skills = _share_root("skill_catalog") / "config" / "skills"
    for entry in profile.get("enabled_skills") or []:
        name = str(entry["name"])
        shutil.copytree(shipped_skills / name, staged_catalog / "config" / "skills" / name)


def stage_pick_sim_config(
    destination: Path,
    *,
    sim_platform: str,
    stage: str = "hybrid",
    e2e_package_root: Path | None = None,
    catalog_source: Path | None = None,
    execution_store_path: Path | None = None,
) -> tuple[Path, Path]:
    """Stage the hybrid pick-sim overlay for the production-pipeline E2E
    (lekiwi-nav-grasp-mobile-manipulation revision 2, R1/R2/R8).

    ``destination`` receives a copy of the shipped base configuration, the
    pick-sim overlay next to it (the loader resolves base_config siblings
    only), the GENERATED pick-sim Nav2 parameters (the shipped file plus the
    final-approach override) and an assembled development staging catalog
    (the committed test profile + the enabled skill packages copied from the
    shipped ``skill_catalog`` share). The staged overlay's
    ``simulation.platform`` carries the platform under test (the D14
    gate-staging pattern) and its ``skill_catalog_source_root`` /
    ``embodied.execution_store_path`` are rewritten to the staged copies —
    the run must not write into the source tree or share another stack's
    single-writer execution store. ``stage`` names the nav stage the overlay
    edits (the E2E stages ``hybrid``). ``e2e_package_root`` overrides the
    package this module lives in (this checkout by default);
    ``catalog_source`` overrides the committed test-profile root likewise.

    Returns ``(staged_overlay, staged_catalog)``.
    """
    e2e_package = Path(e2e_package_root) if e2e_package_root is not None else Path(__file__).resolve().parents[1]
    catalog_root = Path(catalog_source) if catalog_source is not None else e2e_package / "test" / "catalog"
    overlay_source = e2e_package / "test" / "config" / PICK_SIM_OVERLAY_NAME
    if not overlay_source.is_file():
        raise FileNotFoundError(f"pick-sim overlay not found: {overlay_source}")

    from robot_config.testing.navigation_staging import robot_config_share_root

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        robot_config_share_root() / "config" / "robots" / "lekiwi_nav_grasp.yaml",
        destination / "lekiwi_nav_grasp.yaml",
    )

    # the pick-sim Nav2 parameters are GENERATED at staging time (the shipped
    # file + the single final-approach lever); the staged overlay's
    # nav2_bringup params_file points at the generated copy
    staged_params = stage_pick_sim_nav2_params(destination)

    staged_catalog = destination / "catalog"
    if staged_catalog.exists():
        shutil.rmtree(staged_catalog)
    _stage_development_catalog(staged_catalog, catalog_root)

    document = yaml.safe_load(overlay_source.read_text(encoding="utf-8"))
    stages = document.get("robot", {}).get("nav_stages")
    stage_section = stages.get(stage) if isinstance(stages, dict) else None
    if not isinstance(stage_section, dict):
        raise ValueError(
            f"the {PICK_SIM_OVERLAY_NAME} overlay must define robot.nav_stages.{stage} (a mapping) "
            f"for staging; got {type(stage_section).__name__}"
        )
    simulation = stage_section.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError(
            f"the {PICK_SIM_OVERLAY_NAME} overlay must define robot.nav_stages.{stage}.simulation "
            f"(a mapping) to carry the platform under test"
        )
    simulation["platform"] = sim_platform

    navigation = stage_section.setdefault("navigation", {})
    navigation.setdefault("nav2_bringup", {})["params_file"] = str(staged_params)

    embodied = stage_section.setdefault("embodied", {})
    embodied["skill_catalog_source_root"] = str(staged_catalog)
    if execution_store_path is not None:
        embodied["execution_store_path"] = str(execution_store_path)

    staged_overlay = destination / PICK_SIM_OVERLAY_NAME
    staged_overlay.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return staged_overlay, staged_catalog
