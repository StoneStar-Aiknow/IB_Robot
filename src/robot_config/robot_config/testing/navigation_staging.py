"""Staging support for the navigation simulation overlays (D10/D15).

The loader resolves ``base_config`` siblings only, so the Gazebo-free
resolved-config test and the navigation end-to-end test stage the shipped
``config/robots/<base>.yaml`` and the matching ``test/config/<overlay>``
into one directory and launch with ``config_path:=<staged overlay>``. The
``base_config`` parameter selects the runtime under test (D7 of
lekiwi-nav-grasp-sim-platforms): ``lekiwi_lidar`` (base-only) or
``lekiwi_nav_grasp`` (mobile manipulator — the overlay turns off the
consumers of the interfaces the platforms exclude and points Nav2 at the
measured-footprint parameter variant). The same staging covers the
``NAV2_TEST_PARAMS_FILE`` parameter-override mechanism (a temporary Nav2
params variant), which rewrites the overlay's ``params_file`` before
staging, and the per-platform rewrite of ``simulation.platform`` (D15, the
D14 gate staging pattern: the shipped overlay pins ``gazebo``; the staged
copy carries the platform under test so the world owner selects the right
provider branch).

This module holds the GENERIC navigation staging only (review r1 item 12):
the pick-specific staging — the pick-sim overlay, the generated pick-sim
Nav2 parameters and the development staging catalog — lives in
``embodied_bringup``'s test support
(``embodied_bringup/test/pick_sim_staging.py``), next to the E2E it serves.

This is an INSTALLED module (``robot_config.testing``, R8 of
lekiwi-nav-grasp-mobile-manipulation revision 2): tests stage their overlays
through the same code without importing robot_config's source-path ``test``
package. Resources resolve through the ament share directory
(symlinked to this checkout under ``--symlink-install``), or through an
explicit ``package_root`` the caller passes for synthetic layouts. The
staged stage NAME is a parameter (spec review r2-2): the navigation callers
keep ``navigation``; the pick E2E stages the ``hybrid`` stage through its
own staging module.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def robot_config_share_root() -> Path:
    """The robot_config resources root through the ament index.

    The share directory of an ``--symlink-install`` build points at this
    checkout, so the shipped configurations resolve identically from the
    source tree and from an installed layout.
    """
    from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

    return Path(get_package_share_directory("robot_config"))


def overlay_name(base_config: str) -> str:
    """The test-overlay filename of a staging base (D7 parametrization)."""
    return NAVIGATION_STAGING[base_config][1]


#: The staging pairs: staging base -> (base configuration, overlay).
NAVIGATION_STAGING = {
    "lekiwi_lidar": ("lekiwi_lidar.yaml", "lekiwi_navi_sim.yaml"),
    "lekiwi_nav_grasp": ("lekiwi_nav_grasp.yaml", "lekiwi_nav_grasp_navi_sim.yaml"),
}


def stage_navigation_config(
    destination: Path,
    nav2_params_override: str = "",
    sim_platform: str = "gazebo",
    base_config: str = "lekiwi_lidar",
    stage: str = "navigation",
    package_root: Path | None = None,
) -> Path:
    """Stage base + overlay side by side; return the staged overlay path.

    ``destination`` receives a copy of the shipped base configuration and
    its overlay (selected by ``base_config``) with ``simulation.platform``
    set to ``sim_platform``. With ``nav2_params_override`` the staged
    overlay's Nav2 ``params_file`` points at that file instead (the
    temporary parameter-variant mechanism of the e2e run). ``stage`` names
    the nav stage the overlay edits (``navigation`` for the navigation
    callers). ``package_root`` overrides the resource root (ament share by
    default) for synthetic layouts.
    """
    if base_config not in NAVIGATION_STAGING:
        raise ValueError(
            f"unknown navigation staging base {base_config!r}: expected one of {sorted(NAVIGATION_STAGING)}"
        )
    base_name, overlay_name = NAVIGATION_STAGING[base_config]
    destination.mkdir(parents=True, exist_ok=True)
    package = Path(package_root) if package_root is not None else robot_config_share_root()
    shutil.copy2(package / "config" / "robots" / base_name, destination / base_name)
    overlay_path = destination / overlay_name
    document = yaml.safe_load((package / "test" / "config" / overlay_name).read_text(encoding="utf-8"))
    stages = document.get("robot", {}).get("nav_stages")
    stage_section = stages.get(stage) if isinstance(stages, dict) else None
    if not isinstance(stage_section, dict):
        # fail loudly: a missing stage cannot carry the platform under test
        raise ValueError(
            f"the {overlay_name} overlay must define robot.nav_stages.{stage} (a mapping) "
            f"for staging; got {type(stage_section).__name__}"
        )
    simulation = stage_section.get("simulation")
    if not isinstance(simulation, dict):
        # fail loudly: the silent ``(stage.get("simulation") or {})`` write
        # would stage an overlay whose platform nobody set
        raise ValueError(
            f"the {overlay_name} overlay must define robot.nav_stages.{stage}.simulation (a mapping) "
            f"to carry the platform under test; got {type(simulation).__name__}"
        )
    simulation["platform"] = sim_platform
    if nav2_params_override:
        navigation = stage_section["navigation"]["nav2_bringup"]
        navigation["params_file"] = str(Path(nav2_params_override).expanduser().resolve())
    overlay_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return overlay_path


def load_staged_navigation_config(staged_overlay: Path, stage: str = "navigation"):
    """The resolved nav stage of the staged overlay (D10 assertions)."""
    from robot_config.loader import load_robot_config_dict

    return load_robot_config_dict(staged_overlay, nav_stage=stage, defer_interface_binding=True)
