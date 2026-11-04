"""The no-attachment MuJoCo render regression (lekiwi-nav-grasp-sim-
platforms D2, review item 8).

`lekiwi_lidar` is the shipped no-attachment profile on MuJoCo: its rendered
MJCF must stay byte-identical to the pre-change baseline apart from XML
comments (the base template's header comment legitimately changed to
document that the template is now also the composition host for
`lekiwi_nav_grasp`'s arm attachment — D2). This test lives in lekiwi_robot,
not robot_runtime: the profile is this package's asset, and robot_runtime
(the contract package) must not depend on a robot package.

Golden provenance: `test/golden_mjcf/lekiwi_lidar_mjcf_baseline.xml` was
rendered by the BASELINE commit `2d534791`'s `render_robot_mjcf` over the
baseline's own profile and template, with the identity spawn and the
profile's peripherals as-is (the baseline has no per-platform exclusion);
the checkout-dependent absolute meshes directory is normalized to
{{MESHES_DIR}} on both sides. The generation script and the exact
environment are recorded in the change's acceptance notes.
"""

from __future__ import annotations

import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from robot_runtime.mujoco import render_robot_mjcf
from robot_runtime.profile import load_profile

_GOLDEN = Path(__file__).resolve().parent / "golden_mjcf" / "lekiwi_lidar_mjcf_baseline.xml"


def _strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def test_lekiwi_lidar_mjcf_matches_the_pre_change_baseline():
    """The no-attachment render is byte-identical to 2d534791 (comments
    aside): the template structure, wheels, meshes, actuators, the MID-360
    site/sensor/plugin elements and every pose."""
    profile_path = Path(get_package_share_directory("lekiwi_robot")) / "profiles" / "lekiwi_lidar.yaml"
    profile = load_profile(profile_path)
    rendered = Path(render_robot_mjcf(profile, peripherals=profile.get("peripherals") or [])).read_text(
        encoding="utf-8"
    )
    # normalize the checkout-dependent absolute meshes directory
    meshes_dir = str(Path(get_package_share_directory("lekiwi_description")) / "meshes" / "mujoco")
    assert meshes_dir in rendered
    rendered = rendered.replace(meshes_dir, "{{MESHES_DIR}}")
    golden = _GOLDEN.read_text(encoding="utf-8")
    assert _strip_comments(rendered) == _strip_comments(golden), (
        "the lekiwi_lidar MuJoCo render drifted from the 2d534791 baseline "
        "beyond the template header comment (structure, wheels, meshes, "
        "actuators, MID-360 elements or poses changed)"
    )
    # the one comment difference is the documented host-note, nothing else
    assert "HOST of the composed model" in Path(get_package_share_directory("lekiwi_description")).joinpath(
        "mujoco", "lekiwi_base.xml.template"
    ).read_text(encoding="utf-8")
