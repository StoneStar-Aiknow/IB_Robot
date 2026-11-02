"""The no-attachment MuJoCo render regression (lekiwi-nav-grasp-sim-
platforms D2, review item 8).

`so101_single_arm` is the other shipped no-attachment profile on MuJoCo:
its rendered MJCF must stay byte-identical to the pre-change baseline.
This test lives in so101_robot, not robot_runtime: the profile is this
package's asset, and robot_runtime (the contract package) must not depend
on a robot package.

Golden provenance: `test/golden_mjcf/so101_single_arm_mjcf_baseline.xml`
was rendered by the BASELINE commit `2d534791`'s `render_robot_mjcf` over
the baseline's own profile and template, with the identity spawn and the
profile's peripherals as-is; the checkout-dependent absolute meshes
directory is normalized to {{MESHES_DIR}} on both sides (comments are
stripped in the comparison — see the lekiwi twin for the rationale).
"""

from __future__ import annotations

import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from robot_runtime.mujoco import render_robot_mjcf
from robot_runtime.profile import load_profile

_GOLDEN = Path(__file__).resolve().parent / "golden_mjcf" / "so101_single_arm_mjcf_baseline.xml"


def _strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def test_so101_single_arm_mjcf_matches_the_pre_change_baseline():
    """The no-attachment render is byte-identical to 2d534791: the template
    structure, arm bodies, meshes, actuators, the camera elements and every
    pose."""
    profile_path = Path(get_package_share_directory("so101_robot")) / "profiles" / "so101_single_arm.yaml"
    profile = load_profile(profile_path)
    rendered = Path(render_robot_mjcf(profile, peripherals=profile.get("peripherals") or [])).read_text(
        encoding="utf-8"
    )
    meshes_dir = str(Path(get_package_share_directory("so101_description")) / "meshes" / "lerobot" / "so101")
    assert meshes_dir in rendered
    rendered = rendered.replace(meshes_dir, "{{MESHES_DIR}}")
    golden = _GOLDEN.read_text(encoding="utf-8")
    assert _strip_comments(rendered) == _strip_comments(golden), (
        "the so101_single_arm MuJoCo render drifted from the 2d534791 "
        "baseline (structure, arm bodies, meshes, actuators, camera "
        "elements or poses changed)"
    )
