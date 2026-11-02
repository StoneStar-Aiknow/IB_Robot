"""The hardware URDF render regression (lekiwi-nav-grasp-sim-platforms D8,
review item 4).

The change renamed the arm's mid-chain root link `world` -> `arm_root` on
every LeKiwi render (the reserved frame name breaks sdformat) and threaded
it through `base_to_arm`. That rename is the ONLY intended hardware-visible
render change: this test renders the current
`lekiwi_assembled.urdf.xacro` with the hardware arguments, normalizes the
rename back, and compares BYTE-IDENTICAL against a golden generated from
the baseline commit `2d534791` — pinning the joint set, the arm chain, the
`base_to_arm` pose, the hardware ros2_control plugin parameters and every
mesh reference.

Golden provenance (regenerate with the baseline checkout):
`test/golden/lekiwi_assembled_hardware_baseline.urdf.xml` was rendered by
`xacro.process_file` on `2d534791`'s `lekiwi_assembled.urdf.xacro` (whose
`$(find so101_description)` resolved to the baseline install, so the
baseline `so101_base.xacro` is baked in) with the fixed arguments
{use_sim: false, simulated: false, port: /dev/ttyACM0, calib_file:
/calib/so101_follower_calibrate.json}; the xacro autogeneration header
(which embeds the absolute source path) is stripped on both sides. The
calib path is a placeholder — the xacro embeds the string, it does not read
the file — and every mesh is a package:// URI, so the golden is
checkout-independent.
"""

from __future__ import annotations

from pathlib import Path

import xacro

_PACKAGE = Path(__file__).resolve().parents[1]
_GOLDEN = Path(__file__).resolve().parent / "golden" / "lekiwi_assembled_hardware_baseline.urdf.xml"

#: The fixed render arguments (see the module docstring's provenance).
_RENDER_ARGS = {
    "use_sim": "false",
    "simulated": "false",
    "port": "/dev/ttyACM0",
    "calib_file": "/calib/so101_follower_calibrate.json",
}


def _render_hardware() -> str:
    document = xacro.process_file(str(_PACKAGE / "urdf" / "lekiwi_assembled.urdf.xacro"), mappings=_RENDER_ARGS)
    text = document.toxml()
    return text[text.index("<robot") :]


def _strip_comments(text: str) -> str:
    """Remove XML comments: the change documents the rename inside a comment
    ("arm base" -> "D8"), and comments are not hardware-visible."""
    import re

    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def test_the_hardware_render_matches_the_baseline_except_the_root_link_rename():
    """Everything but `world` -> `arm_root` is byte-identical to 2d534791."""
    rendered = _render_hardware()
    # the rename is live on every render (D8): no reserved frame name remains
    assert 'name="arm_root"' in rendered and 'name="world"' not in rendered
    # with the rename normalized back (comments stripped — the change
    # documents the rename inside one), the render equals the baseline golden
    normalized = _strip_comments(rendered.replace("arm_root", "world"))
    golden = _strip_comments(_GOLDEN.read_text(encoding="utf-8"))
    assert normalized == golden, (
        "the hardware render drifted from the 2d534791 baseline beyond the "
        "world->arm_root rename (joint set, arm chain, base_to_arm pose, "
        "hardware plugin parameters or mesh references changed)"
    )


def test_the_baseline_golden_carries_the_expected_hardware_surface():
    """Sanity pins on the golden itself: the arm chain, the base_to_arm joint
    and the hardware plugin the runtime launches on the real robot."""
    golden = _GOLDEN.read_text(encoding="utf-8")
    # the arm's mid-chain root and its fixed joint onto base_link
    assert 'name="world"' in golden
    assert 'joint name="base_to_arm" type="fixed"' in golden
    # the ros2_control hardware plugin is the real LeKiwi bus adapter,
    # not a simulator
    assert "lekiwi_hardware/LeKiwiSystemHardware" in golden
    assert "GazeboSimSystem" not in golden and "MujocoSystemInterface" not in golden
    # the nine LeKiwi joints are all present
    for joint in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
        assert f'name="{joint}"' in golden
