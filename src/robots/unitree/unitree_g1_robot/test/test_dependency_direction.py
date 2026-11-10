"""Dependency direction: the robot package never writes the agent catalog.

The offline generator lives in ``tools/g1_preset_presentation.py`` because the
robot package publishes the profile and ``skill_catalog`` owns its manifests.
Keeping the bridge out of both packages is what prevents a robot-package ->
agent-directory reverse dependency; this test fails if the generator (or any
catalog compiler import) comes back into the published package.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "unitree_g1_robot"


def test_robot_package_has_no_catalog_bridge():
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "skill_catalog" in text or "preset_presentation" in text:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert offenders == []


def test_the_offline_generator_is_not_installed_with_the_robot_package():
    assert not (PACKAGE_ROOT / "preset_presentation.py").exists()
