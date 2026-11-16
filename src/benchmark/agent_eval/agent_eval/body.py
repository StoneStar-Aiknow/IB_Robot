"""The robot body as the evaluation runner sees it, built from the robot configuration.

Everything robot-specific comes from configuration, never from names in this code:

* arm sides and motion groups: ``robot.embodied.execution.arm_groups`` ({side: group});
* gripper joint per group: ``robot.task_dispatch.groups.<group>.gripper_joint``;
* wrist joint per group: the last joint of the runtime profile's ``motion_groups.<group>.joints``
  (profile from ``robot.runtime.provider`` / ``profile``: a file path, or a profile name shipped
  as ``share/<provider>/profiles/<profile>.yaml``, as the runtime launch resolves it);
* end-effector pose topic: the profile's ``motion_groups.<group>.ee_pose_topic``
  (default ``/robot_status/<group>/ee_pose``);
* gripper approach presets and axis: ``robot.embodied.execution.gripper_approach``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from .episode_context import BodyFacts


class BodyConfigError(ValueError):
    """The robot configuration or runtime profile lacks a body fact (named in the message)."""


@dataclass(frozen=True)
class ArmBody:
    side: str
    group: str
    gripper_joint: str
    wrist_joint: str
    ee_pose_topic: str


@dataclass(frozen=True)
class BodyDescription:
    robot_config: str
    robot_config_path: str
    runtime_provider: str
    runtime_profile: str
    runtime_profile_path: str
    arms: tuple[ArmBody, ...]
    approach_axis: str = "z"
    approach_presets: Mapping[str, tuple[float, float, float, float]] = field(default_factory=dict)
    # The robot config file and every base_config it inherits from (resolution order).
    robot_config_sources: tuple[str, ...] = ()

    @property
    def sides(self) -> tuple[str, ...]:
        return tuple(arm.side for arm in self.arms)

    def arm(self, side: str) -> ArmBody:
        return next(arm for arm in self.arms if arm.side == side)

    def facts(self) -> BodyFacts:
        """The body facts the Agent-side episode context consumes."""
        return BodyFacts(
            sides=self.sides, approach_presets=dict(self.approach_presets), approach_axis=self.approach_axis
        )


def _get(data: Any, path: Sequence[str], where: str) -> Any:
    node = data
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise BodyConfigError(f"{where}: missing {'.'.join(path)}")
        node = node[key]
    return node


def resolve_profile_path(provider: str, profile: str, search_dirs: Sequence[str | Path] = ()) -> Path:
    """A profile file path as is; a name: ``<dir>/<name>.yaml`` from ``search_dirs``, then ``share/<provider>/profiles``."""
    given = Path(profile).expanduser()
    if given.is_absolute() or len(given.parts) > 1:
        if given.is_file():
            return given.resolve()
        raise BodyConfigError(f"runtime profile {profile} is not a file")
    file_name = profile if profile.endswith(".yaml") else f"{profile}.yaml"
    candidates = [Path(d).expanduser() / file_name for d in search_dirs]
    with suppress(ImportError):
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        with suppress(PackageNotFoundError):
            candidates.append(Path(get_package_share_directory(provider)) / "profiles" / file_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise BodyConfigError(f"runtime profile {provider}/{profile} not found (searched {[str(c) for c in candidates]})")


def body_from_documents(
    robot: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    robot_config: str = "",
    robot_config_path: str = "",
    profile_path: str = "",
) -> BodyDescription:
    """Build the body from a parsed robot config (``robot:`` mapping) and runtime profile."""
    where = robot_config or "robot config"
    execution = _get(robot, ("embodied", "execution"), where)
    arm_groups = _get(execution, ("arm_groups",), where)
    if not isinstance(arm_groups, Mapping) or not arm_groups:
        raise BodyConfigError(f"{where}: embodied.execution.arm_groups must map arm sides to motion groups")
    dispatch_groups = _get(robot, ("task_dispatch", "groups"), where)
    motion_groups = _get(profile, ("motion_groups",), profile_path or "runtime profile")
    arms = []
    for side, group in arm_groups.items():
        gripper = _get(dispatch_groups, (str(group), "gripper_joint"), f"{where} task_dispatch.groups")
        motion = _get(motion_groups, (str(group),), profile_path or "runtime profile")
        joints = motion.get("joints") if isinstance(motion, Mapping) else None
        if not joints:
            raise BodyConfigError(f"{profile_path or 'runtime profile'}: motion_groups.{group}.joints is empty")
        arms.append(
            ArmBody(
                side=str(side),
                group=str(group),
                gripper_joint=str(gripper),
                wrist_joint=str(joints[-1]),
                ee_pose_topic=str(motion.get("ee_pose_topic") or f"/robot_status/{group}/ee_pose"),
            )
        )
    approach = execution.get("gripper_approach") or {}
    presets = {}
    for name, spec in (approach.get("presets") or {}).items():
        orientation = _get(spec, ("orientation",), f"{where} gripper_approach.presets.{name}")
        presets[str(name)] = tuple(float(orientation[axis]) for axis in ("x", "y", "z", "w"))
    runtime = robot.get("runtime") or {}
    return BodyDescription(
        robot_config=robot_config,
        robot_config_path=robot_config_path,
        runtime_provider=str(runtime.get("provider") or ""),
        runtime_profile=str(runtime.get("profile") or ""),
        runtime_profile_path=profile_path,
        arms=tuple(arms),
        approach_axis=str(approach.get("axis") or "z"),
        approach_presets=presets,
    )


def load_robot_document(
    robot_config: str | None = None, robot_config_path: str | Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """The robot config file and its ``robot:`` mapping with ``base_config`` inheritance resolved.

    Resolution is robot_config's own loader, so the body is the same SSOT the launch uses.
    """
    path, robot, _sources = load_robot_document_with_sources(robot_config, robot_config_path)
    return path, robot


def load_robot_document_with_sources(
    robot_config: str | None = None, robot_config_path: str | Path | None = None
) -> tuple[Path, dict[str, Any], tuple[Path, ...]]:
    from robot_config.config_path import resolve_robot_config_path

    # The raw section with base_config resolved, not load_robot_config_dict: the suite writes
    # this mapping back as the run-local robot config the stack launches, and the full loader
    # adds framework-generated keys (_config_path, _config_sources and, when it filters
    # peripherals, runtime._disabled_peripherals, which the launch's loader then rejects).
    # robot_config has no public raw loader that also returns the source chain.
    from robot_config.loader import _load_robot_section_with_sources

    if robot_config_path is None and robot_config and Path(robot_config).expanduser().is_file():
        robot_config, robot_config_path = None, Path(robot_config).expanduser()
    path = resolve_robot_config_path(config_name=robot_config, config_path=robot_config_path)
    try:
        resolved, robot, sources = _load_robot_section_with_sources(path)
    except ValueError as exc:
        raise BodyConfigError(str(exc)) from exc
    if not isinstance(robot, Mapping):
        raise BodyConfigError(f"{path}: no robot: mapping")
    return Path(resolved), dict(robot), tuple(Path(s) for s in sources)


def load_body(
    robot_config: str | None = None,
    *,
    robot_config_path: str | Path | None = None,
    profile_path: str | Path | None = None,
    profile_search_dirs: Sequence[str | Path] = (),
) -> BodyDescription:
    """Body description for a robot config name (or file), with its runtime profile."""
    path, robot, sources = load_robot_document_with_sources(robot_config, robot_config_path)
    if profile_path is None:
        runtime = robot.get("runtime") or {}
        provider, profile = runtime.get("provider"), runtime.get("profile")
        if not provider or not profile:
            raise BodyConfigError(f"{path}: robot.runtime.provider and robot.runtime.profile are required")
        profile_path = resolve_profile_path(str(provider), str(profile), profile_search_dirs)
    with open(profile_path, encoding="utf-8") as fh:
        profile_doc = yaml.safe_load(fh) or {}
    body = body_from_documents(
        robot,
        profile_doc,
        robot_config=str(robot.get("name") or robot_config or path.stem),
        robot_config_path=str(path),
        profile_path=str(Path(profile_path).resolve()),
    )
    return replace(body, robot_config_sources=tuple(str(p) for p in sources))
