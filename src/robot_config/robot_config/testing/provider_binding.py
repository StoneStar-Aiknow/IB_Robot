"""Offline interface binding for shipped provider configurations in tests.

A provider configuration names logical interfaces (``camera.top.color``,
``joint.arm_stream``) that only resolve against a runtime interface
description. Live launch gets the description from the running runtime; a
unit test builds it here from the runtime profile on disk: the profile's
peripherals with the configuration's overrides applied (the same single merge
the runtime performs), the SDK simulation identity, and a fixture URDF with
the profile's joints and MoveIt frames in place of the rendered xacro. The
result checks interface binding only; it is not the description of a
calibrated physical arm.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from robot_config.interface_binding import bind_robot_interfaces


def fixture_urdf(profile: dict[str, Any]) -> str:
    """A minimal URDF with the profile's joints and MoveIt frames, wide limits."""
    motion = profile.get("motion") or {}
    links = {motion.get(key) for key in ("base_link", "ee_link", "shoulder_link")} - {None}
    gripper = {str(name) for name in profile.get("gripper_joints") or []}
    joints = "".join(
        f'<joint name="{name}" type="revolute"><limit lower="{0.0 if str(name) in gripper else -3.14}" '
        f'upper="{1.0 if str(name) in gripper else 3.14}" effort="10" velocity="10"/></joint>'
        for name in profile.get("joints") or []
    )
    return '<robot name="fixture">' + "".join(f'<link name="{link}"/>' for link in sorted(links)) + joints + "</robot>"


def offline_description(config: dict[str, Any], profile_path: str | Path) -> dict[str, Any]:
    """An SDK-simulated description built from the profile, for binding tests.

    The interfaces follow the profile, effective peripherals and enabled
    navigation.fast_lio fragment, just as runtime composition does; the model
    comes from :func:`fixture_urdf`, so the joint limits and frames are fixtures
    rather than the rendered xacro or a calibrated arm. It proves that a
    configuration's interface IDs and peripheral overrides resolve, not that
    the physical description matches.
    """
    from robot_runtime.interface_description import build_description
    from robot_runtime.peripherals import merge_peripherals
    from robot_runtime.profile import load_profile

    profile = load_profile(str(profile_path))
    runtime = config.get("runtime") or {}
    overrides = list(config.get("peripherals") or []) + list(runtime.get("_disabled_peripherals") or [])
    effective = {**profile, "peripherals": merge_peripherals(profile.get("peripherals"), overrides)}
    fast_lio = (config.get("navigation") or {}).get("fast_lio") or {}
    if fast_lio.get("enabled"):
        effective["fast_lio"] = {**(profile.get("fast_lio") or {}), **fast_lio}
    return build_description(
        effective,
        robot_description=fixture_urdf(effective),
        use_sim=True,
        sim_platform="sdk",
        profile_path=Path(profile_path),
    )


def bind_offline(config: dict[str, Any], profile_path: str | Path) -> dict[str, Any]:
    """Bind a deferred-loaded provider configuration against :func:`offline_description`."""
    return bind_robot_interfaces(copy.deepcopy(config), offline_description(config, profile_path))


def write_offline_snapshot(config_path: str | Path, profile_path: str | Path, directory: str | Path) -> Path:
    """Copy a shipped configuration with an offline description snapshot into ``directory``.

    Readers that load a configuration by path (the typed ``load_robot_config``,
    contract generators) take the description from ``runtime.interface_description``.
    The configuration and every ``base_config`` ancestor are copied as siblings,
    the description is written next to them, and the copy names it, so the
    shipped file stays untouched. Returns the path of the copied configuration.
    """
    import json

    import yaml

    from robot_config.loader import load_robot_config_dict

    source = Path(config_path)
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    current = source
    while True:
        raw = yaml.safe_load(current.read_text(encoding="utf-8"))
        (target_dir / current.name).write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
        base = (raw.get("robot") or {}).get("base_config")
        if not base:
            break
        current = current.parent / (base if str(base).endswith(".yaml") else f"{base}.yaml")
    deferred = load_robot_config_dict(source, defer_interface_binding=True)
    description_path = target_dir / f"{source.stem}.description.json"
    description_path.write_text(json.dumps(offline_description(deferred, profile_path)), encoding="utf-8")
    copied = target_dir / source.name
    data = yaml.safe_load(copied.read_text(encoding="utf-8"))
    data["robot"].setdefault("runtime", {})["interface_description"] = description_path.name
    copied.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return copied
