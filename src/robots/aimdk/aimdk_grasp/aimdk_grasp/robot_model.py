"""The X2 Ultra planning model and its cuRobo collision configuration.

``config/x2_ultra`` holds the cuRobo robot configuration; its URDF is generated
from ``aimdk_description`` (:mod:`aimdk_grasp.planning_model`), with the left arm
narrowed to the ``arm_limits`` of the robot configuration ``robot.yml`` names. Every collision
cover is bound to the description files it was fitted or checked on, and is
refused if they changed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import yaml

from aimdk_grasp.hand import hand_links
from aimdk_grasp.joints import LEFT_ARM
from aimdk_grasp.planning_model import build_planning_model, description_share

#: Wrist links the hand's reference exclusions may also name.
HAND_MOUNT = ("left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link")


def config_directory() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("aimdk_grasp")) / "config" / "x2_ultra"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs_digest(pairs) -> str:
    """Digest of contact exclusion pairs, order-free."""
    canonical = sorted(sorted(pair) for pair in pairs)
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def _sphere_digest(spheres) -> str:
    return hashlib.sha256(json.dumps(spheres, sort_keys=True).encode()).hexdigest()


def left_arm_limits(robot: Mapping) -> dict[str, tuple[float, float]]:
    """The left-arm entries of a loaded robot configuration's ``arm_limits``.

    Every left-arm joint must be present with a finite, increasing range; the
    other arm's entries are not read.
    """
    limits = robot.get("arm_limits")
    if not isinstance(limits, Mapping):
        raise ValueError("robot configuration has no arm_limits mapping")
    result = {}
    for name in LEFT_ARM:
        value = limits.get(name)
        if (
            not isinstance(value, list | tuple)
            or len(value) != 2
            or not all(isinstance(bound, int | float) and math.isfinite(bound) for bound in value)
            or not value[0] < value[1]
        ):
            raise ValueError(f"arm_limits.{name} must be a finite increasing [lower, upper] pair, got {value!r}")
        result[name] = (float(value[0]), float(value[1]))
    return result


def planning_model_options(options: Mapping) -> dict:
    """``robot.yml``'s ``planning_model`` block as :func:`build_planning_model` keywords.

    ``arm_limits_config`` names the robot configuration whose ``arm_limits``
    narrow the left arm; it is resolved like any other robot configuration name.
    """
    from robot_config.config_path import resolve_robot_config_path
    from robot_config.loader import load_robot_config_dict

    options = dict(options)
    path = resolve_robot_config_path(config_name=str(options.pop("arm_limits_config")))
    options["arm_limits"] = left_arm_limits(load_robot_config_dict(str(path), defer_interface_binding=True))
    return options


@lru_cache(maxsize=4)
def _load(directory: str) -> tuple[dict, bytes]:
    config = yaml.safe_load((Path(directory) / "robot.yml").read_text(encoding="utf-8"))
    tree = build_planning_model(**planning_model_options(config["planning_model"]))
    return config, ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)


def planning_model(directory: Path | None = None) -> ET.Element:
    """The generated planning URDF the configuration in ``directory`` was built for."""
    _, urdf = _load(str(directory or config_directory()))
    return ET.fromstring(urdf)


def planning_urdf(directory: Path | None = None) -> Path:
    """The planning URDF as a file (cuRobo loads URDFs by path), named by its content."""
    _, urdf = _load(str(directory or config_directory()))
    path = Path(tempfile.gettempdir()) / "aimdk_grasp" / f"x2_ultra_{hashlib.sha256(urdf).hexdigest()[:16]}.urdf"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, partial_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        partial = Path(partial_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                os.fchmod(stream.fileno(), 0o644)
                stream.write(urdf)
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def _ignore(kinematics: dict, a: str, b: str) -> None:
    for x, y in ((a, b), (b, a)):
        ignored = kinematics["self_collision_ignore"].setdefault(x, [])
        if y not in ignored:
            ignored.append(y)


def _planning_description(config: Mapping) -> Path:
    """The description the planning model is generated from.

    ``robot.yml``'s ``planning_model`` block may name one (it is a
    :func:`build_planning_model` keyword); the cover checks read the meshes
    from the same directory, so a model generated from a pinned description
    is also checked against that description. The default is the installed
    ``aimdk_description`` share.
    """
    description = (config.get("planning_model") or {}).get("description")
    return Path(description) if description else description_share()


def collision_robot_config(
    directory: Path | None = None,
    *,
    reference_hand_collisions: bool = True,
    torso_merge_radius_m: float | None = None,
    torso_merge_protected_spheres=None,
    hand_merge_radius_m: float | None = None,
) -> dict:
    """The cuRobo robot configuration with the verified hand and torso covers.

    ``hand_merge_radius_m`` merges each hand link's spheres (merge_sphere_cover:
    every merged ball contains the balls it replaces, so the cover only grows).
    For an optimizer model; checks keep the original cover.
    """
    directory = Path(directory or config_directory())
    loaded, _ = _load(str(directory))
    config = copy.deepcopy(loaded)
    model = planning_model(directory)
    description = _planning_description(loaded)
    kinematics = config["robot_cfg"]["kinematics"]
    kinematics["urdf_path"] = str(planning_urdf(directory))
    kinematics["asset_root_path"] = str(description)
    hand = set(hand_links(model))

    reference = _read_json(directory / "hand_collision_reference.json")
    if _sphere_digest(kinematics["collision_spheres"]) != reference["collision_spheres_sha256"]:
        raise ValueError("Reference exclusions do not match source collision spheres")
    if reference_hand_collisions:
        for a, b in reference["pairs"]:
            if not ((a in hand and b in hand | set(HAND_MOUNT)) or (b in hand and a in HAND_MOUNT)):
                raise ValueError("Reference exclusion outside hand/mount scope")
            _ignore(kinematics, a, b)

    cover = _read_json(directory / "hand_collision_cover.json")
    covered = {link: item["spheres"] for link, item in cover["links"].items()}
    if _sphere_digest(covered) != cover["collision_spheres_sha256"]:
        raise ValueError("Hand collision cover sphere hash does not match")
    for link, item in cover["links"].items():
        if link not in kinematics["collision_spheres"] or link not in hand:
            raise ValueError(f"Hand collision cover has unknown link: {link}")
        official = item["official_mesh"]
        mesh = description / official["path"]
        # The mesh bytes alone do not place them: the URDF's visual origin and
        # mesh scale move and stretch the same STL inside the link, so the
        # cover is checked against the placement it was fitted at too.
        placed = []
        for visual in model.findall(f"link[@name='{link}']/visual"):
            mesh_element = visual.find("geometry/mesh")
            origin = visual.find("origin")
            placed.append(
                (
                    None if mesh_element is None else Path(mesh_element.get("filename")),
                    None if origin is None else dict(origin.attrib),
                    None if mesh_element is None else mesh_element.get("scale"),
                )
            )
        if placed != [(mesh, official["origin"], official.get("scale"))] or _sha256(mesh) != official["sha256"]:
            raise ValueError(f"Hand collision cover mesh does not match the description: {link}")
        kinematics["collision_spheres"][link] = item["spheres"]
    if reference_hand_collisions:
        # The pairs come from the AimDK SDK hand MJCF (contact_source), which is
        # not vendored here; their digest keeps an edited cover from silently
        # turning off hand self-collision pairs.
        if _pairs_digest(cover["contact_pairs"]) != cover["contact_pairs_sha256"]:
            raise ValueError("Hand contact exclusions do not match their recorded digest; regenerate the cover")
        for a, b in cover["contact_pairs"]:
            if not {a, b} <= hand:
                raise ValueError("Hand contact exclusion outside the hand")
            _ignore(kinematics, a, b)

    torso = _read_json(directory / "torso_collision_cover.json")
    collisions = model.findall("link[@name='torso_link']/collision")
    if len(collisions) != len(torso["sources"]):
        raise ValueError("Torso cover does not match collision meshes; regenerate it")
    for collision, source in zip(collisions, torso["sources"], strict=True):
        geometry = collision.find("geometry/mesh")
        mesh = Path(geometry.get("filename"))
        scale = [float(value) for value in geometry.get("scale", "1 1 1").split()]
        if len(scale) != 3:
            raise ValueError("Torso cover does not match collision meshes; regenerate it")
        if (
            mesh != description / source["path"]
            or dict(collision.find("origin").attrib) != source["origin"]
            or scale != source.get("scale")
        ):
            raise ValueError("Torso cover does not match collision meshes; regenerate it")
        if _sha256(mesh) != source["sha256"]:
            raise ValueError("Torso cover mesh hash does not match; regenerate it")
    kinematics["collision_spheres"]["torso_link"] = torso["spheres"]
    if torso_merge_radius_m is not None:
        from aimdk_grasp.collision_spheres import merge_sphere_cover

        kinematics["collision_spheres"]["torso_link"] = merge_sphere_cover(
            torso["spheres"], torso_merge_radius_m, protected_spheres=torso_merge_protected_spheres
        )
    if hand_merge_radius_m is not None:
        from aimdk_grasp.collision_spheres import merge_sphere_cover

        for link in hand & set(kinematics["collision_spheres"]):
            kinematics["collision_spheres"][link] = merge_sphere_cover(
                kinematics["collision_spheres"][link], hand_merge_radius_m
            )
    for a, b in torso["assembly_pairs"]:
        _ignore(kinematics, a, b)
    return config
