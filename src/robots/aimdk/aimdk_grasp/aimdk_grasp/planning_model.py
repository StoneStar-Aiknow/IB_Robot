"""Build the X2 grasp planning model of one arm from the official X2 description.

The model combines two official AgiBot URDFs from ``aimdk_description``:

* the X2 Ultra upper body (``x2_ultra.urdf``, v1.3.0), whose kinematics match the
  motion-control unit's own model; legs and the IMU, camera and lidar frames are
  dropped because upper-body planning does not use them;
* the left OmniHand of ``x2_ultra_omnihand_v1.4.0.urdf``, mounted on the left
  wrist at the official ``L_palm_joint``. Both wrist roll links take their
  geometry from that model too: the v1.3.0 ones include the dummy hand fitted
  when no OmniHand is mounted. The joints are identical in both versions.

Three conventions of the existing grasp pipeline are kept so its collision
spheres, gripper description and joint commands stay valid:

* **Hand joints are mounted aliases** ``R_<part>_joint`` / links ``R_<part>``. An
  alias is the native position times :data:`HAND_ALIAS_SIGNS`: joints that turn
  about the link's x or z axis are mirrored (axis and limits negated), the
  flexion joints about y are not. The distal joints the official URDF couples
  by linear ``mimic`` become independent (:data:`PIPELINE_COUPLED`): the
  pipeline drives them from the measured, non-linear coupling instead. The
  thumb PIP keeps its ``mimic``, re-expressed between the aliases.
* **``R_palm`` keeps the pipeline's tool-frame orientation.** The official palm
  frame is rotated about 90 degrees about the wrist axis relative to it; the
  palm is re-expressed in the pipeline's orientation, so palm meshes and finger
  joints keep their place on the palm.
* **The hand sits on the AimDK SDK mount** (``palm_mount="legacy"``,
  ``thumb_mount="legacy"``), which the operator confirmed on the robot: the palm
  3 mm closer to the wrist than the official model puts it, the fingers where
  the official model has them on the palm, and the thumb at the SDK's own
  joint. The hand is then the SDK hand the grasp model's gripper description
  and the hand collision spheres were made from; with the official palm and the
  SDK thumb it was neither (the thumb 3.2 mm off on the palm). The official
  URDF's thumb roll axis is about 75 degrees away from the SDK's and does not
  match the hardware; ``"official"`` reproduces the published model for
  comparison.

The left-arm joint limits are intersected with the limits the motion-control
unit enforces, so a plan never commands what the platform would clamp.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimdk_grasp.joints import DIP_FOLLOWERS, HAND_ALIAS_SIGNS, check_side

BODY_URDF = "urdf/x2_ultra.urdf"
HAND_URDF = "urdf/x2_ultra_omnihand_v1.4.0.urdf"
BODY_MESHES = "meshes/v1.3.0"
HAND_MESHES = "meshes/v1.4.0"

#: Body links upper-body planning does not use: legs, and sensor frames whose
#: poses come from TF at run time.
DROPPED_LINK_KEYWORDS = ("hip", "knee", "ankle", "imu", "rgbd_", "stereo_", "rgb_head", "lidar")

#: Distal joints the pipeline drives itself; their official ``mimic`` is dropped.
PIPELINE_COUPLED = tuple(name[2:].removesuffix("_joint") for name in DIP_FOLLOWERS)

#: Pipeline palm tool frame relative to ``left_wrist_roll_link`` (orientation only).
PALM_TOOL_RPY = (np.pi, 0.0, 0.0)
#: AimDK SDK palm position relative to ``left_wrist_roll_link`` (the official model has (0, -0.001, -0.049)).
LEGACY_PALM_XYZ = (0.0, 0.0, -0.046)
#: AimDK SDK (agibot_x2_hand) finger base joints relative to ``left_wrist_roll_link``, by the link they carry.
#: The official model mounts its fingers on the palm 0.53 degrees about the wrist axis away from these.
_SDK_FINGER_RPY = (3.1399999994215753, 0.0, -1.5707963267948966)
LEGACY_FINGER_MOUNTS = {
    "index_abad": ((0.0267994, 0.0, -0.140591), _SDK_FINGER_RPY),
    "middle_pip": ((0.00776823, 0.00201837, -0.160136), _SDK_FINGER_RPY),
    "ring_abad": ((-0.0112006, 0.00151, -0.14053), _SDK_FINGER_RPY),
    "pinky_abad": ((-0.0301926, 0.0, -0.1355), _SDK_FINGER_RPY),
}
#: AimDK SDK thumb roll joint relative to ``left_wrist_roll_link``.
LEGACY_THUMB_ROLL_XYZ = (0.0184452, 0.0, -0.0935804)
LEGACY_THUMB_ROLL_RPY = _SDK_FINGER_RPY
THUMB_MOUNTS = PALM_MOUNTS = ("legacy", "official")
#: Body links whose v1.3.0 geometry is replaced by the OmniHand model's.
OMNIHAND_WRISTS = ("left_wrist_roll_link", "right_wrist_roll_link")


def _matrix(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def _origin_matrix(element: ET.Element) -> np.ndarray:
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
    return _matrix(xyz, rpy)


def _set_origin(element: ET.Element, matrix: np.ndarray) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
    xyz = matrix[:3, 3]
    rpy = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")
    origin.set("xyz", " ".join(repr(float(value)) for value in xyz))
    origin.set("rpy", " ".join(repr(float(value)) for value in rpy))


def _alias(name: str) -> str:
    return "R_" + name[2:] if name.startswith("L_") else name


def _rehome_meshes(element: ET.Element, mesh_root: Path) -> None:
    for mesh in element.iter("mesh"):
        mesh.set("filename", str(mesh_root / Path(mesh.get("filename")).name))


def _body(description: Path) -> ET.Element:
    robot = ET.parse(description / BODY_URDF).getroot()
    dropped = {
        link.get("name")
        for link in robot.findall("link")
        if any(keyword in link.get("name") for keyword in DROPPED_LINK_KEYWORDS)
    }
    for link in robot.findall("link"):
        if link.get("name") in dropped:
            robot.remove(link)
    for joint in robot.findall("joint"):
        if joint.find("child").get("link") in dropped:
            robot.remove(joint)
    _rehome_meshes(robot, description / BODY_MESHES)
    return robot


def _mount_omnihand_wrists(robot: ET.Element, source: ET.Element, description: Path) -> None:
    for name in OMNIHAND_WRISTS:
        link = robot.find(f"link[@name='{name}']")
        for element in [*link.findall("visual"), *link.findall("collision")]:
            link.remove(element)
        for element in source.find(f"link[@name='{name}']"):
            if element.tag in ("visual", "collision"):
                geometry = copy.deepcopy(element)
                _rehome_meshes(geometry, description / HAND_MESHES)
                link.append(geometry)


def _hand(
    source: ET.Element, description: Path, thumb_mount: str, palm_mount: str = "legacy"
) -> tuple[list[ET.Element], list[ET.Element]]:
    """The left OmniHand as alias links and joints, mounted on ``left_wrist_roll_link``."""
    links = [copy.deepcopy(link) for link in source.findall("link") if link.get("name").startswith("L_")]
    joints = [
        copy.deepcopy(joint) for joint in source.findall("joint") if joint.find("child").get("link").startswith("L_")
    ]
    palm_joint = next(joint for joint in joints if joint.get("name") == "L_palm_joint")
    if palm_joint.find("parent").get("link") != "left_wrist_roll_link":
        raise ValueError("official OmniHand palm is not mounted on left_wrist_roll_link")

    # Re-express the palm in the pipeline's tool-frame orientation without moving
    # anything attached to it: the palm frame turns by C, so its geometry and child
    # joints are expressed through C^-1.
    official_palm = _origin_matrix(palm_joint)
    tool = official_palm.copy()
    tool[:3, :3] = _matrix(rpy=PALM_TOOL_RPY)[:3, :3]
    correction = np.linalg.inv(tool) @ official_palm
    _set_origin(palm_joint, tool)
    palm = next(link for link in links if link.get("name") == "L_palm")
    for element in [*palm.findall("visual"), *palm.findall("collision"), *palm.findall("inertial")]:
        _set_origin(element, correction @ _origin_matrix(element))
    for joint in joints:
        if joint.find("parent").get("link") == "L_palm":
            _set_origin(joint, correction @ _origin_matrix(joint))
    if palm_mount == "legacy":
        tool[:3, 3] = LEGACY_PALM_XYZ
        _set_origin(palm_joint, tool)
        for part, (xyz, rpy) in LEGACY_FINGER_MOUNTS.items():
            _mount_on_wrist(joints, f"L_{part}", tool, _matrix(xyz, rpy))
    elif palm_mount != "official":
        raise ValueError(f"palm_mount must be one of {PALM_MOUNTS}, got {palm_mount!r}")

    if thumb_mount == "legacy":
        _mount_on_wrist(joints, "L_thumb_roll", tool, _matrix(LEGACY_THUMB_ROLL_XYZ, LEGACY_THUMB_ROLL_RPY))
    elif thumb_mount != "official":
        raise ValueError(f"thumb_mount must be one of {THUMB_MOUNTS}, got {thumb_mount!r}")

    for link in links:
        link.set("name", _alias(link.get("name")))
    for joint in joints:
        joint.set("name", _alias(joint.get("name")))
        for end in ("parent", "child"):
            joint.find(end).set("link", _alias(joint.find(end).get("link")))
        part = joint.get("name")[2:].removesuffix("_joint")
        if joint.get("type") != "revolute":
            continue
        sign = HAND_ALIAS_SIGNS[part]
        mimic = joint.find("mimic")
        if mimic is not None and part in PIPELINE_COUPLED:
            joint.remove(mimic)
        elif mimic is not None:
            leader = mimic.get("joint")[2:].removesuffix("_joint")
            mimic.set("joint", _alias(mimic.get("joint")))
            mimic.set("multiplier", repr(sign * HAND_ALIAS_SIGNS[leader] * float(mimic.get("multiplier", "1"))))
            mimic.set("offset", repr(sign * float(mimic.get("offset", "0")) + 0.0))
        if sign < 0:
            axis = joint.find("axis")
            axis.set("xyz", " ".join(repr(-float(value) + 0.0) for value in axis.get("xyz").split()))
            limit = joint.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            limit.set("lower", repr(-upper + 0.0))
            limit.set("upper", repr(-lower + 0.0))
    for link in links:
        _rehome_meshes(link, description / HAND_MESHES)
    return links, joints


def _mirror_hand(links: list[ET.Element], joints: list[ET.Element]) -> tuple[list[ET.Element], list[ET.Element]]:
    """The aliased left hand mirrored onto ``right_wrist_roll_link``, as the right hand.

    The X2's right arm mirrors the left one about the torso's sagittal plane,
    and the right OmniHand mirrors the left one. Every link frame is mirrored
    by a reflection ``K`` of its own: ``y`` for the wrist and the fingers (the
    official right hand's convention), ``x`` for ``R_palm``. With ``x`` the
    mirrored palm tool frame keeps the gripper description's ``base_rotation``
    valid: a grasp frame ``G = P inv(B)`` then mirrors to ``M G diag(1, -1, 1)``,
    a symmetry of the GraspGen-X sweep volumes, so the right hand uses the left
    description unchanged. ``y`` would tilt the grasp frame instead.

    A joint origin ``T`` from parent ``p`` to child ``c`` becomes
    ``K_p T K_c``; a revolute axis ``a`` becomes ``-K_c a`` with the same angle,
    so the aliases keep their values and limits: a right-hand alias position is
    the native right-hand position (the left alias of the same physical shape).
    Geometry keeps the left meshes, reflected by a mesh scale; inertia follows.
    """
    flip_y, flip_x = np.diag([1.0, -1.0, 1.0, 1.0]), np.diag([-1.0, 1.0, 1.0, 1.0])

    def reflection(link: str) -> np.ndarray:
        return flip_x if link == "R_palm" else flip_y

    links, joints = copy.deepcopy(links), copy.deepcopy(joints)
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        if parent.get("link") == "left_wrist_roll_link":
            parent.set("link", "right_wrist_roll_link")
        elif not parent.get("link").startswith("R_"):
            raise ValueError(f"hand joint {joint.get('name')} hangs off {parent.get('link')}")
        k_parent, k_child = reflection(parent.get("link")), reflection(child.get("link"))
        _set_origin(joint, k_parent @ _origin_matrix(joint) @ k_child)
        axis = joint.find("axis")
        if axis is not None:
            mirrored = -(k_child[:3, :3] @ np.array([float(value) for value in axis.get("xyz").split()]))
            axis.set("xyz", " ".join(repr(float(value) + 0.0) for value in mirrored))
    for link in links:
        k_link = reflection(link.get("name"))
        for element in [*link.findall("visual"), *link.findall("collision")]:
            _set_origin(element, k_link @ _origin_matrix(element) @ flip_y)
            mesh = element.find("geometry/mesh")
            if mesh is not None:
                scale = [float(value) for value in (mesh.get("scale") or "1 1 1").split()]
                mesh.set("scale", " ".join(repr(value + 0.0) for value in (scale[0], -scale[1], scale[2])))
        inertial = link.find("inertial")
        if inertial is not None:
            _set_origin(inertial, k_link @ _origin_matrix(inertial) @ flip_y)
            inertia = inertial.find("inertia")
            if inertia is not None:
                # The y reflection of the principal frame flips the products that involve y.
                for name in ("ixy", "iyz"):
                    if inertia.get(name) is not None:
                        inertia.set(name, repr(-float(inertia.get(name)) + 0.0))
    return links, joints


def _mount_on_wrist(joints: list[ET.Element], link: str, palm: np.ndarray, wrist_from_link: np.ndarray) -> None:
    """Place ``link`` at ``wrist_from_link`` (joints at zero) by moving the joint that attaches its chain to the palm.

    The official chain may have fixed links the SDK does not (the middle
    finger's abad): those stay as they are on the moved base joint.
    """
    by_child = {joint.find("child").get("link"): joint for joint in joints}
    chain = [by_child[link]]
    while chain[-1].find("parent").get("link") != "L_palm":
        chain.append(by_child[chain[-1].find("parent").get("link")])
    base, after = chain[-1], chain[-2::-1]
    base_to_link = np.eye(4)
    for joint in after:
        base_to_link = base_to_link @ _origin_matrix(joint)
    _set_origin(base, np.linalg.inv(palm) @ wrist_from_link @ np.linalg.inv(base_to_link))


def description_share() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("aimdk_description"))


def build_planning_model(
    description: Path | None = None,
    *,
    arm_limits: Mapping[str, tuple[float, float]],
    thumb_mount: str = "legacy",
    palm_mount: str = "legacy",
    side: str = "left",
) -> ET.ElementTree:
    """The planning URDF; ``arm_limits`` narrows joint ranges, never widens them.

    ``side`` names the arm that plans and the hand mounted as the ``R_*``
    aliases: the left OmniHand as described above, or for ``"right"`` its
    mirror image on the right wrist (:func:`_mirror_hand`). The other wrist
    carries no hand in either model.

    The deployed limits are the robot configuration's ``arm_limits``
    (:func:`aimdk_grasp.robot_model.left_arm_limits`); an empty mapping keeps the
    description's own ranges.

    ``description`` defaults to the installed ``aimdk_description`` share; mesh
    paths in the result are absolute paths into it.
    """
    side = check_side(side)
    description = Path(description) if description is not None else description_share()
    robot = _body(description)
    hand = ET.parse(description / HAND_URDF).getroot()
    _mount_omnihand_wrists(robot, hand, description)
    links, joints = _hand(hand, description, thumb_mount, palm_mount)
    if side == "right":
        links, joints = _mirror_hand(links, joints)
    robot.set("name", f"x2_ultra_{side}_omnihand")
    for element in (*links, *joints):
        robot.append(element)
    for name, (lower, upper) in arm_limits.items():
        limit = robot.find(f"joint[@name='{name}']/limit")
        if limit is None:
            raise ValueError(f"arm limit for unknown joint {name!r}")
        narrowed = (max(float(limit.get("lower")), lower), min(float(limit.get("upper")), upper))
        if narrowed[0] > narrowed[1]:
            raise ValueError(f"{name}: limits {lower}..{upper} do not overlap the URDF range")
        limit.set("lower", repr(narrowed[0]))
        limit.set("upper", repr(narrowed[1]))
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    return tree
