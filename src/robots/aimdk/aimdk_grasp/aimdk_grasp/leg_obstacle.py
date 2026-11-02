"""The fixed pelvis and left leg the fine collision check keeps the left hand away from.

The planning model is torso-rooted and has no legs. Live 2026-10-15 (move_A1) a
left-arm approach swung the palm behind the torso and into the left hip, so every
fine validation scene carries the pelvis and left leg as one fixed mesh in
``torso_link`` (``scene_estimation.LEG_OBSTACLE``).

Geometry: the official X2 Ultra pelvis, left hip and knee collision meshes, one
convex hull per link, placed by forward kinematics at the measured STAND_DEFAULT
joint angles (boot20261014stand1: waist pitch 0.144, hip pitch -0.271, knee
0.506). The legs and waist hold this pose while the upper body is in
UPPERBODY_REMOTE_SPLIT. Mesh sources are recorded relative to the
``aimdk_description`` share with their SHA-256, and the obstacle is refused when
they change.

Usage:
    export_x2_leg_obstacle <output.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from aimdk_grasp.planning_model import BODY_MESHES, BODY_URDF, description_share

FILENAME = "leg_obstacle.json"
GRID_M = 0.004
LINKS = ("pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link")
# Measured STAND_DEFAULT (boot20261014stand1, last samples); joints not listed are 0.
STANDING = {
    "waist_yaw_joint": -0.0005,
    "waist_pitch_joint": 0.1443,
    "waist_roll_joint": -0.0117,
    "left_hip_pitch_joint": -0.2708,
    "left_hip_roll_joint": 0.0147,
    "left_hip_yaw_joint": -0.0281,
    "left_knee_joint": 0.5055,
}


def floats(text, default):
    return [float(v) for v in (text or default).split()]


def transform(element):
    out = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        out[:3, :3] = Rotation.from_euler("xyz", floats(origin.get("rpy"), "0 0 0")).as_matrix()
        out[:3, 3] = floats(origin.get("xyz"), "0 0 0")
    return out


def link_poses(robot):
    """Each link's pose in the URDF root at the standing joint angles."""
    joints = {j.find("child").get("link"): j for j in robot.findall("joint")}

    def pose(link):
        out = np.eye(4)
        while link in joints:
            joint = joints[link]
            motion = np.eye(4)
            if joint.get("type") in ("revolute", "continuous"):
                axis = floats(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, "1 0 0")
                motion[:3, :3] = Rotation.from_rotvec(
                    np.asarray(axis) * STANDING.get(joint.get("name"), 0.0)
                ).as_matrix()
            out = transform(joint) @ motion @ out
            link = joint.find("parent").get("link")
        return out

    return pose


def build_leg_obstacle(description: Path) -> dict:
    import trimesh

    robot = ET.parse(description / BODY_URDF).getroot()
    pose = link_poses(robot)
    torso_from_root = np.linalg.inv(pose("torso_link"))
    links = {link.get("name"): link for link in robot.findall("link")}
    vertices, faces, sources = [], [], {}
    for name in LINKS:
        collision = links[name].find("collision")
        geometry = collision.find("geometry/mesh")
        path = description / BODY_MESHES / Path(geometry.get("filename")).name
        mesh = trimesh.load(path, force="mesh")
        # Hull of the vertices on a GRID_M grid: a few hundred faces instead of
        # thousands, within GRID_M * sqrt(3) / 2 of the exact hull.
        points = np.unique(np.round(mesh.vertices * floats(geometry.get("scale"), "1 1 1") / GRID_M), axis=0) * GRID_M
        hull = trimesh.Trimesh(points).convex_hull
        placed = torso_from_root @ pose(name) @ transform(collision)
        faces.append(hull.faces + sum(len(v) for v in vertices))
        vertices.append(hull.vertices @ placed[:3, :3].T + placed[:3, 3])
        sources[name] = {
            "path": str(path.relative_to(description)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    vertices = np.vstack(vertices)
    urdf = description / BODY_URDF
    return {
        "frame": "torso_link",
        "standing_joints": STANDING,
        "source": {
            "urdf": BODY_URDF,
            "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
            "meshes": sources,
        },
        "bounds_m": [vertices.min(0).round(4).tolist(), vertices.max(0).round(4).tolist()],
        "vertices": vertices.round(5).tolist(),
        "faces": np.vstack(faces).astype(int).reshape(-1).tolist(),
    }


@lru_cache(maxsize=1)
def _load(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["frame"] != "torso_link":
        raise ValueError("The leg obstacle must be in torso_link")
    description = description_share()
    # Where the leg sits comes from the URDF's waist, hip and knee joints, not
    # from the meshes: a description that moves a joint but no mesh would
    # otherwise leave the obstacle quietly misplaced.
    urdf = description / data["source"]["urdf"]
    if not urdf.is_file() or hashlib.sha256(urdf.read_bytes()).hexdigest() != data["source"]["urdf_sha256"]:
        raise ValueError(f"Leg obstacle was built from another {data['source']['urdf']}; regenerate it")
    for name, source in data["source"]["meshes"].items():
        mesh = description / source["path"]
        if not mesh.is_file() or hashlib.sha256(mesh.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Leg obstacle mesh does not match the description: {name}; regenerate it")
    vertices, faces = np.asarray(data["vertices"], dtype=np.float64), np.asarray(data["faces"], dtype=np.int64)
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return {"vertices": vertices, "faces": faces, "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}


def leg_obstacle(directory: Path | None = None) -> dict:
    """The fixed pelvis + left leg mesh in torso_link, as a cuRobo mesh entry."""
    from aimdk_grasp.robot_model import config_directory

    return _load(str(Path(directory or config_directory()) / FILENAME))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    obstacle = build_leg_obstacle(description_share())
    args.output.write_text(json.dumps(obstacle, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: obstacle[key] for key in ("frame", "bounds_m")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
