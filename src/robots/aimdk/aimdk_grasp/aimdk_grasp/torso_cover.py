"""Fit the certified torso sphere cover of the X2 planning model.

The torso is filled with 40 mm voxels, each replaced by its circumscribed sphere
plus 1 mm; the cover is certified by checking that every triangle of the torso
collision mesh, subdivided to 5 mm, lies inside a single sphere. Mesh sources are
recorded relative to the ``aimdk_description`` share with their SHA-256 and
normalized URDF scale, so a changed description is detected instead of silently
mis-covered.

Usage:
    fit_x2_torso_cover <output.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from aimdk_grasp.planning_model import build_planning_model, description_share

PITCH_M = 0.04
MARGIN_M = 0.001
SUBDIVISION_M = 0.005


def fit_torso_cover(model: ET.Element, description: Path) -> dict:
    parts, sources = [], []
    for collision in model.findall("link[@name='torso_link']/collision"):
        geometry = collision.find("geometry/mesh")
        path = Path(geometry.get("filename"))
        scale = np.fromstring(geometry.get("scale", "1 1 1"), sep=" ")
        if scale.shape != (3,):
            raise ValueError(f"Torso mesh scale must contain three values: {geometry.get('scale')}")
        origin = collision.find("origin")
        mesh = trimesh.load(path, force="mesh")
        rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
        mesh.vertices = (mesh.vertices * scale) @ rotation.T + np.fromstring(origin.get("xyz"), sep=" ")
        parts.append(mesh)
        sources.append(
            {
                "path": str(path.relative_to(description)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "origin": dict(origin.attrib),
                "scale": scale.tolist(),
            }
        )
    mesh = trimesh.util.concatenate(parts)
    centers = mesh.voxelized(PITCH_M).fill().points
    radius = float(np.sqrt(3) * PITCH_M / 2 + MARGIN_M)
    # Certify each subdivided triangle lies in one convex sphere, not only its vertices.
    vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=SUBDIVISION_M)
    triangles = vertices[faces]
    _, indices = cKDTree(centers).query(triangles.mean(axis=1), k=8)
    excess = np.linalg.norm(triangles[:, None] - centers[indices, None], axis=-1).max(axis=-1).min(axis=1) - radius
    if excess.max() > 1e-9:
        raise RuntimeError(f"torso surface is not covered: {excess.max()} m")
    return {
        "method": "40 mm filled mesh voxels, circumscribed spheres plus 1 mm surface-cover margin",
        "sources": sources,
        "pitch_m": PITCH_M,
        "sphere_count": len(centers),
        "surface_triangles_certified": len(faces),
        "max_triangle_excess_m": float(excess.max()),
        "physical_validation": "URDF mesh coverage only; hardware geometry uncalibrated",
        "assembly_pairs": [["torso_link", f"{side}_shoulder_yaw_link"] for side in ("left", "right")],
        "assembly_reason": "Shoulder assembly sphere envelope overlaps torso; elbows and distal links remain checked",
        "spheres": [{"center": center.tolist(), "radius": radius} for center in centers],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    description = description_share()
    # The cover fits the torso mesh alone; arm limits do not move any link.
    cover = fit_torso_cover(build_planning_model(description, arm_limits={}).getroot(), description)
    args.output.write_text(json.dumps(cover, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in cover.items() if key not in ("spheres", "sources")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
