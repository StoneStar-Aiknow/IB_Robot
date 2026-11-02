"""The X2 Ultra planning configuration stays consistent with the generated model."""

from __future__ import annotations

import json
import os
import shutil
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import yaml
from aimdk_grasp import robot_model
from aimdk_grasp.planning_model import BODY_URDF, build_planning_model, description_share
from aimdk_grasp.robot_model import collision_robot_config, config_directory, planning_model_options


def _json(name: str) -> dict:
    return json.loads((config_directory() / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((config_directory() / "robot.yml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model(config):
    return build_planning_model(**planning_model_options(config["planning_model"])).getroot()


def test_the_configuration_names_only_links_and_joints_of_the_model(config, model):
    kinematics = config["robot_cfg"]["kinematics"]
    links = {link.get("name") for link in model.findall("link")}
    named = {
        kinematics["base_link"],
        *kinematics["tool_frames"],
        *kinematics["collision_link_names"],
        *kinematics["mesh_link_names"],
        *kinematics["collision_spheres"],
        *kinematics["self_collision_buffer"],
        *kinematics["self_collision_ignore"],
        *(link for ignored in kinematics["self_collision_ignore"].values() for link in ignored),
    }
    assert named <= links

    below_base, frontier = set(), {kinematics["base_link"]}
    while frontier:
        children = [joint for joint in model.findall("joint") if joint.find("parent").get("link") in frontier]
        below_base |= {joint.get("name") for joint in children if joint.get("type") == "revolute"}
        mimics = {joint.get("name") for joint in children if joint.find("mimic") is not None}
        below_base -= mimics
        frontier = {joint.find("child").get("link") for joint in children}
    assert set(kinematics["lock_joints"]) <= {joint.get("name") for joint in model.findall("joint")}
    assert below_base == set(kinematics["lock_joints"]) | set(kinematics["cspace"]["joint_names"])


def test_planning_urdf_first_write_is_atomic_across_threads(tmp_path, monkeypatch):
    urdf = b"<?xml version='1.0'?><robot name='concurrent'/>"
    monkeypatch.setattr(robot_model, "_load", lambda _directory: ({}, urdf))
    monkeypatch.setattr(robot_model.tempfile, "gettempdir", lambda: os.fspath(tmp_path))

    barrier = threading.Barrier(2)
    replace_sources: list[Path] = []
    real_replace = robot_model.os.replace

    def replace_together(source, destination):
        replace_sources.append(Path(source))
        barrier.wait(timeout=5.0)
        real_replace(source, destination)

    monkeypatch.setattr(robot_model.os, "replace", replace_together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _index: robot_model.planning_urdf(tmp_path / "config"), range(2)))

    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == urdf
    assert len(set(replace_sources)) == 2
    assert all(not source.exists() for source in replace_sources)


def test_covers_are_certified_and_contact_pairs_excluded():
    cover = json.loads((config_directory() / "hand_collision_cover.json").read_text())
    assert len(cover["contact_pairs"]) == 16
    for item in cover["links"].values():
        assert item["certificate"]["max_cell_vertex_excess_m"] < 0
        assert item["certificate"]["relative_volume_error"] < 1e-8
        assert len(item["spheres"]) == item["certificate"]["certified_cells"]
    kinematics = collision_robot_config()["robot_cfg"]["kinematics"]
    for a, b in cover["contact_pairs"]:
        assert b in kinematics["self_collision_ignore"][a] and a in kinematics["self_collision_ignore"][b]
    # The old false-positive pair continues to be checked geometrically.
    assert "R_index_dip" not in kinematics["self_collision_ignore"]["R_thumb_mcp"]


@pytest.mark.parametrize(
    ("name", "tamper", "message"),
    [
        ("hand_collision_cover.json", lambda c: c["links"]["R_thumb_mcp"]["spheres"].pop(), "sphere hash"),
        ("hand_collision_cover.json", lambda c: c["links"]["R_thumb_mcp"]["official_mesh"].update(sha256="0"), "mesh"),
        (
            "hand_collision_cover.json",
            lambda c: c["links"]["R_thumb_mcp"]["official_mesh"].update(origin={"rpy": "0 0 0", "xyz": "0.001 0 0"}),
            "mesh",
        ),
        (
            "hand_collision_cover.json",
            lambda c: c["links"]["R_thumb_mcp"]["official_mesh"].update(scale="2 2 2"),
            "mesh",
        ),
        ("torso_collision_cover.json", lambda c: c["sources"][0].update(sha256="0"), "Torso cover mesh hash"),
        ("hand_collision_reference.json", lambda c: c.update(collision_spheres_sha256="0"), "Reference exclusions"),
        ("hand_collision_cover.json", lambda c: c["contact_pairs"].pop(), "contact exclusions"),
    ],
)
def test_a_cover_that_no_longer_matches_is_refused(tmp_path, name, tamper, message):
    directory = tmp_path / "x2_ultra"
    shutil.copytree(config_directory(), directory)
    document = json.loads((directory / name).read_text())
    tamper(document)
    (directory / name).write_text(json.dumps(document))

    with pytest.raises(ValueError, match=message):
        collision_robot_config(directory)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda root: root.find("link[@name='L_thumb_mcp']/visual/origin").set("xyz", "0.001 0 0"),
        lambda root: root.find("link[@name='L_thumb_mcp']/visual/geometry/mesh").set("scale", "2 2 2"),
    ],
    ids=["origin", "scale"],
)
def test_a_hand_urdf_geometry_change_alone_is_refused(tmp_path, tamper):
    """The same STL, at a different placement, must not pass the old cover.

    Only the hand URDF's geometry placement changes — the mesh files, their
    hashes and the body URDF are untouched — and the cover is refused: the
    spheres were fitted where the description used to put the mesh, not where
    it puts it now.
    """
    directory = tmp_path / "x2_ultra"
    shutil.copytree(config_directory(), directory)
    share = description_share()
    description = tmp_path / "description"
    shutil.copytree(share / "urdf", description / "urdf")
    (description / "meshes").symlink_to(share / "meshes", target_is_directory=True)
    hand_urdf = description / "urdf" / "x2_ultra_omnihand_v1.4.0.urdf"
    tree = ET.parse(hand_urdf)
    tamper(tree.getroot())
    tree.write(hand_urdf, encoding="utf-8", xml_declaration=True)
    document = yaml.safe_load((directory / "robot.yml").read_text(encoding="utf-8"))
    document["planning_model"]["description"] = os.fspath(description)
    (directory / "robot.yml").write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="Hand collision cover mesh does not match"):
        collision_robot_config(directory)


def test_a_torso_mesh_scale_change_alone_is_refused(tmp_path):
    directory = tmp_path / "x2_ultra"
    shutil.copytree(config_directory(), directory)
    share = description_share()
    description = tmp_path / "description"
    shutil.copytree(share / "urdf", description / "urdf")
    (description / "meshes").symlink_to(share / "meshes", target_is_directory=True)
    body_urdf = description / BODY_URDF
    tree = ET.parse(body_urdf)
    tree.getroot().find("link[@name='torso_link']/collision/geometry/mesh").set("scale", "2 1 1")
    tree.write(body_urdf, encoding="utf-8", xml_declaration=True)
    document = yaml.safe_load((directory / "robot.yml").read_text(encoding="utf-8"))
    document["planning_model"]["description"] = os.fspath(description)
    (directory / "robot.yml").write_text(yaml.safe_dump(document))

    with pytest.raises(ValueError, match="Torso cover does not match collision meshes"):
        collision_robot_config(directory)


def test_torso_cover_applies_non_unit_mesh_scale(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from aimdk_grasp.torso_cover import fit_torso_cover

    description = tmp_path / "description"
    mesh_path = description / "meshes" / "torso.stl"
    mesh_path.parent.mkdir(parents=True)
    source_mesh = trimesh.creation.box(extents=(0.08, 0.08, 0.08))
    source_mesh.export(mesh_path)
    scale = np.array([3.0, 1.0, 0.5])
    model = ET.fromstring(
        f"""
        <robot name="scaled_torso">
          <link name="torso_link">
            <collision>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry><mesh filename="{mesh_path}" scale="3 1 0.5"/></geometry>
            </collision>
          </link>
        </robot>
        """
    )

    fitted = fit_torso_cover(model, description)
    centers = np.asarray([sphere["center"] for sphere in fitted["spheres"]])
    radius = fitted["spheres"][0]["radius"]
    scaled_vertices = np.asarray(source_mesh.vertices) * scale
    nearest = np.linalg.norm(scaled_vertices[:, None] - centers[None, :], axis=2).min(axis=1)

    assert fitted["sources"][0]["scale"] == scale.tolist()
    assert nearest.max() <= radius + 1e-9
    assert fitted["max_triangle_excess_m"] < 0.0


def test_the_torso_cover_is_reproduced_from_the_description(model):
    pytest.importorskip("trimesh")
    from aimdk_grasp.torso_cover import fit_torso_cover

    stored = _json("torso_collision_cover.json")
    fitted = fit_torso_cover(model, description_share())

    assert fitted["sources"] == stored["sources"]
    assert fitted["sphere_count"] == stored["sphere_count"]
    np.testing.assert_allclose(
        [sphere["center"] for sphere in fitted["spheres"]],
        [sphere["center"] for sphere in stored["spheres"]],
        atol=1e-9,
    )
    assert fitted["max_triangle_excess_m"] < 0.0
