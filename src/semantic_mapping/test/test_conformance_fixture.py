"""Validate the fixed RGB-D/TF conformance fixture and reference outputs."""

import json
from pathlib import Path

import cv2
import numpy as np

from semantic_mapping.geometry import project_masked_depth, quaternion_matrix, transform_geometry

FIXTURE = Path(__file__).parents[2] / "perception_service" / "test" / "fixtures" / "realsense_rgbd_frame"


def _load_fixture():
    reference = json.loads((FIXTURE / "conformance_reference.json").read_text(encoding="utf-8"))
    camera_info = json.loads((FIXTURE / reference["source"]["camera_info"]).read_text(encoding="utf-8"))
    color = cv2.imread(str(FIXTURE / reference["source"]["rgb"]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(FIXTURE / reference["source"]["aligned_depth"]), cv2.IMREAD_UNCHANGED)
    return reference, camera_info, color, depth


def test_conformance_fixture_source_contract_is_fixed() -> None:
    reference, camera_info, color, depth = _load_fixture()

    assert reference["format_version"] == 1
    assert color.shape == (480, 640, 3)
    assert depth.shape == color.shape[:2]
    assert depth.dtype == np.uint16
    assert camera_info["header"]["frame_id"] == reference["source"]["source_frame"]
    assert camera_info["header"]["stamp"] == reference["source"]["stamp"]


def test_reference_outputs_cover_all_backend_conformance_stages() -> None:
    reference, _, _, _ = _load_fixture()
    outputs = reference["reference_outputs"]

    assert set(outputs) == {"sam2", "ram_plus", "siglip2", "grounding_dino", "geometry"}
    assert outputs["sam2"]["mask_count"] == 1
    assert len(outputs["ram_plus"]["tags"]) == len(outputs["ram_plus"]["scores"])
    embedding = np.asarray(outputs["siglip2"]["embedding"], dtype=np.float32)
    assert len(embedding) == outputs["siglip2"]["embedding_dim"]
    assert np.isclose(np.linalg.norm(embedding), 1.0)
    assert outputs["grounding_dino"]["label"] in reference["inputs"]["candidate_labels"]


def test_reference_geometry_recomputes_from_aligned_depth_and_tf() -> None:
    reference, camera_info, _, depth = _load_fixture()
    inputs = reference["inputs"]
    expected = reference["reference_outputs"]["geometry"]
    x1, y1, x2, y2 = inputs["mask_rect_xyxy"]
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    intrinsics = np.asarray(camera_info["k"], dtype=np.float64).reshape(3, 3)

    geometry = project_masked_depth(
        mask,
        depth,
        intrinsics,
        inputs["depth_scale"],
        inputs["depth_trunc_m"],
        inputs["min_points"],
    )
    assert geometry is not None
    transform = inputs["global_transform"]
    rotation = quaternion_matrix(*transform["quaternion_xyzw"])
    world = transform_geometry(geometry, np.asarray(transform["translation_xyz"]), rotation)

    assert len(geometry.points) == expected["valid_point_count"]
    assert np.allclose(geometry.centroid, expected["camera_centroid_xyz"], atol=1e-7)
    assert np.allclose(world.centroid, expected["world_centroid_xyz"], atol=1e-7)
    assert np.allclose(world.size, expected["extent_xyz"], atol=1e-7)
