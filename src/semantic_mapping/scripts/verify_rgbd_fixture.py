#!/usr/bin/env python3
"""Verify the semantic pipeline against the repository RealSense RGB-D fixture."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from semantic_mapping.association import SemanticObservation, SemanticTracker
from semantic_mapping.database import SemanticMapDatabase
from semantic_mapping.geometry import project_masked_depth, transform_geometry
from semantic_mapping.hf_grounded_sam2 import HFGroundedSAM2
from semantic_mapping.siglip_encoder import SigLIPEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--grounding-model", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--siglip-model", required=True)
    parser.add_argument("--prompt", default="strawberry. white cube. object.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = Path(args.fixture)
    image = cv2.imread(str(fixture / "color.png"))
    depth = cv2.imread(str(fixture / "depth.png"), cv2.IMREAD_UNCHANGED)
    if image is None or depth is None:
        raise FileNotFoundError(f"RGB-D fixture is incomplete: {fixture}")
    camera_info = json.loads((fixture / "camera_info.json").read_text())
    intrinsics = np.asarray(camera_info["k"], dtype=np.float64).reshape(3, 3)

    started = time.perf_counter()
    detector = HFGroundedSAM2(
        grounding_model_path=args.grounding_model,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        device=args.device,
    )
    print(f"detector_load_sec={time.perf_counter() - started:.2f}")
    started = time.perf_counter()
    detections = detector.detect_and_segment(image, args.prompt, 0.25, 0.20)
    print(f"detection_sec={time.perf_counter() - started:.2f} count={len(detections)}")

    started = time.perf_counter()
    encoder = SigLIPEncoder(args.siglip_model, device=args.device)
    print(f"siglip_load_sec={time.perf_counter() - started:.2f}")
    tracker = SemanticTracker(association_distance_m=0.45, embedding_similarity_threshold=0.70)
    tracks = []
    for index, detection in enumerate(detections):
        geometry = project_masked_depth(detection.mask, depth, intrinsics, 1000.0, 4.0, 30)
        if geometry is None:
            print(f"detection[{index}] label={detection.label!r} has no valid depth")
            continue
        world = transform_geometry(geometry, np.array([1.0, 2.0, 0.0]), np.eye(3))
        embedding = encoder.encode(image, detection.mask, detection.bbox_xyxy)
        observation = SemanticObservation(
            label=detection.label,
            confidence=detection.confidence,
            position=world.centroid,
            size=world.size,
            point_count=world.points.shape[0],
            stamp_ns=1_000_000_000 + index,
            embedding=embedding,
        )
        first = tracker.update(observation)
        repeated = SemanticObservation(
            label=observation.label,
            confidence=observation.confidence,
            position=observation.position + np.array([0.01, 0.0, 0.0]),
            size=observation.size,
            point_count=observation.point_count,
            stamp_ns=2_000_000_000 + index,
            embedding=observation.embedding,
        )
        second = tracker.update(repeated)
        if first.object_id != second.object_id:
            raise RuntimeError("repeated observation did not preserve its semantic object ID")
        tracks.append(second)
        print(
            f"detection[{index}] label={detection.label!r} score={detection.confidence:.3f} "
            f"points={world.points.shape[0]} centroid={world.centroid.round(4).tolist()} "
            f"embedding_dim={embedding.size} stable_id={second.object_id}"
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        database = SemanticMapDatabase(str(Path(temporary_directory) / "semantic.sqlite3"))
        for track in tracks:
            database.upsert(track)
        loaded = database.load()
        database.close()
        if {track.object_id for track in loaded} != {track.object_id for track in tracks}:
            raise RuntimeError("semantic database round trip changed persistent object IDs")

    if not tracks:
        raise RuntimeError("no detection had valid aligned depth")
    print(f"database_roundtrip_count={len(tracks)}")
    print("REAL_MODEL_RGBD_SEMANTIC_PIPELINE=PASS")


if __name__ == "__main__":
    main()
