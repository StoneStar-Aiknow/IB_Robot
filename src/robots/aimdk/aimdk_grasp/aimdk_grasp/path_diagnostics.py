"""Portable evidence for rejected pose paths; inspect with python -m aimdk_grasp.path_diagnostics FILE.

Archives use pickle, like plan_replay inputs, and must come from a trusted local
recording. Classification needs only NumPy; it does not connect to the robot.
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
from contextlib import suppress
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

FAILURES_DIR_ENV = "IBROBOT_X2_PATH_FAILURES_DIR"


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def summarize(positions, times, diagnostics, joints, links, bounds=None):
    """First and worst sample per constraint, keeping metres, radians and costs separate."""
    timed = times is not None
    times = np.arange(len(positions)) if times is None else times
    positions = np.asarray(positions)
    bad = np.flatnonzero(~np.asarray(diagnostics["valid"]))
    summary = {"sample_count": len(positions), "first_invalid_sample": int(bad[0]) if len(bad) else None, "types": {}}
    for name, key in (
        ("joint_bounds", "bound_cost"),
        ("self_collision", "self_collision_cost"),
        ("world_collision", "world_penetration_m"),
        ("hand_behind_torso", "hand_behind_torso"),
    ):
        values = np.asarray(diagnostics[key]).reshape(len(positions), -1)
        severity = values.max(axis=1)
        violating = np.flatnonzero(severity > 0)
        if not len(violating):
            continue
        worst = int(np.argmax(severity))
        row = {
            "first_sample": int(violating[0]),
            "first_time_s": float(times[violating[0]]),
            "worst_sample": worst,
            "worst_time_s": float(times[worst]),
            "max_value": float(severity[worst]),
            "positions_rad": np.asarray(positions[worst]).tolist(),
        }
        if name == "joint_bounds" and bounds is not None:
            lower, upper = map(np.asarray, bounds)
            excess = np.maximum(lower - positions, positions - upper)
            sample, joint = np.unravel_index(np.argmax(excess), excess.shape)
            row.update(
                worst_sample=int(sample),
                worst_time_s=float(times[sample]),
                positions_rad=np.asarray(positions[sample]).tolist(),
                joint=joints[joint],
                actual_rad=float(positions[sample, joint]),
                lower_rad=float(lower[joint]),
                upper_rad=float(upper[joint]),
                excess_rad=float(excess[sample, joint]),
            )
        elif name == "world_collision":
            sphere = int(np.argmax(values[worst]))
            row.update(
                link=links[sphere], sphere_index=sphere, sphere_torso_m=diagnostics["spheres"][worst, sphere].tolist()
            )
        elif name == "self_collision" and "self_collision_pairs" in diagnostics:
            pairs = np.asarray(diagnostics["self_collision_pairs"], dtype=int)
            padding = np.asarray(diagnostics["self_collision_padding"])
            # Only inspect the samples the actual checker rejected. Use its
            # exact pair mask and padding, including its disabled spheres.
            deepest = None
            for sample in violating:
                balls = diagnostics["spheres"][sample]
                a, b = pairs.T
                radii = balls[:, 3] + padding
                distance = np.linalg.norm(balls[a, :3] - balls[b, :3], axis=1)
                # cuRobo ranks squared sphere penetration after adding its
                # padding; zero effective radii remain enabled in the kernel.
                squared = (radii[a] + radii[b]) ** 2 - distance**2
                squared = np.where((radii[a] >= 0) & (radii[b] >= 0), squared, -np.inf)
                if len(squared) and (deepest is None or squared.max() > deepest[0]):
                    pair = int(np.argmax(squared))
                    overlap = float(radii[a[pair]] + radii[b[pair]] - distance[pair])
                    deepest = float(squared[pair]), int(sample), int(a[pair]), int(b[pair]), overlap
            if deepest is not None:
                squared, sample, a, b, depth = deepest
                row.update(
                    pair_links=[links[a], links[b]],
                    pair_spheres=[a, b],
                    overlap_m=depth,
                    squared_penetration_m2=squared,
                    pair_sample=sample,
                    pair_time_s=float(times[sample]),
                )
        elif name == "hand_behind_torso" and "hand_min_x_m" in diagnostics:
            x = diagnostics["hand_min_x_m"]
            sample = int(np.argmin(x))
            row.update(
                worst_sample=sample,
                worst_time_s=float(times[sample]),
                positions_rad=np.asarray(positions[sample]).tolist(),
                min_x_m=float(x[sample]),
                required_min_x_m=float(diagnostics["required_hand_min_x_m"]),
            )
        if not timed:
            row = {key: value for key, value in row.items() if not key.endswith("time_s")}
        summary["types"][name] = row
    return summary


def save_failure(record, directory=None):
    """Save geometry and model files with content identities, without retaining GPU objects."""
    directory = Path(directory or os.environ.get(FAILURES_DIR_ENV) or Path.home() / ".ros/x2_path_failures")
    directory.mkdir(parents=True, exist_ok=True)
    assets = {}

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"urdf_path", "file_path"} and isinstance(item, str) and item not in assets:
                    assets[item] = Path(item).read_bytes()
                else:
                    collect(item)
        elif isinstance(value, list | tuple):
            for item in value:
                collect(item)

    collect(record)
    versions = {}
    for package in ("nvidia-curobo", "curobo", "torch", "numpy", "warp-lang"):
        with suppress(PackageNotFoundError):
            versions[package] = version(package)
    record = dict(
        record,
        schema_version=1,
        assets=assets,
        versions=versions,
        asset_sha256={name: hashlib.sha256(data).hexdigest() for name, data in assets.items()},
    )
    record["identities"] = {key: digest(record[key]) for key in ("planning_scene", "validation_scene", "models")}
    path = directory / f"pose_failure_{time.time_ns()}_{os.getpid()}.pkl"
    with path.open("xb") as stream:
        pickle.dump(record, stream, protocol=5)
    return path


def recheck(record):
    """Re-run the original fine checker on CUDA with archived assets and attachment.

    This checks the recorded candidate, without a new IK/search or any ROS I/O.
    Use the recorded cuRobo/Torch versions when comparing numerical results.
    """
    import torch
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

    from .planner import ATTACH_LINK, DISABLED_RADIUS, LeftArmPlanner, _ensure_mesh_query_distance

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for --recheck; omit it to inspect the recorded evidence")
    with TemporaryDirectory(prefix="x2_path_recheck_") as temporary:
        paths = {}
        for index, (name, data) in enumerate(record["assets"].items()):
            path = Path(temporary) / f"{index}{Path(name).suffix}"
            path.write_bytes(data)
            paths[name] = str(path)

        def relocate(value):
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, str):
                return paths.get(value, value)
            return value

        scene = relocate(copy.deepcopy(record["validation_scene"]))
        checker = RobotCollisionChecker(
            RobotCollisionCheckerCfg.load_from_config(
                robot_config=relocate(copy.deepcopy(record["models"]["fine"])),
                scene_model=scene,
                collision_activation_distance=0.0,
                n_meshes=max(1, len(scene.get("mesh", {}))),
                n_cuboids=max(1, len(scene.get("cuboid", {}))),
            )
        )
        parameters = checker.kinematics.config.kinematics_config
        index = parameters.get_sphere_index_from_link_name(ATTACH_LINK)
        attached = record["attached_spheres"]
        values = np.full((len(index), 4), [0.0, 0.0, 0.0, DISABLED_RADIUS])
        values[: len(attached)] = attached
        parameters.link_spheres[:, index, :] = torch.as_tensor(
            values, device=parameters.link_spheres.device, dtype=parameters.link_spheres.dtype
        )
        for cost in (checker.collision_cost, checker.collision_constraint):
            cost.config.update_num_spheres(parameters.total_spheres)
        _ensure_mesh_query_distance(checker.scene_model, record["mesh_query_distance"])
        # Reuse the same constraint calculation, with no solver or motion IO.
        replay = object.__new__(LeftArmPlanner)
        replay.checker, replay._checker_batch_size = checker, None
        replay._query_shared_spheres_once = False
        replay._hand_min_x_m = record["hand_min_x_m"]
        replay._hand_spheres = torch.as_tensor(
            [name.startswith("R_") for name in record["sphere_links"]], device="cuda"
        )
        return replay._constraint_diagnostics(record["positions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--recheck", action="store_true", help="re-run the archived fine check on CUDA")
    args = parser.parse_args()
    record = pickle.loads(args.archive.read_bytes())
    diagnostics = recheck(record) if args.recheck else record["diagnostics"]
    print(
        json.dumps(
            summarize(
                record["positions"],
                record["times"],
                diagnostics,
                record["joint_names"],
                record["sphere_links"],
                record["bounds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
