"""Offline-only seed compiler; run on the workstation that has MuJoCo."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from aimdk_grasp.fixed_pose_backend import START_TOLERANCE_RAD
from aimdk_robot.pose_backend import model, safety
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from pose_composer import spec
from pose_composer.fixed_backend import FixedBackend, digest
from robot_runtime.task_poses import normalize_task_poses, task_pose_arm_targets


def transform(pos, quat):
    out = np.eye(4)
    out[:3, 3] = pos
    out[:3, :3] = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    return out.tolist()


def export_geometry(robot):
    m = robot.model
    bodies = []
    for i in range(m.nbody):
        joints = []
        for j in range(m.njnt):
            if m.jnt_bodyid[j] != i:
                continue
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("only welded/hinge body models can be compiled")
            joints.append({"name": m.joint(j).name, "axis": m.jnt_axis[j].tolist(), "position": m.jnt_pos[j].tolist()})
        bodies.append(
            {
                "parent": int(m.body_parentid[i]),
                "transform": transform(m.body_pos[i], m.body_quat[i]),
                "joints": joints,
                "mass": float(m.body_mass[i]),
                "com": m.body_ipos[i].tolist(),
            }
        )
    pairs = safety.pairs()
    ids = sorted({g for a, b, *_ in pairs for g in (a, b)})
    shapes = []
    for g in ids:
        shape = {
            "body": int(m.geom_bodyid[g]),
            "transform": transform(m.geom_pos[g], m.geom_quat[g]),
            "size": m.geom_size[g].tolist(),
        }
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.geom_dataid[g]
            vertices = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]].astype(float)
            vertices = vertices[ConvexHull(vertices).vertices]
            shape.update(kind="mesh", vertices=vertices.tolist(), radius=float(np.linalg.norm(vertices, axis=1).max()))
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape.update(kind="cylinder", radius=float(np.linalg.norm(m.geom_size[g][:2])))
        else:
            raise ValueError("unsupported geometry in compiled body model")
        shapes.append(shape)
    return {
        "bodies": bodies,
        "shapes": shapes,
        "pairs": [[ids.index(a), ids.index(b), margin, name] for a, b, margin, name in pairs],
    }


def compile_library(mjcf, profile, limits, seeds, selected, *, task_poses=None, hand_path_check="online"):
    if hand_path_check not in {"online", "off"}:
        raise ValueError("hand_path_check must be online or off")
    snapshot = normalize_task_poses(task_poses) if task_poses else {}
    robot = model.configure(mjcf, profile, limits, task_poses=snapshot)
    body = export_geometry(robot)
    backend = FixedBackend(profile, limits, body_geometry=body)
    home = dict(zip(model.ARM_JOINTS, [0.4, 0, 0, -1.2, 0, 0, 0] * 2, strict=True))
    if snapshot:
        if "arms_rest" not in snapshot["poses"]:
            raise ValueError("fixed seed compilation requires the configured arms_rest anchor")
        home = snapshot["poses"]["arms_rest"]["joint_positions"]
    data = {
        "schema_version": 1,
        "joint_names": list(model.ARM_JOINTS),
        "robot_fingerprint": backend.fingerprint,
        "body_geometry": body,
        "poses": {},
        "disabled": {},
        # This records compilation scope; it never bypasses the live validation policy.
        "compile_validation": {"hand": hand_path_check, "body": "online"},
    }
    for record in yaml.safe_load(Path(seeds).read_text())["poses"]:
        name, description = record["id"], record["description"]
        if name not in selected:
            data["disabled"][name] = "not selected for this deployment"
            continue
        if any(description.get(side, {}).get("keep") for side in ("left", "right")):
            raise ValueError(f"{name}: keep-current semantics require a separate compiler")
        composition = spec.compose(description, home)
        if composition.kind != "pose" or len(composition.frames) != 1:
            raise ValueError(f"{name}: not a single executable seed pose")
        target = composition.frames[0]
        anchors, references = task_pose_arm_targets(description, snapshot)
        if any(target[n] != value for n, value in anchors.items()):
            raise ValueError(f"{name}: task-pose anchor changed during composition")
        if references:
            data["task_poses"] = snapshot
        if composition.approx:
            raise ValueError(f"{name}: approximate seed target is not a fixed action")
        if not safety.check_pose(target).ok or not backend.check_pose(target).ok:
            raise ValueError(f"{name}: target rejected")
        for start, end in ((home, target), (target, home)):
            plan_builder = backend.plan_safe if hand_path_check == "online" else backend.plan_for_execution
            plan = plan_builder(start, [end], [0.0])
            if not plan.ok:
                raise ValueError(f"{name}: CPU path rejected: {plan.report}")
            backend.body_geometry.check_path(model.ARM_JOINTS, plan.traj.positions)
            backend.body_geometry.check_path(
                model.ARM_JOINTS, plan.traj.positions[:2], start_tolerance_rad=START_TOLERANCE_RAD
            )
            if hand_path_check == "online":
                backend.geometry.check_arm_path(backend.observation(start), model.ARM_JOINTS, plan.traj.positions)
            # Match the live off policy: validate the entry's reference open
            # fingers, without claiming a future finger-error tube was checked.
            options = {"finger_radius": 0.0} if hand_path_check == "off" else {}
            backend.geometry.check_arm_path(
                backend.observation(start),
                model.ARM_JOINTS,
                plan.traj.positions[:2],
                start_tolerance_rad=START_TOLERANCE_RAD,
                **options,
            )
            original = safety.check_path(
                plan.traj.times, plan.traj.positions, model.ARM_JOINTS, samples=len(plan.traj.times)
            )
            if not original.ok:
                raise ValueError(f"{name}: workstation path rejected: {original}")
        data["poses"][name] = {
            "phrases": record["phrases"],
            "user_message": record["user_message"],
            "description": description,
            "target": target,
        }
    if set(data["poses"]) != set(selected):
        raise ValueError("unknown or uncompiled seed selection")
    data["digest"] = digest(data)
    return data


def update_fixed_catalog(library, catalog_root):
    """Refresh a candidate catalog from compiled aliases, never from rejected seeds.

    The caller selects an existing catalog copy. No runtime reload or deployment
    is performed; the artifact and this catalog must be reviewed together.
    """
    path = Path(catalog_root) / "config/skills/free_pose/manifest.yaml"
    manifest = yaml.safe_load(path.read_text())
    if manifest.get("name") != "free_pose":
        raise ValueError("expected a free_pose catalog manifest")
    aliases, targets = [], []
    for name, record in library["poses"].items():
        targets.append(name)
        targets.extend(record["phrases"])
        aliases.extend(record["phrases"])
    manifest["capability"]["parameters"]["properties"]["target_name"] = {
        "type": "string",
        "enum": list(dict.fromkeys(targets)),
    }
    manifest["description"]["aliases_zh"] = list(dict.fromkeys(aliases))
    manifest["description"]["summary"] = (
        "Execute a compiled seed; task-pose arms use configured arms_rest, never keep-current."
    )
    manifest["description"]["when_to_use"] = [
        "the user requests a pose in the compiled enum",
        "arms_down returns both arms to arms_rest and runs its configured hand preparation",
    ]
    manifest["capability"]["summary"] = (
        "Execute only a listed compiled pose. Single-arm seeds return the other arm to arms_rest. "
        "arms_down runs arms_rest preparation, including its configured hands."
    )
    path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--limits", required=True, help="YAML mapping of all 14 arm limits")
    parser.add_argument("--task-poses", help="SSOT task-pose YAML; required for task_pose arm anchors")
    parser.add_argument("--catalog-root", help="optional candidate catalog copy whose free_pose enum is refreshed")
    parser.add_argument(
        "--hand-path-check",
        choices=("online", "off"),
        default="online",
        help="off checks the reference hand entry only; live policy still independently gates execution",
    )
    parser.add_argument("--pose", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = compile_library(
        args.mjcf,
        args.profile,
        yaml.safe_load(Path(args.limits).read_text()),
        args.seeds,
        args.pose,
        task_poses=yaml.safe_load(Path(args.task_poses).read_text()) if args.task_poses else None,
        hand_path_check=args.hand_path_check,
    )
    with Path(args.output).open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
    if args.catalog_root:
        update_fixed_catalog(data, args.catalog_root)
    print(f"compiled {list(data['poses'])}; digest={data['digest']}")


if __name__ == "__main__":
    main()
