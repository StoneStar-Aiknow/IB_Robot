#!/usr/bin/env python3
"""Search and preview an unapproved X2 two-hand overhead-heart candidate.

Run on the workstation with existing numerical packages. This tool never sends
robot commands. Its output is explicitly unvalidated until model uncertainty,
head motion/sensor envelope and all entry/exit paths have also been checked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import differential_evolution, minimize


def source_packages(root):
    """Use only this worktree's pure numerical source packages, never an overlay."""
    for relative in ("src/robots/aimdk/aimdk_grasp", "src/robot_runtime"):
        sys.path.insert(0, str(root / relative))


def models(root):
    from aimdk_grasp import robot_model
    from aimdk_grasp.joints import ARM_MIRROR_SIGNS, HEAD, LEFT_ARM, RIGHT_ARM
    from aimdk_grasp.robot_spheres import RobotSphereModel

    description = root / "src/robots/aimdk/aimdk_description"
    configuration = root / "src/robots/aimdk/aimdk_grasp/config"
    profile_file = root / "src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml"
    profile = yaml.safe_load(profile_file.read_text())
    limits = profile["vendor"]["arm"]["limits"]

    def options(value):
        side = value.get("side", "left")
        return {
            "description": description,
            "arm_limits": {name: tuple(limits[name]) for name in (LEFT_ARM if side == "left" else RIGHT_ARM)},
            "side": side,
            "palm_mount": value.get("palm_mount", "legacy"),
            "thumb_mount": value.get("thumb_mount", "legacy"),
        }

    # Offline path resolution only. The model builder and cover/hash checks stay
    # unchanged; ROS/ament package discovery is not needed on this workstation.
    robot_model.description_share = lambda: description
    robot_model.planning_model_options = options
    paths = {side: configuration / ("x2_ultra" if side == "left" else "x2_ultra_right") for side in ("left", "right")}
    result = {side: RobotSphereModel(directory=paths[side]) for side in paths}
    trees = {side: robot_model.planning_model(paths[side]) for side in paths}
    return result, trees, profile, (LEFT_ARM, RIGHT_ARM, HEAD, ARM_MIRROR_SIGNS)


class Feedback:
    def __init__(self, names):
        self.names = names

    def vector(self, names):
        return np.zeros(len(names))

    def hand(self, _side):
        return np.zeros(10)


def hand_pose(model, curl):
    aliases = {
        name: 0.0 for name in model.mapping.to_alias(np.clip(np.zeros(10), model.mapping.lower, model.mapping.upper))
    }
    for name in ("R_index_pip_joint", "R_middle_pip_joint", "R_ring_pip_joint", "R_pinky_pip_joint"):
        aliases[name] = float(curl)
    native = model.mapping.to_native(aliases)
    model.mapping.kinematic_alias(native)
    return native


def positions(model, arm, other, hand, feedback):
    placed, valid = model.spheres(feedback, np.atleast_2d(arm), other=np.asarray(other), head=np.zeros(2), hand=hand)
    return placed[0], bool(valid[0]), model.last_link_frames[0].copy()


def surface_gap(a, b):
    values = np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=-1) - a[:, None, 3] - b[None, :, 3]
    index = np.unravel_index(np.argmin(values), values.shape)
    return float(values[index]), [int(v) for v in index]


def group_indices(model):
    names = model.sphere_links
    return {
        "hand": np.array([i for i, name in enumerate(names) if name.startswith("R_")], int),
        "forearm_wrist": np.array(
            [
                i
                for i, name in enumerate(names)
                if name.startswith(model.side + "_") and any(part in name for part in ("elbow", "wrist"))
            ],
            int,
        ),
        "head": np.array([i for i, name in enumerate(names) if name.startswith("head_")], int),
        "fingertips": np.array(
            [i for i, name in enumerate(names) if name in ("R_index_dip", "R_middle_dip", "R_ring_dip", "R_pinky_dip")],
            int,
        ),
    }


def finger_geometry(model, frames):
    """Actual fixed URDF fingertip markers, not distal cover-sphere centers."""
    tips, bases = [], []
    for finger in ("index", "middle", "ring", "pinky"):
        tip = next(joint for joint in model.joints if joint["child"] == f"R_{finger}_tip")
        distal = frames[model.links.index(tip["parent"])]
        tips.append(distal[:3, :3] @ tip["transform"][:3, 3] + distal[:3, 3])
        bases.append(frames[model.links.index(f"R_{finger}_pip"), :3, 3])
    return np.mean(tips, axis=0), np.mean(bases, axis=0)


class Search:
    def __init__(self, model, names, comparison_clearance=0.20):
        self.model, self.names = model, names
        self.comparison_clearance = comparison_clearance
        self.feedback = Feedback(names)
        self.indices = {side: group_indices(item) for side, item in model.items()}
        self.best = None
        self.frame_evidence = self.verify_palm_axes()

    def verify_palm_axes(self):
        """Check the actual SDK fingers, rather than a grasp-frame comment.

        In the generated R_palm frame the straight fingers extend along +Z
        and curl toward +Y, on BOTH hands. The planner_service +X comment is
        about the OmniHand grasp frame, not this palm tool frame. The right
        palm's X-reflection leaves its physical +Y normal unchanged.
        """
        evidence = {}
        for side, model in self.model.items():
            samples = []
            for curl in (0.0, 0.01):
                spheres, valid, frames = positions(
                    model, np.zeros(7), np.zeros(7), hand_pose(model, curl), self.feedback
                )
                if not valid:
                    raise ValueError("SDK palm-frame calibration pose is invalid")
                palm = frames[model.links.index("R_palm")]
                tips = np.mean(spheres[self.indices[side]["fingertips"], :3], axis=0)
                samples.append(palm[:3, :3].T @ (tips - palm[:3, 3]))
            derivative = (samples[1] - samples[0]) / 0.01
            if derivative[1] <= 0.05 or abs(derivative[0]) > 0.01 or samples[0][2] <= 0.1:
                raise ValueError(f"Unexpected {side} SDK finger curl/forward axes: {derivative}")
            evidence[side] = {
                "palm_normal_local": [0.0, 1.0, 0.0],
                "hand_forward_local": [0.0, 0.0, 1.0],
                "zero_curl_tip_center_local_m": samples[0].tolist(),
                "tip_center_local_derivative_per_rad": derivative.tolist(),
            }
        return evidence

    def evaluate(self, values, detailed=False):
        left = np.asarray(values[:7])
        right = left * self.names[3]
        curl = float(values[7])
        hands = {side: hand_pose(item, curl) for side, item in self.model.items()}
        placed, frames = {}, {}
        for side, q, other in (("left", left, right), ("right", right, left)):
            placed[side], valid, frames[side] = positions(self.model[side], q, other, hands[side], self.feedback)
            if not valid:
                return (1e6, {}) if detailed else 1e6
        metrics, points = {}, {}
        for side in self.model:
            q = left if side == "left" else right
            indices = self.indices[side]
            head = placed[side][indices["head"]]
            for group in ("hand", "forearm_wrist"):
                gap, pair = surface_gap(placed[side][indices[group]], head)
                metrics[f"{side}_{group}_head_gap_m"] = gap
                points[f"{side}_{group}_head_pair"] = pair
            palm_index = self.model[side].links.index("R_palm")
            points[f"{side}_palm_m"] = frames[side][palm_index, :, 3].tolist()
            palm_rotation = frames[side][palm_index, :3, :3]
            normal, forward = palm_rotation[:, 1], palm_rotation[:, 2]
            points[f"{side}_palm_normal"] = normal.tolist()
            points[f"{side}_hand_forward"] = forward.tolist()
            metrics[f"{side}_palm_down_error_deg"] = float(np.degrees(np.arccos(np.clip(-normal[2], -1, 1))))
            metrics[f"{side}_palm_down_tilt_deg"] = float(np.degrees(np.arcsin(np.clip(-normal[2], -1, 1))))
            metrics[f"{side}_hand_forward_inward"] = float(forward[1] * (-1 if side == "left" else 1))
            metrics[f"{side}_head_surface_top_m"] = float(np.max(head[:, 2] + head[:, 3]))
            # Remove the wrist roll analytically, with all upstream joints held
            # fixed. Tool-frame Rx(pi) commutes with the wrist's local X axis.
            roll = q[6] * (1 if side == "left" else -1)
            neutral_forward = palm_rotation @ np.array([0.0, np.sin(roll), np.cos(roll)])
            metrics[f"{side}_wrist_roll_forward_z_lowering"] = float(neutral_forward[2] - forward[2])
            hand_spheres = placed[side][indices["hand"]]
            points[f"{side}_hand_top_m"] = float(np.max(hand_spheres[:, 2] + hand_spheres[:, 3]))
            tips, bases = finger_geometry(self.model[side], frames[side])
            actual_forward = (tips - bases) / np.linalg.norm(tips - bases)
            points[f"{side}_fingertips_center_m"] = tips.tolist()
            points[f"{side}_finger_bases_center_m"] = bases.tolist()
            points[f"{side}_actual_finger_direction"] = actual_forward.tolist()
            metrics[f"{side}_actual_finger_inward_dot"] = float(actual_forward[1] * (-1 if side == "left" else 1))
            metrics[f"{side}_actual_finger_direction_z"] = float(actual_forward[2])
            metrics[f"{side}_tip_below_palm_m"] = float(points[f"{side}_palm_m"][2] - tips[2])
        metrics["hand_hand_gap_m"], _ = surface_gap(
            placed["left"][self.indices["left"]["hand"]], placed["right"][self.indices["right"]["hand"]]
        )
        lp, rp = np.array(points["left_palm_m"]), np.array(points["right_palm_m"])
        metrics["palm_span_m"] = float(np.linalg.norm(lp - rp))
        metrics["palm_height_m"] = float((lp[2] + rp[2]) / 2)
        metrics["palm_forward_m"] = float((lp[0] + rp[0]) / 2)
        lt, rt = (np.asarray(points[f"{side}_fingertips_center_m"]) for side in ("left", "right"))
        metrics["fingertip_span_m"] = float(np.linalg.norm(lt - rt))
        metrics["fingertip_height_m"] = float((lt[2] + rt[2]) / 2)
        for side, q in (("left", left), ("right", right)):
            metrics[f"{side}_wrist_yaw_rad"] = float(q[4])
            metrics[f"{side}_wrist_pitch_rad"] = float(q[5])
            metrics[f"{side}_wrist_roll_rad"] = float(q[6])
        metrics["minimum_head_gap_m"] = min(value for key, value in metrics.items() if key.endswith("head_gap_m"))
        # The 20 cm requirement stays fixed. We optimize a visible raised arc,
        # but a numeric minimum is not a recognition or safety certificate.
        head_gaps = [value for key, value in metrics.items() if key.endswith("head_gap_m")]
        penalty = sum(max(0.0, self.comparison_clearance + 0.003 - gap) ** 2 for gap in head_gaps) * 20000
        penalty += max(0.0, 0.015 - metrics["hand_hand_gap_m"]) ** 2 * 2000
        penalty += (metrics["palm_span_m"] - 0.32) ** 2 * 3
        penalty += (metrics["palm_forward_m"] - 0.20) ** 2 * 2
        penalty += (metrics["palm_height_m"] - 0.70) ** 2 * 5
        penalty += (metrics["fingertip_span_m"] - 0.06) ** 2 * 60
        # Euclidean head clearance alone can also be satisfied in front of the
        # chest. An overhead design must explicitly stay above the head.
        head_top = max(metrics[f"{side}_head_surface_top_m"] for side in self.model)
        penalty += max(0.0, head_top + 0.015 - metrics["palm_height_m"]) ** 2 * 3000
        penalty += max(0.0, head_top - metrics["fingertip_height_m"]) ** 2 * 3000
        penalty += (metrics["fingertip_height_m"] - (head_top + 0.035)) ** 2 * 10
        for side in self.model:
            normal = np.asarray(points[f"{side}_palm_normal"])
            forward = np.asarray(points[f"{side}_hand_forward"])
            penalty += float(np.sum((normal - [0, 0, -1]) ** 2)) * 25
            penalty += float(np.sum((forward - [0, -0.98 if side == "left" else 0.98, -0.20]) ** 2)) * 6
            penalty += max(0.0, 0.30 - metrics[f"{side}_wrist_roll_forward_z_lowering"]) ** 2 * 8
            penalty += max(0.0, 0.015 - metrics[f"{side}_tip_below_palm_m"]) ** 2 * 10000
            penalty += max(0.0, metrics[f"{side}_actual_finger_direction_z"] + 0.10) ** 2 * 10
        # The SDK's wrist_roll is flexion toward the palm (+Y in R_palm).
        # A substantial bend is desired, with reserved travel at every wrist.
        penalty += (abs(left[6]) - 1.05) ** 2 * 0.5
        penalty += left[5] ** 2 * 0.01
        report = {
            "arm_target_rad": np.r_[left, right].tolist(),
            "hand_target_rad": np.r_[hands["left"], hands["right"]].tolist(),
            "metrics": metrics,
            "points": points,
            "palm_frame_evidence": self.frame_evidence,
            "objective": float(penalty),
        }
        if self.best is None or penalty < self.best[0]:
            self.best = float(penalty), report
        return (float(penalty), report) if detailed else float(penalty)

    def wrist_diagnostics(self, candidate):
        """Vary one wrist at a time, holding all proximal joints fixed."""
        arms = np.asarray(candidate["arm_target_rad"]).reshape(2, 7)
        hands = np.asarray(candidate["hand_target_rad"]).reshape(2, 10)
        result = {}
        for index, side in enumerate(("left", "right")):
            model, q, other = self.model[side], arms[index], arms[1 - index]

            def measure(arm, model=model, other=other, hand=hands[index]):
                _, valid, frames = positions(model, arm, other, hand, self.feedback)
                if not valid:
                    raise ValueError("Wrist diagnostic sample exceeds model bounds")
                tip, base = finger_geometry(model, frames)
                return tip, (tip - base) / np.linalg.norm(tip - base)

            tip, direction = measure(q)
            neutral = q.copy()
            neutral[6] = 0.0
            neutral_tip, neutral_direction = measure(neutral)
            derivatives = {}
            for axis, name in zip((4, 5, 6), ("yaw", "pitch", "roll"), strict=True):
                samples = []
                for step in (-1e-4, 1e-4):
                    shifted = q.copy()
                    shifted[axis] += step
                    samples.append(measure(shifted)[0][2])
                derivatives[name] = float((samples[1] - samples[0]) / 2e-4)
            result[side] = {
                "fixed_proximal_joint_values_rad": q[:4].tolist(),
                "fingertip_z_derivative_m_per_rad": derivatives,
                "roll_neutral_tip_z_m": float(neutral_tip[2]),
                "current_tip_z_m": float(tip[2]),
                "tip_z_change_from_roll_neutral_m": float(tip[2] - neutral_tip[2]),
                "finger_bend_from_roll_neutral_deg": float(
                    np.degrees(np.arccos(np.clip(direction @ neutral_direction, -1, 1)))
                ),
                "roll_increase_lowers_tip": derivatives["roll"] < 0,
                "scope": "static local FK diagnostic only; no continuous motion or safety approval",
            }
        return result


def merged_urdf(trees):
    root = copy.deepcopy(trees["left"])
    for element in trees["right"]:
        name = element.get("name", "")
        if not name.startswith("R_"):
            continue
        item = copy.deepcopy(element)
        for child in item.iter():
            for key in ("name", "link", "joint"):
                if child.get(key, "").startswith("R_"):
                    child.set(key, "physical_right_" + child.get(key)[2:])
        root.append(item)
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.SubElement(root, "mujoco")
    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.attrib.update({"discardvisual": "false", "strippath": "false", "balanceinertia": "true"})
    return root


def render(root, trees, candidate, model, names, output):
    import mujoco
    from PIL import Image, ImageDraw

    merged = merged_urdf(trees)
    path = output / "model" / "x2_two_sdk_hands.urdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged).write(path, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    values = dict(zip(names[0] + names[1], candidate["arm_target_rad"], strict=True))
    native = np.asarray(candidate["hand_target_rad"])
    for side, offset in (("left", 0), ("right", 10)):
        aliases = model[side].mapping.kinematic_alias(native[offset : offset + 10])
        for name, value in aliases.items():
            values[name if side == "left" else "physical_right_" + name[2:]] = value
    # The thumb PIP is mechanically coupled by the planning URDF's mimic.
    for side, tree in trees.items():
        prefix = "" if side == "left" else "physical_right_"
        for joint in tree.findall("joint"):
            mimic = joint.find("mimic")
            if mimic is not None and joint.get("name", "").startswith("R_"):
                source = mimic.get("joint")
                source = source if not prefix else prefix + source[2:]
                target = joint.get("name") if not prefix else prefix + joint.get("name")[2:]
                values[target] = values[source] * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
    for index in range(m.njnt):
        name = m.joint(index).name
        if name in values:
            d.qpos[m.jnt_qposadr[index]] = values[name]
    mujoco.mj_forward(m, d)
    torso = m.body("torso_link").id
    torso_position, torso_rotation = d.xpos[torso].copy(), d.xmat[torso].reshape(3, 3).copy()

    def world_point(value):
        return torso_position + torso_rotation @ np.asarray(value)

    for index in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[index])).name or ""
        if body.startswith("physical_right_") or body.startswith("R_"):
            m.geom_rgba[index] = [0.65, 0.68, 0.75, 1]
        else:
            m.geom_rgba[index] = [0.22, 0.35, 0.52, 1]
    m.vis.headlight.ambient[:] = 0.55
    m.vis.headlight.diffuse[:] = 0.8
    m.vis.global_.offwidth = 1000
    m.vis.global_.offheight = 1000
    renderer = mujoco.Renderer(m, height=1000, width=1000)
    output.joinpath("preview").mkdir(exist_ok=True)
    origins, lookats = {}, {}
    low = np.min(d.geom_xpos - m.geom_rbound[:, None], axis=0)
    high = np.max(d.geom_xpos + m.geom_rbound[:, None], axis=0)
    lookat = (low + high) / 2
    for label, azimuth in (("front", 0), ("side", 90), ("right", -90), ("hands_front", 0)):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance = max(float(np.linalg.norm(high - low)) * 1.5, 1.8)
        camera.azimuth, camera.elevation = azimuth, -5
        if label == "hands_front":
            camera.lookat[:] = world_point(
                np.mean([candidate["points"][f"{side}_palm_m"] for side in ("left", "right")], axis=0)
            )
            camera.distance, camera.elevation = 0.9, -12
        renderer.update_scene(d, camera=camera)
        position = np.asarray(renderer.scene.camera[0].pos)
        wanted_axis, wanted_sign = (0, 1) if "front" in label else (1, 1 if label == "side" else -1)
        if wanted_sign * (position[wanted_axis] - camera.lookat[wanted_axis]) < 0:
            camera.azimuth += 180
            renderer.update_scene(d, camera=camera)
        # Diagnostic overlays are scene geoms only, never added to the source
        # model, sphere cover, dynamics, or collision calculations. Mark each
        # physical palm face green and its back magenta; cyan arrows are normals.
        for side in ("left", "right"):
            palm = world_point(candidate["points"][f"{side}_palm_m"])
            normal = torso_rotation @ np.asarray(candidate["points"][f"{side}_palm_normal"])
            forward = torso_rotation @ np.asarray(candidate["points"][f"{side}_hand_forward"])
            across = np.cross(normal, forward)
            matrix = np.column_stack((across, normal, forward))
            for sign, color in ((1, [0.1, 0.85, 0.3, 1]), (-1, [0.95, 0.15, 0.65, 1])):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([0.020, 0.002, 0.028]),
                    palm + forward * 0.040 + normal * sign * 0.026,
                    matrix.flatten(),
                    np.asarray(color, np.float32),
                )
                renderer.scene.ngeom += 1
            for direction, color in ((normal, [0, 0.85, 1, 1]), (forward, [1, 0.75, 0, 1])):
                geom = renderer.scene.geoms[renderer.scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).flatten(),
                    np.asarray(color, np.float32),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, palm, palm + direction * 0.14)
                renderer.scene.ngeom += 1
        origins[label] = np.asarray(renderer.scene.camera[0].pos).tolist()
        lookats[label] = camera.lookat.tolist()
        picture = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, 1000, 112), fill=(245, 245, 245))
        draw.text((15, 12), "X2 + two articulated SDK-mount hands | KINEMATIC CANDIDATE", fill=(15, 15, 15))
        draw.text((15, 33), "UNVALIDATED: no dynamics, head envelope or transition approval", fill=(150, 15, 15))
        clearance = float(candidate.get("comparison_clearance_m", 0.20))
        draw.text(
            (15, 53),
            f"Comparison search: {clearance * 100:.0f} cm | runtime minimum remains 20 cm",
            fill=(150, 15, 15),
        )
        draw.text(
            (15, 73), "Face markers: GREEN=palm, MAGENTA=back | CYAN=normal, GOLD=finger-forward", fill=(15, 15, 15)
        )
        metrics = candidate["metrics"]
        draw.text(
            (15, 91),
            f"Down error L/R: {metrics['left_palm_down_error_deg']:.1f}/{metrics['right_palm_down_error_deg']:.1f} deg"
            f" | minimum head gap: {metrics['minimum_head_gap_m'] * 100:.2f} cm | {label}",
            fill=(15, 15, 15),
        )
        picture.save(output / "preview" / f"{label}.png")
    renderer.close()
    return {
        "mujoco_version": mujoco.__version__,
        "nq": int(m.nq),
        "joints": int(m.njnt),
        "geoms": int(m.ngeom),
        "camera_origins_m": origins,
        "camera_lookats_m": lookats,
        "camera_lookat_m": camera.lookat.tolist(),
        "view_directions": {"front": "from +X", "side": "from +Y", "right": "from -Y", "hands_front": "from +X"},
        "overlays": "diagnostic only: green palm face, magenta back, cyan palm normal, gold finger-forward",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=35)
    parser.add_argument("--seed", type=int, default=84)
    parser.add_argument("--comparison-clearance", type=float, choices=(0.15, 0.20), default=0.20)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    source_packages(root)
    started = time.monotonic()
    model, trees, profile, names = models(root)
    search = Search(model, names, args.comparison_clearance)
    limits = profile["vendor"]["arm"]["limits"]
    bounds = []
    for name in names[0]:
        low, high = limits[name]
        margin = 0.12 if "wrist" in name else 0.025
        bounds.append((low + margin, high - margin))
    bounds.append((0.15, 0.55))
    result = differential_evolution(
        search.evaluate, bounds, seed=args.seed, maxiter=args.maxiter, popsize=8, polish=False, workers=1
    )
    polished = minimize(search.evaluate, result.x, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
    selected = polished.x if polished.fun < result.fun else result.x
    _, candidate = search.evaluate(selected, detailed=True)
    candidate["wrist_flexion_diagnostics"] = search.wrist_diagnostics(candidate)
    candidate.update(
        {
            "schema_version": 1,
            "name": "overhead_heart",
            "validated": False,
            "required_head_clearance_m": 0.20,
            "comparison_clearance_m": args.comparison_clearance,
            "validation_scope": "static conservative covers at zero head angles only",
            "missing_acceptance": [
                "head motion and sensor envelope",
                "continuous entry/exit paths",
                "self/body collision acceptance",
                "human visual review",
                "hardware test",
            ],
            "arm_joint_names": list(names[0] + names[1]),
            "model_sources": {
                "description": str(root / "src/robots/aimdk/aimdk_description"),
                "palm_mount": "legacy",
                "thumb_mount": "legacy",
            },
            "optimizer": {"seed": args.seed, "iterations": int(result.nit), "evaluations": int(result.nfev)},
            "wrist_soft_margin_rad": 0.12,
            "search_arm_bounds_rad": bounds[:7],
            "palm_semantics": "physical R_palm +Y normal down; +Z hand-forward inward on both SDK hands",
            "review_state": "unvalidated_palms_down_candidate",
        }
    )
    candidate["wrist_hard_limit_reserve_rad"] = {
        name: float(min(value - limits[name][0], limits[name][1] - value))
        for name, value in zip(names[0] + names[1], candidate["arm_target_rad"], strict=True)
        if "wrist" in name
    }
    candidate["palms_down_threshold_passed"] = all(
        candidate["metrics"][f"{side}_palm_down_error_deg"] <= 20.0
        and candidate["metrics"][f"{side}_actual_finger_inward_dot"] >= 0.75
        for side in ("left", "right")
    )
    candidate["static_head_threshold_passed"] = all(
        value >= 0.20 for key, value in candidate["metrics"].items() if key.endswith("head_gap_m")
    )
    candidate["comparison_head_threshold_passed"] = all(
        value >= args.comparison_clearance for key, value in candidate["metrics"].items() if key.endswith("head_gap_m")
    )
    head_top = max(candidate["metrics"][f"{side}_head_surface_top_m"] for side in ("left", "right"))
    candidate["overhead_height_targets_m"] = {
        "head_cover_top": head_top,
        "palm": head_top + 0.015,
        "fingertips": head_top,
    }
    candidate["orientation_passed"] = candidate["palms_down_threshold_passed"]
    candidate["overhead_height_passed"] = (
        candidate["metrics"]["palm_height_m"] >= head_top + 0.015
        and candidate["metrics"]["fingertip_height_m"] >= head_top
    )
    candidate["head_clearance_passed"] = candidate["comparison_head_threshold_passed"]
    candidate["wrist_margin_passed"] = min(candidate["wrist_hard_limit_reserve_rad"].values()) >= 0.12 - 1e-8
    candidate["wrist_downbend_passed"] = all(
        item["tip_z_change_from_roll_neutral_m"] <= -0.025 and item["finger_bend_from_roll_neutral_deg"] >= 35
        for item in candidate["wrist_flexion_diagnostics"].values()
    )
    candidate["fingertips_down_passed"] = all(
        candidate["metrics"][f"{side}_tip_below_palm_m"] >= 0.015
        and candidate["metrics"][f"{side}_actual_finger_direction_z"] < 0
        for side in ("left", "right")
    )
    candidate["fingertip_closure_passed"] = candidate["metrics"]["fingertip_span_m"] <= 0.10
    candidate["overall_shape_candidate"] = all(
        candidate[key]
        for key in (
            "orientation_passed",
            "overhead_height_passed",
            "head_clearance_passed",
            "wrist_margin_passed",
            "wrist_downbend_passed",
            "fingertips_down_passed",
            "fingertip_closure_passed",
        )
    )
    candidate["motion_enabled"] = False
    if not args.skip_render:
        try:
            candidate["render"] = render(root, trees, candidate, model, names, args.output)
        except Exception as exc:
            candidate["render_error"] = f"{type(exc).__name__}: {exc}"
    candidate["elapsed_s"] = time.monotonic() - started
    candidate_file = args.output / "overhead_heart_candidate.json"
    candidate_file.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    digest = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "candidate": str(candidate_file),
                "sha256": digest,
                "metrics": candidate["metrics"],
                "static_head_threshold_passed": candidate["static_head_threshold_passed"],
                "render": candidate.get("render"),
                "render_error": candidate.get("render_error"),
                "elapsed_s": candidate["elapsed_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
