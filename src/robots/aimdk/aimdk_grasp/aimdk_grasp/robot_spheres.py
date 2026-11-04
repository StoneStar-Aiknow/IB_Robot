"""Collision spheres of the torso-rooted X2 upper body at given joint states.

The spheres are the planner's own cover (``robot_model.collision_robot_config``),
placed by a level-batched forward kinematics of the generated planning model.
They feed the scene's self filter and the planner's start state; the planning
arm's reach bounds crop the scene and prefilter grasp candidates. A model plans
the arm its configuration names (``robot_model.model_side``); the other arm and
the head are locked. Collision checking
itself is cuRobo's, on the GPU.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from threading import RLock

import numpy as np
from scipy.spatial.transform import Rotation

from aimdk_grasp.hand import HandMapping
from aimdk_grasp.joints import DIP_FOLLOWERS, HAND_ALIASES, HEAD, arm_joints, check_side, other_side
from aimdk_grasp.robot_model import collision_robot_config, config_directory, model_side, planning_model

_EYE3 = np.eye(3)
_reach_bounds: dict[str, tuple] = {}
_reach_lock = RLock()


def reach_bounds(side: str):
    """Cached torso-frame (center, arm radius, palm radius) of ``side``'s planning model; see arm_reach/palm_reach."""
    with _reach_lock:
        if check_side(side) not in _reach_bounds:
            model = RobotSphereModel(directory=config_directory(side))
            center, arm = model.arm_reach()
            _reach_bounds[side] = center, arm, model.palm_reach()[1]
        return _reach_bounds[side]


def left_reach_bounds():
    return reach_bounds("left")


class RobotSphereModel:
    """``mapping`` supplies the hand limits; by default the planning model's.

    ``directory`` selects the planning model (default: the left arm's); its
    side is the moving arm, whose hand ``mapping`` must describe.
    """

    def __init__(self, mapping: HandMapping | None = None, directory: Path | None = None):
        self.side = model_side(directory)
        self.arm, self.other_arm = arm_joints(self.side), arm_joints(other_side(self.side))
        self.mapping = HandMapping(planning_model(directory), side=self.side) if mapping is None else mapping
        if self.mapping.side != self.side:
            raise ValueError(f"a {self.mapping.side} hand mapping cannot drive the {self.side} planning model")
        kin = collision_robot_config(directory)["robot_cfg"]["kinematics"]
        self.base = kin["base_link"]
        self.links = kin["collision_link_names"]
        self.joints = []
        reached = {self.base}
        pending = list(ET.parse(kin["urdf_path"]).findall("joint"))
        while pending:
            children = []
            for j in pending:
                parent_el = j.find("parent")
                if parent_el is not None and parent_el.attrib["link"] in reached:
                    children.append(j)
            if not children:
                break  # lower body is outside the torso-rooted planner model
            for j in children:
                pending.remove(j)
                parent_el, child_el = j.find("parent"), j.find("child")
                if parent_el is None or child_el is None:
                    raise ValueError("Joint parent/child required")
                parent, child = parent_el.attrib["link"], child_el.attrib["link"]
                if child in reached or j.attrib["type"] not in ("fixed", "revolute"):
                    raise ValueError("Unsupported collision-model joint tree")
                reached.add(child)
                origin = j.find("origin")
                if origin is None:
                    raise ValueError("Joint origin required")
                transform = np.eye(4)
                transform[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
                mimic, limit, axis_el = j.find("mimic"), j.find("limit"), j.find("axis")
                if j.attrib["type"] != "fixed" and (axis_el is None or limit is None):
                    raise ValueError("Revolute joint axis/limit required")
                axis = np.zeros(3) if j.attrib["type"] == "fixed" else np.fromstring(axis_el.attrib["xyz"], sep=" ")
                if j.attrib["type"] != "fixed" and not np.isclose(np.linalg.norm(axis), 1.0):
                    raise ValueError("Unit joint axis required")
                x, y, z = axis
                skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
                self.joints.append(
                    dict(
                        name=j.attrib["name"],
                        parent=parent,
                        child=child,
                        transform=transform,
                        skew=skew,
                        square=skew @ skew,
                        mimic=None if mimic is None else dict(mimic.attrib),
                        bounds=None if limit is None else (float(limit.attrib["lower"]), float(limit.attrib["upper"])),
                    )
                )
        if not set(self.links) <= reached:
            raise ValueError("Collision links missing from FK tree")
        self.local = np.array(
            [[*s["center"], s["radius"]] for link in self.links for s in kin["collision_spheres"][link]], dtype=float
        )
        self.sphere_links = [link for link in self.links for _ in kin["collision_spheres"][link]]
        self.local[:, 3] += kin["collision_sphere_buffer"]
        self.link_slices = {link: np.flatnonzero(np.array(self.sphere_links) == link) for link in self.links}
        self.link_indices = {link: index for index, link in enumerate(self.links)}
        # Level-batched FK tables. SOC0 ARM64 NumPy is per-call bound (r14
        # profile: ~800 tiny ops = 12 ms per tick), so every joint of one
        # tree depth is composed in a single stacked matmul. Frame 0 is the
        # base; frame k + 1 is the child of joint k.
        frame = {self.base: 0}
        frame.update({joint["child"]: k + 1 for k, joint in enumerate(self.joints)})
        self.fk_parent = np.array([frame[joint["parent"]] for joint in self.joints])
        depth = np.zeros(len(self.joints) + 1, dtype=int)
        for k, parent in enumerate(self.fk_parent):
            depth[k + 1] = depth[parent] + 1  # joints are discovered parent-first
        self.fk_levels = [np.flatnonzero(depth[1:] == level) for level in range(1, depth.max() + 1)]
        self.fk_transform = np.stack([joint["transform"] for joint in self.joints])
        self.fk_skew = np.stack([joint["skew"] for joint in self.joints])
        self.fk_square = np.stack([joint["square"] for joint in self.joints])
        # Joint angle = input[:, fk_source] * fk_scale + fk_offset over the
        # columns planning arm + other arm + HEAD + hand alias names; fixed
        # joints read column 0 scaled by zero.
        alias_names = list(HAND_ALIASES) + list(DIP_FOLLOWERS)
        self.fk_inputs = list(self.arm) + list(self.other_arm) + list(HEAD) + alias_names
        self.fk_alias_names = alias_names
        column = {name: index for index, name in enumerate(self.fk_inputs)}
        source, scale, offset = {}, {}, {}
        for joint in self.joints:
            name, mimic = joint["name"], joint["mimic"]
            if joint["bounds"] is None:
                source[name], scale[name], offset[name] = 0, 0.0, 0.0
            elif mimic is not None:
                parent = mimic["joint"]
                multiplier = float(mimic.get("multiplier", 1.0))
                source[name] = source[parent]
                scale[name] = scale[parent] * multiplier
                offset[name] = offset[parent] * multiplier + float(mimic.get("offset", 0.0))
            else:
                source[name], scale[name], offset[name] = column[name], 1.0, 0.0
        names = [joint["name"] for joint in self.joints]
        self.fk_source = np.array([source[name] for name in names])
        self.fk_scale = np.array([scale[name] for name in names])
        self.fk_offset = np.array([offset[name] for name in names])
        self.fk_bounded = np.flatnonzero([joint["bounds"] is not None for joint in self.joints])
        self.fk_lower = np.array([self.joints[k]["bounds"][0] for k in self.fk_bounded]) - 1e-6
        self.fk_upper = np.array([self.joints[k]["bounds"][1] for k in self.fk_bounded]) + 1e-6
        self.link_frames = np.array([frame[link] for link in self.links])
        self.sphere_link_index = np.array([self.link_indices[link] for link in self.sphere_links])

    def left_arm_reach(self):
        if self.side != "left":
            raise ValueError("left_arm_reach of a right planning model")
        return self.arm_reach()

    def left_palm_reach(self):
        if self.side != "left":
            raise ValueError("left_palm_reach of a right planning model")
        return self.palm_reach()

    def arm_reach(self):
        """Torso-frame (center, radius) bounding every planning-arm/hand sphere at any joint state.

        Center is the first planning-arm joint origin; joints upstream of it are fixed
        in this torso-rooted model. By the triangle inequality a link origin lies
        within the summed joint-offset lengths from it, so radius is an upper
        bound (not a sample maximum) including sphere offsets, radii and buffer.
        """
        center, bounds = self._arm_link_bounds()
        moving = [index for index, link in enumerate(self.sphere_links) if link in bounds]
        radius = max(
            bounds[self.sphere_links[index]] + np.linalg.norm(self.local[index, :3]) + self.local[index, 3]
            for index in moving
        )
        return center.copy(), float(radius)

    def palm_reach(self):
        """Torso-frame (center, radius) bounding the R_palm origin at any joint state.

        Same triangle-inequality bound as arm_reach, for the palm frame
        that grasp goals are expressed in. A goal outside it has no IK solution.
        """
        center, bounds = self._arm_link_bounds()
        return center, float(bounds["R_palm"])

    def _arm_link_bounds(self):
        """Shoulder center and, per moving link, a bound on its origin's distance from it."""
        joints = {joint["child"]: joint for joint in self.joints}
        shoulder = next(joint for joint in self.joints if joint["name"] == self.arm[0])
        transform, link = np.eye(4), shoulder["parent"]
        while link != self.base:
            joint = joints[link]
            transform, link = joint["transform"] @ transform, joint["parent"]
        center = (transform @ shoulder["transform"])[:3, 3]
        # A link's origin is a fixed offset from the nearest frame a revolute joint
        # turns (its anchor); fixed joints in between are composed first, so a
        # rigid chain (wrist -> palm -> finger base) adds one offset, not several.
        bounds, anchors = {shoulder["child"]: 0.0}, {shoulder["child"]: (shoulder["child"], np.eye(4))}
        pending = [shoulder["child"]]
        while pending:
            parent = pending.pop()
            anchor, rigid = anchors[parent]
            for joint in self.joints:
                if joint["parent"] == parent:
                    offset = rigid @ joint["transform"]
                    child = joint["child"]
                    bounds[child] = bounds[anchor] + np.linalg.norm(offset[:3, 3])
                    anchors[child] = (anchor, offset) if joint["bounds"] is None else (child, np.eye(4))
                    pending.append(child)
        return center.copy(), bounds

    def spheres(self, feedback, positions, *, other=None, head=None, hand=None):
        """Torso-frame spheres ``(N, S, 4)`` for N planning-arm states, and whether each
        state is inside every joint's limits.

        The other arm and head come from ``feedback`` unless given (per state or
        shared); the hand from ``feedback.hand(side)`` unless ``hand`` gives a native
        (10,) pose or one per state. ``last_link_frames`` keeps the link frames.
        """
        q = np.asarray(positions, dtype=float)
        if q.ndim != 2 or q.shape[1] != 7 or not len(q) or not np.isfinite(q).all():
            raise ValueError(f"Finite Nx7 {self.side}-arm positions required")
        body_columns = [q]
        for names, vector in (
            (self.other_arm, feedback.vector(self.other_arm) if other is None else other),
            (HEAD, feedback.vector(HEAD) if head is None else head),
        ):
            vector = np.asarray(vector, float)
            if vector.shape not in ((len(names),), (len(q), len(names))) or not np.isfinite(vector).all():
                raise ValueError("Invalid other-arm/head state")
            if vector.ndim == 1:
                vector = np.broadcast_to(vector, (len(q), len(names)))
            else:
                vector = np.asarray(vector)
            body_columns.append(vector)
        n = len(q)
        if hand is None:
            hand = feedback.hand(self.side)
        hand = np.asarray(hand, float)
        if hand.ndim == 1:
            alias = self.mapping.kinematic_alias(hand)
            hand_columns = np.broadcast_to([alias[name] for name in self.fk_alias_names], (n, len(self.fk_alias_names)))
        elif hand.shape == (n, 10):
            # The measured/target hand pattern repeats row-wise; alias each
            # distinct row once instead of every state (r48 profile: 26
            # polynomial evaluations per tick for two distinct hands).
            unique_hands, hand_map = self._stable_unique(hand)
            aliases = [self.mapping.kinematic_alias(row) for row in unique_hands]
            hand_columns = np.array([[row[name] for name in self.fk_alias_names] for row in aliases])[hand_map]
        else:
            raise ValueError("Invalid hand state dimensions")
        inputs = np.concatenate(body_columns + [hand_columns], axis=1)
        angle = inputs[:, self.fk_source] * self.fk_scale + self.fk_offset
        bounded = angle[:, self.fk_bounded]
        in_bounds = np.all((bounded >= self.fk_lower) & (bounded <= self.fk_upper), axis=1)
        sin, cos = np.sin(angle)[..., None, None], (1 - np.cos(angle))[..., None, None]
        local = np.broadcast_to(self.fk_transform, (n, *self.fk_transform.shape)).copy()
        local[..., :3, :3] = self.fk_transform[:, :3, :3] @ (_EYE3 + sin * self.fk_skew + cos * self.fk_square)
        world = np.empty((n, len(self.joints) + 1, 4, 4))
        world[:, 0] = np.eye(4)
        for level in self.fk_levels:
            world[:, level + 1] = world[:, self.fk_parent[level]] @ local[:, level]
        link = world[:, self.link_frames, :3]
        # Explicit column products: stacked matmul/einsum over (n, 788) small
        # matrices measured 2-4x slower than these elementwise ops.
        placed = link[:, self.sphere_link_index]
        output = np.empty((n, len(self.local), 4))
        output[..., :3] = (
            placed[..., 0] * self.local[:, None, 0]
            + placed[..., 1] * self.local[:, None, 1]
            + placed[..., 2] * self.local[:, None, 2]
            + placed[..., 3]
        )
        output[..., 3] = self.local[:, 3]
        self.last_link_frames = link
        return output, in_bounds

    @staticmethod
    def _stable_unique(rows):
        # Byte-order np.unique instead of pairwise array_equal: every caller
        # validates finiteness first, so byte equality matches numeric
        # equality while turning O(N^2) Python comparisons into a sort.
        viewed = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
        _, index, inverse = np.unique(viewed, return_index=True, return_inverse=True)
        order = np.argsort(index, kind="stable")
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order))
        return rows[index[order]], rank[inverse]
