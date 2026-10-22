"""Numpy forward kinematics for the X2 MJCF model (stands in for mujoco's mj_forward)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

# Attributes that would move a frame but are not implemented here: refuse rather than ignore.
_UNSUPPORTED_BODY_ATTRS = ("axisangle", "xyaxes", "zaxis")
_UNSUPPORTED_ELEMENTS = ("frame", "include", "replicate", "attach", "composite", "flexcomp")
_KINEMATIC_JOINT_DEFAULTS = ("type", "pos", "axis", "ref", "range")


def _vec(element, attr, default):
    text = element.get(attr)
    if text is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in text.split()], dtype=np.float64)


def _quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _axis_angle_mat(axis, angle):
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _euler_to_mat(angles, seq):
    # mujoco: lowercase = intrinsic (post-multiply), uppercase = extrinsic (pre-multiply).
    rot = np.eye(3)
    for angle, ch in zip(angles, seq):
        r = _axis_angle_mat(np.eye(3)["xyz".index(ch.lower())], angle)
        rot = rot @ r if ch.islower() else r @ rot
    return rot


def _transform(rot, pos):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


class MjcfKinematics:
    """Numpy forward kinematics over an MJCF body tree; the subset of mujoco X2Kinematics needs."""

    def __init__(self, xml_path, base_qpos=(0.0, 0.0, 0.95, 1.0, 0.0, 0.0, 0.0)):
        root = ET.parse(str(xml_path)).getroot()
        compiler = root.find("compiler")
        compiler = compiler if compiler is not None else ET.Element("compiler")
        if compiler.get("angle", "degree") != "radian":
            raise ValueError("only compiler angle='radian' is supported")
        if compiler.get("fusestatic", "false") == "true":
            raise ValueError("compiler fusestatic is not supported")
        self._eulerseq = compiler.get("eulerseq", "xyz")
        for default in root.iter("default"):
            for joint in default.findall("joint"):
                bad = [a for a in _KINEMATIC_JOINT_DEFAULTS if a in joint.attrib]
                if bad:
                    raise ValueError(f"joint defaults with kinematic attributes {bad} are not supported")
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF has no <worldbody>")

        base_qpos = np.asarray(base_qpos, dtype=np.float64)
        # Bodies in breadth-first order; index 0 is the world.
        self._body_index = {"world": 0}
        parents = [-1]
        static = [np.eye(4)]
        depths = [0]
        hinge_body, hinge_pos, hinge_axis = [], [], []
        self._joint_index = {}   # hinge name -> hinge slot
        self._ranges = {}        # every joint name -> (2,)
        queue = [(worldbody, 0)]
        while queue:
            parent_element, parent = queue.pop(0)
            for element in parent_element:
                if element.tag in _UNSUPPORTED_ELEMENTS:
                    raise ValueError(f"<{element.tag}> is not supported")
                if element.tag != "body":
                    continue
                index = len(parents)
                name = element.get("name", f"body{index}")
                self._body_index[name] = index
                parents.append(parent)
                depths.append(depths[parent] + 1)
                static.append(self._body_frame(element, name))
                joints = [j for j in element if j.tag in ("joint", "freejoint")]
                if len(joints) > 1:
                    raise ValueError(f"body {name!r}: more than one joint is not supported")
                for joint in joints:
                    jname = joint.get("name")
                    jtype = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
                    if float(joint.get("ref", "0")) != 0.0:
                        raise ValueError(f"joint {jname!r}: non-zero ref is not supported")
                    self._ranges[jname] = _vec(joint, "range", (0.0, 0.0))
                    if jtype == "free":
                        if parent != 0:
                            raise ValueError(f"free joint {jname!r} must be on a top-level body")
                        # mujoco: free-joint qpos replaces the body's own pos/quat.
                        static[index] = _transform(_quat_to_mat(base_qpos[3:7]), base_qpos[:3])
                    elif jtype == "hinge":
                        axis = _vec(joint, "axis", (0.0, 0.0, 1.0))
                        self._joint_index[jname] = len(hinge_body)
                        hinge_body.append(index)
                        hinge_pos.append(_vec(joint, "pos", (0.0, 0.0, 0.0)))
                        hinge_axis.append(axis / np.linalg.norm(axis))
                    else:
                        raise ValueError(f"joint {jname!r}: type {jtype!r} is not supported")
                queue.append((element, index))

        self._hinge_body = np.array(hinge_body, dtype=np.intp)
        self._hinge_pos = np.array(hinge_pos).reshape(-1, 3)
        self._hinge_axis = np.array(hinge_axis).reshape(-1, 3)
        self._hinge_offcenter = bool(np.any(self._hinge_pos))
        k = np.zeros((len(hinge_body), 3, 3))
        a = self._hinge_axis
        k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
        k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = a[:, 2], -a[:, 1], a[:, 0]
        self._k = k
        self._k2 = k @ k
        self._static = np.array(static)
        self._static_hinge = self._static[self._hinge_body]
        self._local = self._static.copy()
        self._joint_tf = np.tile(np.eye(4), (len(hinge_body), 1, 1))
        self._world = np.tile(np.eye(4), (len(parents), 1, 1))
        # Breadth-first order makes every depth a contiguous slice.
        depths = np.array(depths)
        parents = np.array(parents, dtype=np.intp)
        self._levels = []
        for depth in range(1, int(depths.max()) + 1 if len(depths) > 1 else 1):
            idx = np.flatnonzero(depths == depth)
            self._levels.append((slice(int(idx[0]), int(idx[-1]) + 1), parents[idx]))
        self._q = np.zeros(len(hinge_body))
        self.forward({})

    def _body_frame(self, element, name):
        bad = [a for a in _UNSUPPORTED_BODY_ATTRS if a in element.attrib]
        if bad:
            raise ValueError(f"body {name!r}: orientation attributes {bad} are not supported")
        if "quat" in element.attrib and "euler" in element.attrib:
            raise ValueError(f"body {name!r}: both quat and euler given")
        if "euler" in element.attrib:
            rot = _euler_to_mat(_vec(element, "euler", (0.0, 0.0, 0.0)), self._eulerseq)
        else:
            rot = _quat_to_mat(_vec(element, "quat", (1.0, 0.0, 0.0, 0.0)))
        return _transform(rot, _vec(element, "pos", (0.0, 0.0, 0.0)))

    def joint_range(self, joint_name) -> np.ndarray:
        return self._ranges[joint_name].copy()

    def has_joint(self, name) -> bool:
        return name in self._ranges

    def has_body(self, name) -> bool:
        return name in self._body_index

    def forward(self, joint_values) -> None:
        q = self._q
        q[:] = 0.0
        index = self._joint_index
        for name, value in joint_values.items():
            q[index[name]] = value
        s = np.sin(q)[:, None, None]
        c = np.cos(q)[:, None, None]
        rot = self._k * s
        rot += self._k2 * (1.0 - c)
        rot[:, 0, 0] += 1.0
        rot[:, 1, 1] += 1.0
        rot[:, 2, 2] += 1.0
        tf = self._joint_tf
        tf[:, :3, :3] = rot
        if self._hinge_offcenter:
            # Rotate about the anchor: x -> R (x - p) + p.
            tf[:, :3, 3] = self._hinge_pos - np.einsum("nij,nj->ni", rot, self._hinge_pos)
        self._local[self._hinge_body] = np.matmul(self._static_hinge, tf)
        world, local = self._world, self._local
        for sl, parent in self._levels:
            np.matmul(world[parent], local[sl], out=world[sl])

    def _hinge(self, joint_name):
        return self._hinge_body[self._joint_index[joint_name]], self._joint_index[joint_name]

    def anchor(self, joint_name) -> np.ndarray:
        body, j = self._hinge(joint_name)
        tf = self._world[body]
        return tf[:3, :3] @ self._hinge_pos[j] + tf[:3, 3]

    def axis(self, joint_name) -> np.ndarray:
        body, j = self._hinge(joint_name)
        return self._world[body, :3, :3] @ self._hinge_axis[j]

    def body_pose(self, body_name):
        tf = self._world[self._body_index[body_name]]
        return tf[:3, 3].copy(), tf[:3, :3].copy()
