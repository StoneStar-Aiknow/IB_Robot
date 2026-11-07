"""CPU convex separation and gravity for an offline-exported welded X2 model.

Support planes prove clearance; unresolved GJK iterations refuse the path.
This module has no dependency on the engine used to export the model.
"""

from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation


def closest_simplex(points):
    best = None
    for count in range(1, min(4, len(points)) + 1):
        for indices in combinations(range(len(points)), count):
            vertices = points[list(indices)]
            if count == 1:
                weights = np.ones(1)
            else:
                edges = (vertices[1:] - vertices[0]).T
                tail = np.linalg.lstsq(edges, -vertices[0], rcond=None)[0]
                weights = np.r_[1 - tail.sum(), tail]
            if weights.min() < -1e-10:
                continue
            point = weights @ vertices
            distance = float(point @ point)
            if best is None or distance < best[0]:
                best = distance, point, vertices[weights > 1e-10]
    return best[1], best[2]


def separated(a, b, clearance, *, initial=None, remember=lambda _: None):
    """True only with a separating plane at least clearance apart."""
    direction = a.center - b.center if initial is None else initial
    if np.linalg.norm(direction) < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    simplex = np.empty((0, 3))
    for _ in range(64):
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            return False
        vertex = a.support(-direction) - b.support(direction)
        if float(direction @ vertex) / norm >= clearance + 1e-9:
            remember(direction)
            return True
        if len(simplex) and np.min(np.linalg.norm(simplex - vertex, axis=1)) < 1e-10:
            return False
        direction, simplex = closest_simplex(np.vstack([simplex, vertex]))
        if np.linalg.norm(direction) < clearance:
            return False
    return False


class Shape:
    def __init__(self, record, transform):
        self.record = record
        self.center = transform[:3, 3]
        self.rotation = transform[:3, :3]
        self.radius = record["radius"]
        self.vertices = None
        if record["kind"] == "mesh":
            self.vertices = np.asarray(record["vertices"]) @ self.rotation.T + self.center

    def support(self, direction):
        if self.vertices is not None:
            return self.vertices[np.argmax(self.vertices @ direction)]
        # The exported head is a cylinder; refuse all other primitive types.
        local = self.rotation.T @ direction
        size = self.record["size"]
        radial = np.linalg.norm(local[:2])
        point = np.r_[local[:2] * (size[0] / radial if radial > 1e-12 else 0), size[1] if local[2] >= 0 else -size[1]]
        return self.center + self.rotation @ point


class CompiledGeometry:
    def __init__(self, data):
        self.data = data
        self.chains = []
        self.body_transforms = []
        self._directions = {}
        for index, body in enumerate(data["bodies"]):
            parent = body["parent"]
            if index and not 0 <= parent < index:
                raise ValueError("invalid compiled body tree")
            chain = dict(self.chains[parent]) if index else {}
            # Reach bounds are derived per shape below, rather than using a global arm radius.
            for joint in body["joints"]:
                chain[joint["name"]] = index
            self.chains.append(chain)
            self.body_transforms.append(np.asarray(body["transform"], float))
        self.levers = []
        for shape in data["shapes"]:
            if shape["kind"] not in {"mesh", "cylinder"}:
                raise ValueError("unsupported compiled collision primitive")
            radius = float(shape["radius"]) + np.linalg.norm(np.asarray(shape["transform"])[:3, 3])
            index = shape["body"]
            levers = {}
            while index:
                body = data["bodies"][index]
                for joint in body["joints"]:
                    levers[joint["name"]] = radius + np.linalg.norm(joint["position"])
                radius += np.linalg.norm(self.body_transforms[index][:3, 3])
                index = body["parent"]
            self.levers.append(levers)
        self._descendants = {
            name: [i for i, chain in enumerate(self.chains) if name in chain]
            for name in {name for chain in self.chains for name in chain}
        }
        self._centers = np.asarray([b["com"] for b in data["bodies"]])
        self._forces = np.array([[0.0, 0.0, -9.81 * b["mass"]] for b in data["bodies"]])
        # Mesh vertices/transforms are immutable model data. Converting their
        # JSON lists at every trajectory sample dominated FK on the 310P.
        self._shapes = []
        for shape in data["shapes"]:
            record = {**shape, "transform": np.asarray(shape["transform"], float)}
            if shape["kind"] == "mesh":
                record["vertices"] = np.asarray(shape["vertices"], float)
            self._shapes.append(record)
        self._descendant_masks = {
            name: np.isin(np.arange(len(data["bodies"])), indices) for name, indices in self._descendants.items()
        }

    def forward(self, q):
        transforms, axes, origins = [], {}, {}
        for i, body in enumerate(self.data["bodies"]):
            transform = self.body_transforms[i].copy()
            if i:
                transform = transforms[body["parent"]] @ transform
            for joint in body["joints"]:
                pos, axis = np.asarray(joint["position"]), np.asarray(joint["axis"])
                origins[joint["name"]] = transform[:3, 3] + transform[:3, :3] @ pos
                axes[joint["name"]] = transform[:3, :3] @ axis
                rotate = Rotation.from_rotvec(axis * q.get(joint["name"], 0.0)).as_matrix()
                local = np.eye(4)
                local[:3, :3], local[:3, 3] = rotate, pos - rotate @ pos
                transform = transform @ local
            transforms.append(transform)
        shapes = [Shape(s, transforms[s["body"]] @ s["transform"]) for s in self._shapes]
        torques = dict.fromkeys(q, 0.0)
        transforms = np.asarray(transforms)
        centers = transforms[:, :3, 3] + np.einsum("nij,nj->ni", transforms[:, :3, :3], self._centers)
        names = list(torques.keys() & self._descendants.keys())
        if names:
            offsets = centers[None, :, :] - np.asarray([origins[n] for n in names])[:, None, :]
            moments = np.cross(offsets, self._forces[None, :, :])
            values = -np.einsum(
                "njc,nc,nj->n",
                moments,
                np.asarray([axes[n] for n in names]),
                np.asarray([self._descendant_masks[n] for n in names]),
            )
            torques.update(zip(names, map(float, values), strict=True))
        return shapes, torques

    def check(self, q, delta=None, *, path=False):
        shapes, torques = self.forward(q)
        delta = {} if delta is None else delta
        for a, b, margin, label in self.data["pairs"]:
            left, right = self.levers[a], self.levers[b]
            error = sum(left[n] * delta.get(n, 0.0) for n in left.keys() - right.keys())
            error += sum(right[n] * delta.get(n, 0.0) for n in right.keys() - left.keys())
            needed = max(0.0, margin - (0.015 if path else 0.0)) + error
            sa, sb = shapes[a], shapes[b]
            if np.linalg.norm(sa.center - sb.center) - sa.radius - sb.radius > needed:
                continue
            if not separated(
                sa,
                sb,
                needed,
                initial=self._directions.get((a, b)),
                remember=lambda value, key=(a, b): self._directions.__setitem__(key, value),
            ):
                return False, label, torques
        return True, "", torques

    def check_path(
        self,
        joints,
        positions,
        *,
        head=None,
        guard=lambda: None,
        start_tolerance_rad=0.0,
        uncertainty=None,
        head_radius=0.005,
        bound_gravity=False,
    ):
        if not np.isfinite(start_tolerance_rad) or start_tolerance_rad < 0:
            raise ValueError("start tolerance must be finite and nonnegative")
        positions = np.asarray(positions)
        uncertainty = np.zeros(len(joints)) if uncertainty is None else np.asarray(uncertainty, float)
        if (
            uncertainty.shape != (len(joints),)
            or not np.isfinite(uncertainty).all()
            or (uncertainty < 0).any()
            or not np.isfinite(head_radius)
            or head_radius < 0
        ):
            raise ValueError("invalid path uncertainty")
        gravity_upper = dict.fromkeys(joints, 0.0)
        # A minimum-jerk fixed move follows one joint-space line. Its timing
        # samples do not add geometry; keep the complete line, with the same
        # conservative interpolation bounds, instead of checking endpoints twice.
        if len(positions) > 2:
            direction = positions[-1] - positions[0]
            index = int(np.argmax(np.abs(direction)))
            if abs(direction[index]) > 1e-12:
                fractions = (positions[:, index] - positions[0, index]) / direction[index]
                if (
                    np.all(np.diff(fractions) >= 0)
                    and np.all((fractions >= 0) & (fractions <= 1))
                    and np.allclose(positions, positions[0] + fractions[:, None] * direction, atol=1e-12, rtol=0)
                ):
                    positions = positions[[0, -1]]
        for a, b in zip(positions[:-1], positions[1:], strict=True):
            count = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.02)))
            delta = dict(zip(joints, np.abs(b - a) / (2 * count) + start_tolerance_rad + uncertainty, strict=True))
            delta.update({"head_yaw_joint": head_radius, "head_pitch_joint": head_radius})
            for row in np.linspace(a, b, count + 1):
                guard()
                q = {**dict(zip(joints, row, strict=True)), **(head or {})}
                ok, label, torques = self.check(q, delta, path=True)
                if not ok:
                    raise ValueError(f"compiled body clearance failed: {label}")
                errors = self.gravity_error(delta) if bound_gravity else {}
                for name in joints:
                    suffix = name.split("_", 1)[1].removesuffix("_joint")
                    effort = (
                        36
                        if suffix in ("shoulder_pitch", "shoulder_roll")
                        else 4.8
                        if suffix in ("wrist_pitch", "wrist_roll")
                        else 24
                    )
                    upper = abs(torques[name]) + errors.get(name, 0.0)
                    gravity_upper[name] = max(gravity_upper[name], upper)
                    if upper > 0.29 * effort:
                        raise ValueError(f"compiled gravity limit: {name}")
        return gravity_upper

    def gravity_error(self, delta):
        """Global Lipschitz enclosure for gravity torque over a joint box.

        For tau = axis dot (r cross force), use |d tau| <= mg*(|dr| + R*|d axis|).
        Reach bounds sum fixed link lengths, so hold for all rotations. This
        first exporter supports one zero-pivot hinge per body; refuse other
        models rather than assuming a bound for them.
        """
        if any(
            len(b["joints"]) > 1 or any(np.linalg.norm(j["position"]) > 1e-12 for j in b["joints"])
            for b in self.data["bodies"]
        ):
            raise ValueError("gravity enclosure requires single zero-pivot hinges")

        def reach(index, local):
            radius, result = float(np.linalg.norm(local)), {}
            while index:
                body = self.data["bodies"][index]
                result.update({j["name"]: radius for j in body["joints"]})
                radius += float(np.linalg.norm(self.body_transforms[index][:3, 3]))
                index = body["parent"]
            return result

        centers = [reach(i, b["com"]) for i, b in enumerate(self.data["bodies"])]
        result = {}
        for name, descendants in self._descendants.items():
            body = next(i for i, b in enumerate(self.data["bodies"]) if any(j["name"] == name for j in b["joints"]))
            origins = reach(body, [0, 0, 0])
            origin_error = sum(r * delta.get(n, 0) for n, r in origins.items())
            axis_error = sum(delta.get(n, 0) for n in self.chains[body])
            result[name] = sum(
                9.81
                * self.data["bodies"][i]["mass"]
                * (
                    origin_error
                    + sum(r * delta.get(n, 0) for n, r in centers[i].items())
                    + centers[i][name] * axis_error
                )
                for i in descendants
            )
        return result
