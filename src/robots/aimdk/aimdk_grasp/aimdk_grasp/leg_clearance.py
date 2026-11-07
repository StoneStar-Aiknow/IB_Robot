"""CPU distance to the verified standing-leg convex solids.

Boxes/planes only prove separation. Near cases use triangle distances to the
convex solid, including face, edge, vertex and interior cases. No MuJoCo.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull

# Preserve the previous box expansion as an isotropic uncertainty on the hull.
# This covers the exported mesh grid and placement error, not joint motion.
LEG_MODEL_PADDING_M = 0.01


class ConvexLeg:
    def __init__(self, vertices):
        vertices = np.asarray(vertices, float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("leg hull requires finite 3D vertices")
        hull = ConvexHull(vertices)
        centre = vertices.mean(axis=0)
        self.axes = np.linalg.svd(vertices - centre, full_matrices=False)[2]
        local = (vertices - centre) @ self.axes.T
        low, high = local.min(axis=0), local.max(axis=0)
        self.centre = centre + self.axes.T @ ((low + high) / 2)
        self.half_size = (high - low) / 2
        self.normals, self.offsets = hull.equations[:, :3], hull.equations[:, 3]
        self.triangles = vertices[hull.simplices]
        self.a = self.triangles[:, 0]
        self.ab = self.triangles[:, 1] - self.a
        self.ac = self.triangles[:, 2] - self.a
        self.aa = np.einsum("ij,ij->i", self.ab, self.ab)
        self.cc = np.einsum("ij,ij->i", self.ac, self.ac)
        self.ac_dot = np.einsum("ij,ij->i", self.ab, self.ac)
        self.determinant = self.aa * self.cc - self.ac_dot**2
        if np.any(self.determinant <= 0):
            raise ValueError("degenerate leg hull triangles")

    def lower_bound(self, spheres):
        """Conservative sphere-to-solid clearance; cheap rejection-free broad phase."""
        spheres = np.asarray(spheres, float)
        points, radii = spheres[..., :3], spheres[..., 3]
        gap = np.abs((points - self.centre) @ self.axes.T) - self.half_size
        return np.linalg.norm(np.maximum(gap, 0), axis=-1) + np.minimum(np.max(gap, axis=-1), 0) - radii

    def clearance(self, spheres):
        """Signed distance to a convex solid minus sphere radius (metres)."""
        spheres = np.asarray(spheres, float)
        if spheres.ndim != 2 or spheres.shape[1] != 4 or not np.isfinite(spheres).all():
            raise ValueError("leg distance requires finite Nx4 spheres")
        if np.any(spheres[:, 3] < 0):
            raise ValueError("sphere radius must be nonnegative")
        out = np.empty(len(spheres))
        for start in range(0, len(spheres), 64):
            balls = spheres[start : start + 64]
            points = balls[:, :3]
            planes = points @ self.normals.T + self.offsets
            signed = np.max(planes, axis=1)
            outside = signed > 0
            if outside.any():
                p = points[outside]
                ap = p[:, None, :] - self.a
                u = np.einsum("pfi,fi->pf", ap, self.ab)
                v = np.einsum("pfi,fi->pf", ap, self.ac)
                s = (self.cc * u - self.ac_dot * v) / self.determinant
                t = (self.aa * v - self.ac_dot * u) / self.determinant
                # Projection onto the face plane is closest only inside its
                # triangle. Otherwise the closest point lies on an edge.
                interior = (s >= 0) & (t >= 0) & (s + t <= 1)
                distance2 = np.where(interior, planes[outside] ** 2, np.inf)
                for i, j in ((0, 1), (1, 2), (2, 0)):
                    a = self.triangles[:, i]
                    edge = self.triangles[:, j] - a
                    delta = p[:, None, :] - a
                    fraction = np.einsum("pfi,fi->pf", delta, edge) / np.einsum("fi,fi->f", edge, edge)
                    delta -= np.clip(fraction, 0, 1)[..., None] * edge
                    distance2 = np.minimum(distance2, np.einsum("pfi,pfi->pf", delta, delta))
                signed[outside] = np.sqrt(np.maximum(0, np.min(distance2, axis=1)))
            out[start : start + len(balls)] = signed - balls[:, 3]
        return out


def load_leg_hulls(side):
    """One convex solid per connected exported link, retaining coarse-box names."""
    from .leg_obstacle import leg_obstacle
    from .robot_model import config_directory

    mesh = leg_obstacle(config_directory(side))  # Checks the source URDF/mesh hashes.
    vertices = np.asarray(mesh["vertices"])
    faces = np.asarray(mesh["faces"]).reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]]])
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2)
    _, labels = connected_components(adjacency, directed=False)
    return {f"{side}_leg_{label}": ConvexLeg(vertices[labels == label]) for label in np.unique(labels)}
