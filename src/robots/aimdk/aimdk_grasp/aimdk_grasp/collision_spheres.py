"""Conservative compression of a certified sphere cover on one rigid link."""

import numpy as np


def merge_sphere_cover(spheres, max_radius_m=0.06, *, protected_spheres=None, clearance_m=0.01):
    """Merge the nearest pair while its enclosing sphere fits the radius cap.

    Each replacement contains both complete input balls. By induction the
    resulting union contains the original filled cover, including its interior.
    This does not rely on surface sampling or relax self-collision exclusions.
    """
    balls = np.array([[*s["center"], s["radius"]] for s in spheres], dtype=float)
    if (
        balls.ndim != 2
        or balls.shape[1] != 4
        or not len(balls)
        or not np.isfinite(balls).all()
        or np.any(balls[:, 3] <= 0)
        or not np.isfinite(max_radius_m)
        or max_radius_m <= 0
    ):
        raise ValueError("Finite positive sphere radii and radius cap required")
    protected = np.empty((0, 4)) if protected_spheres is None else np.asarray(protected_spheres, dtype=float)
    if (
        protected.ndim != 2
        or protected.shape[1] != 4
        or not np.isfinite(protected).all()
        or not np.isfinite(clearance_m)
        or clearance_m < 0
    ):
        raise ValueError("Finite Nx4 protected spheres and nonnegative clearance required")
    protected = protected[protected[:, 3] > 0]
    while len(balls) > 1:
        distance = np.linalg.norm(balls[:, None, :3] - balls[None, :, :3], axis=-1)
        radius = np.maximum(
            (distance + balls[:, None, 3] + balls[None, :, 3]) / 2, np.maximum(balls[:, None, 3], balls[None, :, 3])
        )
        radius[np.tril_indices(len(balls))] = np.inf
        # Keep the original cover around locked bodies and the requested work
        # envelope. Reject replacements, not inputs: their inflation is the risk.
        if len(protected):
            for i, j in zip(*np.where(radius <= max_radius_m), strict=True):
                a, b, r = balls[i], balls[j], radius[i, j]
                fraction = np.clip((r - a[3]) / max(distance[i, j], 1e-30), 0.0, 1.0)
                center = a[:3] + (b[:3] - a[:3]) * fraction
                if np.any(np.linalg.norm(protected[:, :3] - center, axis=1) <= protected[:, 3] + r + clearance_m):
                    radius[i, j] = np.inf
        i, j = np.unravel_index(np.argmin(radius), radius.shape)
        if radius[i, j] > max_radius_m:
            break
        a, b = balls[i], balls[j]
        r = radius[i, j]
        if a[3] >= distance[i, j] + b[3]:
            center = a[:3]
        elif b[3] >= distance[i, j] + a[3]:
            center = b[:3]
        else:
            center = a[:3] + (b[:3] - a[:3]) * ((r - a[3]) / distance[i, j])
        # Include numerical roundoff, too: do not shrink the certified union.
        r = np.nextafter(max(np.linalg.norm(center - a[:3]) + a[3], np.linalg.norm(center - b[:3]) + b[3]), np.inf)
        balls[i] = [*center, r]
        balls = np.delete(balls, j, axis=0)
    return [{"center": b[:3].tolist(), "radius": float(b[3])} for b in balls]
