"""Foot-ground evidence, separate from root displacement or visual impressions."""

import mujoco
import numpy as np


class GaitMonitor:
    def __init__(self, sim):
        self.sim = sim
        self.feet = [sim.model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.geoms = [
            np.flatnonzero((sim.model.geom_bodyid == body) & (sim.model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
            for body in self.feet
        ]
        if any(len(group) == 0 for group in self.geoms):
            raise ValueError("Gait measurement requires spherical foot collision pads")
        self.samples = []

    def sample(self):
        model, data = self.sim.model, self.sim.data
        normal = np.zeros(2)
        weighted_speed = np.zeros(2)
        force = np.zeros(6)
        jacobian = np.zeros((3, model.nv))
        for index, contact in enumerate(data.contact):
            first, second = model.geom_bodyid[contact.geom]
            if contact.efc_address < 0 or 0 not in (first, second):
                continue
            body = second if first == 0 else first
            if body not in self.feet:
                continue
            foot = self.feet.index(body)
            mujoco.mj_contactForce(model, data, index, force)
            load = max(0.0, float(force[0]))
            mujoco.mj_jac(model, data, jacobian, None, contact.pos, body)
            normal[foot] += load
            weighted_speed[foot] += load * np.linalg.norm((jacobian @ data.qvel)[:2])
        clearance = [float(np.min(data.geom_xpos[g, 2] - model.geom_size[g, 0])) for g in self.geoms]
        self.samples.append(
            {
                "clearance": clearance,
                "load": normal.tolist(),
                "slip": np.divide(weighted_speed, normal, out=np.zeros(2), where=normal > 0).tolist(),
            }
        )

    def summary(self):
        # Exclude acceleration/transient first 20%; all measurements are at 50 Hz.
        samples = self.samples[len(self.samples) // 5 :]
        if not samples:
            return {"passed": False, "reason": "No foot samples"}
        support = np.array([s["load"] for s in samples]) > 5.0
        result = {
            "passed": True,
            "sample_rate_hz": 50,
            "minimum_clearance_m": 0.015,
            "single_support_fraction": float(np.mean(np.sum(support, axis=1) == 1)),
        }
        for foot, side in enumerate(("left", "right")):
            clearance = np.array([s["clearance"][foot] for s in samples])
            swing = (~support[:, foot]) & (clearance > 0.003)
            # At least two consecutive samples avoids counting contact chatter as a step.
            sustained = swing[:-1] & swing[1:]
            swings = int(np.count_nonzero(sustained & ~np.r_[False, sustained[:-1]]))
            loaded_slip = [s["slip"][foot] for s in samples if s["load"][foot] > 100]
            result[side] = {
                "swing_count": swings,
                "max_clearance_m": float(clearance.max()),
                "contact_fraction": float(support[:, foot].mean()),
                "loaded_slip_median_m_s": float(np.median(loaded_slip)) if loaded_slip else None,
                "loaded_slip_p95_m_s": float(np.percentile(loaded_slip, 95)) if loaded_slip else None,
            }
            if swings < 3 or clearance.max() < result["minimum_clearance_m"]:
                result["passed"] = False
            if not loaded_slip or np.median(loaded_slip) > 0.02 or np.percentile(loaded_slip, 95) > 0.1:
                result["passed"] = False
        if result["single_support_fraction"] < 0.1:
            result["passed"] = False
        result["classification"] = "stepping" if result["passed"] else "shallow_shuffle_or_sliding"
        return result
