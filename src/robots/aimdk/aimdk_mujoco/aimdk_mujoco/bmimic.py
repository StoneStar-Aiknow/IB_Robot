"""X2's 151-value BMIMIC contract, not the 154-value Unitree anchor contract.

Observation semantics follow AimDK extra/x2_rl_deploy. Model metadata supplies
joint ordering, gains, default pose and action scale; weights remain local-only.
"""

import numpy as np


class BMimic:
    def __init__(self, session, joint_names, start, end, frame_count=None):
        inputs = {item.name: (item.shape, item.type) for item in session.get_inputs()}
        outputs = {item.name: (item.shape, item.type) for item in session.get_outputs()}
        if inputs != {"obs": ([1, 151], "tensor(float)"), "time_step": ([1, 1], "tensor(float)")}:
            raise ValueError("BMIMIC requires float obs [1,151] and time_step [1,1]")
        for name in ("actions", "joint_pos", "joint_vel"):
            if outputs.get(name) != ([1, 29], "tensor(float)"):
                raise ValueError(f"BMIMIC requires float {name} [1,29]")
        metadata = session.get_modelmeta().custom_metadata_map
        if (
            metadata.get("observation_names") != ("command,projected_gravity,base_ang_vel,joint_pos,joint_vel,actions")
            or metadata.get("command_names") != "motion"
        ):
            raise ValueError("Unsupported BMIMIC observation semantics")
        names = metadata.get("joint_names", "").split(",")
        expected = {name for name in joint_names if not name.startswith("head_")}
        if len(names) != 29 or len(set(names)) != 29 or set(names) != expected:
            raise ValueError("BMIMIC must declare exactly the 29 X2 non-head joints")
        self.indices = np.array([joint_names.index(name) for name in names])

        def vector(key, positive=False):
            values = np.array([float(value) for value in metadata.get(key, "").split(",")], dtype=np.float32)
            if values.shape != (29,) or not np.isfinite(values).all() or (positive and np.any(values <= 0)):
                raise ValueError(f"Invalid BMIMIC metadata: {key}")
            return values

        self.default = vector("default_joint_pos")
        self.kp = vector("joint_stiffness", positive=True)
        self.kd = vector("joint_damping", positive=True)
        self.scale = vector("action_scale", positive=True)
        declared = metadata.get("time_step_total")
        if declared is None and frame_count is None:
            raise ValueError("BMIMIC requires time_step_total metadata or a verified frame_count in config")
        total = int(declared) if declared is not None else frame_count
        if frame_count is not None and frame_count != total:
            raise ValueError("BMIMIC frame_count disagrees with time_step_total metadata")
        if not 0 <= start <= end < total:
            raise ValueError("BMIMIC phase must satisfy 0 <= start <= end < time_step_total")
        self.frame = start
        self.end = end
        self.finished = False
        self.session = session
        self.last_action = np.zeros(29, dtype=np.float32)

    def target(self, positions, velocities, gravity, gyro):
        if self.finished:
            raise RuntimeError("BMIMIC motion finished; restart explicitly to replay")
        feeds = {
            "obs": np.zeros((1, 151), dtype=np.float32),
            "time_step": np.array([[self.frame]], dtype=np.float32),
        }
        # Reference tables are independent of obs. Fetch this frame before the
        # actor call, avoiding the SDK example's one-frame reference cache lag.
        references = self.session.run(["joint_pos", "joint_vel"], feeds)
        if len(references) != 2 or any(value.shape != (1, 29) or not np.isfinite(value).all() for value in references):
            raise ValueError("Invalid BMIMIC reference output")
        observation = np.concatenate(
            (
                references[0][0],
                references[1][0],
                gravity,
                gyro * 0.25,
                positions[self.indices] - self.default,
                velocities[self.indices] * 0.05,
                self.last_action,
            )
        )
        if observation.shape != (151,) or not np.isfinite(observation).all():
            raise ValueError("Invalid BMIMIC observation")
        feeds["obs"] = np.clip(observation, -100, 100).astype(np.float32).reshape(1, 151)
        action = self.session.run(["actions"], feeds)[0]
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise ValueError("Invalid BMIMIC action output")
        self.last_action[:] = np.clip(action[0], -100, 100)
        target = self.default + self.scale * self.last_action
        self.finished = self.frame == self.end
        if not self.finished:
            self.frame += 1
        return target
