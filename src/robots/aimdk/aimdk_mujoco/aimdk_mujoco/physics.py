"""X2 contact dynamics and ONNX locomotion, without ROS or the vendor runtime.

Policy contract and gains follow AimDK mc-rl x2_config (Apache-2.0).
The supplied weights are a separate, local-only artifact; see README.md.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument, RuntimeException

from aimdk_mujoco.assets import POLICY_SHA256, digest
from aimdk_mujoco.bmimic import BMimic

# Isaac Lab's interleaved observation order, not the MJCF declaration order.
JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
)
GAINS = {
    "hip_pitch": (120, 5),
    "hip_roll": (100, 4),
    "hip_yaw": (100, 4),
    "knee": (150, 5),
    "ankle_pitch": (40, 2),
    "ankle_roll": (40, 2),
    "waist_yaw": (40.18, 2.56),
    "waist_pitch": (200, 2),
    "waist_roll": (200, 2),
    "head_yaw": (0, 2),
    "head_pitch": (0, 2),
    "shoulder_pitch": (50, 3),
    "shoulder_roll": (50, 3),
    "shoulder_yaw": (50, 3),
    "elbow": (50, 3),
    "wrist_yaw": (20, 2),
    "wrist_pitch": (20, 2),
    "wrist_roll": (20, 2),
}
DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "left_elbow_joint": -0.3,
    "right_elbow_joint": -0.3,
}
TERM_DIMS = (3, 3, 3, 31, 31, 29)


class ObservationHistory:
    """Per-term, oldest-first history, as in training (not per-frame stacking)."""

    def __init__(self):
        self.terms = [np.zeros((5, dim), dtype=np.float32) for dim in TERM_DIMS]

    def append(self, terms):
        for history, term in zip(self.terms, terms, strict=True):
            history[:-1] = history[1:]
            history[-1] = term
        return np.clip(np.concatenate([h.ravel() for h in self.terms]), -100, 100).reshape(1, 500)


class X2Simulation:
    def __init__(self, asset_dir: Path, controller_config: Path | None = None, motion: str | None = None):
        asset_dir = Path(asset_dir).resolve(strict=True)
        self.policy_path = None
        self.asset_dir = asset_dir
        self.controller = None
        self.motion = None
        self.motion_running = False
        self._configure(controller_config, motion)
        manifest = json.loads((asset_dir / "assets.json").read_text())
        for name, expected in manifest["files"].items():
            if name == "policy.onnx":
                continue
            path = (asset_dir / name).resolve(strict=True)
            if not path.is_relative_to(asset_dir) or digest(path) != expected:
                raise ValueError(f"Asset is outside cache or has changed: {name}")
        self.model = mujoco.MjModel.from_xml_path(str(asset_dir / "scene.xml"))
        self.model.opt.timestep = 0.001
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
        self.data = mujoco.MjData(self.model)
        self.joint_names = JOINT_NAMES

        def object_id(kind, name):
            result = mujoco.mj_name2id(self.model, kind, name)
            if result < 0:
                raise ValueError(f"Missing model object: {name}")
            return result

        joints = [object_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names]
        self.qpos_indices = self.model.jnt_qposadr[joints]
        self.qvel_indices = self.model.jnt_dofadr[joints]
        self.actuators = np.array(
            [object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}") for name in self.joint_names]
        )
        self.pelvis_id = object_id(mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        gyro = object_id(mujoco.mjtObj.mjOBJ_SENSOR, "body-angular-velocity")
        self.gyro_address = self.model.sensor_adr[gyro]
        gain_names = [name.removeprefix("left_").removeprefix("right_").removesuffix("_joint") for name in JOINT_NAMES]
        self.kp, self.kd = np.array([GAINS[name] for name in gain_names]).T
        self.default = np.array([DEFAULT_POSE.get(name, 0.0) for name in JOINT_NAMES])
        self.action_indices = np.array([i for i, name in enumerate(JOINT_NAMES) if not name.startswith("head_")])
        self.session = None
        self.history = ObservationHistory()
        self.last_action = np.zeros(29, dtype=np.float32)
        self.policy_steps = 0
        self.fault = None
        base_joint = object_id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        base_qpos = self.model.jnt_qposadr[base_joint]
        self.data.qpos[base_qpos + 2] = 0.57
        self.data.qpos[self.qpos_indices] = self.default
        mujoco.mj_forward(self.model, self.data)
        self._integrate(self.default, 500)
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        self._initial_gains = (self.kp.copy(), self.kd.copy(), self.default.copy())
        self.execution_running = True
        self.selection = None
        self.selection_finished = False

    def _configure(self, controller_config, motion):
        asset_dir = self.asset_dir
        if motion is not None and controller_config is None:
            raise ValueError("Motion selection requires controller_config")
        if controller_config is not None:
            config = yaml.safe_load(Path(controller_config).read_text())
            if not isinstance(config, dict):
                raise ValueError("Controller config must be a mapping")
            self.controller = config.get("controller")
            fields = {"controller", "policy"}
            if self.controller == "bmimic":
                if set(config) != {"controller", "motions"} or not isinstance(config["motions"], dict):
                    raise ValueError("BMIMIC config must contain exactly controller and a motions mapping")
                motions = config["motions"]
                if not motions or any(not isinstance(name, str) or not name.strip() for name in motions):
                    raise ValueError("BMIMIC motions must have nonempty string names")
                if not motion or motion not in motions:
                    raise ValueError(f"Select a BMIMIC motion explicitly; available: {', '.join(motions)}")
                config = motions[motion]
                if not isinstance(config, dict):
                    raise ValueError("BMIMIC motion must be a mapping")
                fields = {"policy", "start_count", "end_count"}
                if "frame_count" in config:
                    fields.add("frame_count")
                    if type(config["frame_count"]) is not int or config["frame_count"] <= 0:
                        raise ValueError("BMIMIC frame_count must be a positive integer")
                if any(type(config.get(key)) is not int for key in ("start_count", "end_count")) or not (
                    0 <= config["start_count"] <= config["end_count"]
                ):
                    raise ValueError("BMIMIC phase requires integer 0 <= start_count <= end_count")
            elif self.controller != "mc_rl_velocity":
                raise ValueError(f"Unsupported controller: {self.controller!r}; supported: mc_rl_velocity, bmimic")
            elif motion is not None:
                raise ValueError("Motion selection requires a BMIMIC controller")
            if set(config) != fields:
                raise ValueError(f"Controller config must contain exactly {sorted(fields)}")
            self.controller_options = config
            policy = config["policy"]
            if not isinstance(policy, str) or not policy.strip():
                raise ValueError("Controller policy must be a nonempty asset-relative path")
            relative = Path(policy)
            if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
                raise ValueError("Controller policy must be a file path within assets")
            self.policy_path = asset_dir / relative
            if not self.policy_path.resolve().is_relative_to(asset_dir):
                raise ValueError("Controller policy must stay within assets")

    def select_policy(self, config, motion, reset):
        if self.fault:
            raise RuntimeError(self.fault)
        selection = (str(Path(config).resolve()), motion)
        if not reset:
            if (
                self.selection != selection
                or self.session is None
                or self.selection_finished
                or (self.motion and self.motion.finished)
            ):
                raise ValueError(
                    "Resume requires the same unfinished policy; select reset_before_start to switch/replay"
                )
            self.execution_running = True
            self.motion_running = self.controller == "bmimic"
            return
        self.activate_policy(self.prepare_policy(config, motion))

    def prepare_policy(self, config, motion):
        # Only reads immutable configuration: safe to load on a worker while paused.
        candidate = copy.copy(self)
        candidate.kp, candidate.kd, candidate.default = (value.copy() for value in self._initial_gains)
        candidate.session = None
        candidate.motion = None
        candidate._configure(config, motion)
        candidate._load_policy()
        candidate.selection = (str(Path(config).resolve()), motion)
        return candidate

    def activate_policy(self, candidate, *, transition=False):
        if self.fault:
            raise RuntimeError(self.fault)
        if transition:
            truth = self.ground_truth()
            if not all(np.isfinite(self.data.qpos)) or not all(np.isfinite(self.data.qvel)):
                raise ValueError("Nonfinite state cannot enter a policy handoff")
            if candidate.controller == "bmimic":
                if not self.is_standing():
                    raise ValueError("Dance handoff requires standing recovery first")
                positions = self.data.qpos[self.qpos_indices]
                velocities = self.data.qvel[self.qvel_indices]
                if np.max(np.abs(positions - candidate.default)) > 0.5:
                    raise ValueError("Dance handoff joint pose exceeds 0.5 rad from candidate default")
                # Probe a private motion cursor: validation must not consume frame zero.
                motion = copy.copy(candidate.motion)
                motion.last_action = candidate.motion.last_action.copy()
                rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
                target = candidate.default.copy()
                target[motion.indices] = motion.target(
                    positions,
                    velocities,
                    rotation.T @ np.array([0.0, 0.0, -1.0]),
                    self.data.sensordata[self.gyro_address : self.gyro_address + 3],
                )
                torque = candidate.kp * (target - positions) - candidate.kd * velocities
                if np.max(np.abs(target - positions)) > 0.5 or np.max(np.abs(torque)) > 80:
                    raise ValueError("Dance handoff initial target exceeds 0.5 rad or 80 Nm guard")
            elif (
                truth["height"] < 0.5
                or truth["tilt_rad"] > math.radians(25)
                or np.linalg.norm(truth["linear_velocity_world"]) > 0.6
                or np.linalg.norm(truth["angular_velocity_body"]) > 2.0
            ):
                raise ValueError("State outside experimental balance handoff limits")
        else:
            now = self.data.time
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = self._initial_qpos
            self.data.qvel[:] = self._initial_qvel
            self.data.time = now
            mujoco.mj_forward(self.model, self.data)
        for key in ("controller", "controller_options", "policy_path", "session", "motion", "kp", "kd", "default"):
            setattr(self, key, getattr(candidate, key))
        if self.controller == "mc_rl_velocity":
            self.input, self.output = candidate.input, candidate.output
        self.history = ObservationHistory()
        self.last_action[:] = 0
        self.policy_steps = 0
        self.selection = candidate.selection
        self.selection_finished = False
        self.execution_running = True
        self.motion_running = self.controller == "bmimic"

    def is_standing(self):
        truth = self.ground_truth()
        return (
            np.isfinite(truth["height"])
            and np.isfinite(truth["tilt_rad"])
            and truth["height"] >= 0.55
            and truth["tilt_rad"] >= 0
            and truth["tilt_rad"] <= math.radians(8)
            and np.linalg.norm(truth["linear_velocity_world"]) <= 0.05
            and np.linalg.norm(truth["angular_velocity_body"]) <= 0.15
            and np.max(np.abs(self.data.qvel[self.qvel_indices])) <= 0.5
        )

    def pause_execution(self):
        self.execution_running = False
        self.motion_running = False

    def _load_policy(self):
        if not self.policy_path.resolve(strict=True).is_relative_to(self.asset_dir):
            raise ValueError("Controller policy must stay within assets")
        if self.controller == "mc_rl_velocity" and digest(self.policy_path) != POLICY_SHA256:
            raise ValueError("Unsupported walking policy")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(self.policy_path), options, providers=["CPUExecutionProvider"])
        if self.controller == "bmimic":
            self.motion = BMimic(
                session,
                self.joint_names,
                self.controller_options["start_count"],
                self.controller_options["end_count"],
                self.controller_options.get("frame_count"),
            )
            self.default[self.motion.indices] = self.motion.default
            self.kp[self.motion.indices] = self.motion.kp
            self.kd[self.motion.indices] = self.motion.kd
        else:
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != [1, 500] or outputs[0].shape != [1, 29]:
                raise ValueError("Policy must have input [1,500] and output [1,29]")
            self.input, self.output = inputs[0], outputs[0]
        self.session = session

    def _integrate(self, target, steps):
        for _ in range(steps):
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.fault = "Nonfinite physics state"
                raise RuntimeError(self.fault)
            torque = (
                self.kp * (target - self.data.qpos[self.qpos_indices]) - self.kd * self.data.qvel[self.qvel_indices]
            )
            self.data.ctrl[self.actuators] = torque
            mujoco.mj_step(self.model, self.data)
            for warning in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
                mujoco.mjtWarning.mjWARN_BADCTRL,
            ):
                if self.data.warning[warning].number:
                    self.fault = f"MuJoCo numerical instability: {warning.name}"
                    raise RuntimeError(self.fault)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.fault = "Nonfinite physics state"
                raise RuntimeError(self.fault)
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        if self.data.xpos[self.pelvis_id, 2] < 0.3 or rotation[2, 2] < math.cos(math.radians(46)):
            self.fault = "Robot fell or exceeded 46 degree tilt; restart explicitly"
            raise RuntimeError(self.fault)

    def step(self, command=(0.0, 0.0, 0.0)):
        if self.fault:
            raise RuntimeError(self.fault)
        if not self.execution_running:
            return
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Velocity command must contain three finite numbers")
        if np.any(command < [-0.3, -0.3, -0.2]) or np.any(command > [0.5, 0.3, 0.2]):
            raise ValueError("Velocity command exceeds supported policy limits")
        if self.controller == "bmimic":
            if np.any(command):
                raise ValueError("BMIMIC does not accept velocity commands")
            if not self.motion_running:
                return
        if self.session is None:
            if self.policy_path is None or not self.policy_path.is_file():
                if self.controller == "bmimic":
                    self.motion_running = False
                    raise RuntimeError(f"Place authorized weights at {self.policy_path}")
                if not np.any(command):
                    return  # Paused model preview, not active balance without a controller.
                raise RuntimeError(self.preview_message)
            try:
                self._load_policy()
            except (ValueError, OSError, Fail, InvalidArgument, RuntimeException) as exc:
                self.fault = f"Policy loading failed: {exc}"
                raise RuntimeError(self.fault) from exc
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        if self.data.xpos[self.pelvis_id, 2] < 0.3 or -gravity[2] < math.cos(math.radians(46)):
            self.fault = "Robot fell or exceeded 46 degree tilt; restart explicitly"
            raise RuntimeError(self.fault)
        gyro = self.data.sensordata[self.gyro_address : self.gyro_address + 3]
        if self.controller == "bmimic":
            try:
                target = self.default.copy()
                target[self.motion.indices] = self.motion.target(
                    self.data.qpos[self.qpos_indices],
                    self.data.qvel[self.qvel_indices],
                    gravity,
                    gyro,
                )
                self._integrate(target, 20)
            except (ValueError, Fail, InvalidArgument, RuntimeException) as exc:
                self.fault = f"BMIMIC inference failed: {exc}"
                raise RuntimeError(self.fault) from exc
            self.policy_steps += 1
            if self.motion.finished:
                self.motion_running = False
            return
        obs = self.history.append(
            (
                gyro * 0.25,
                gravity,
                command,
                self.data.qpos[self.qpos_indices] - self.default,
                self.data.qvel[self.qvel_indices] * 0.05,
                self.last_action,
            )
        )
        try:
            action = self.session.run([self.output.name], {self.input.name: obs})[0][0]
        except (Fail, InvalidArgument, RuntimeException) as exc:
            self.fault = f"Policy inference failed: {exc}"
            raise RuntimeError(self.fault) from exc
        if action.shape != (29,) or not np.isfinite(action).all():
            self.fault = "Invalid policy output"
            raise RuntimeError(self.fault)
        action = np.clip(action, -5.0, 5.0)
        if self.policy_steps < 5:
            action[:] = 0.0
        self.last_action[:] = action
        target = self.default.copy()
        target[self.action_indices] += 0.25 * action
        self._integrate(target, 20)
        self.policy_steps += 1

    @property
    def preview_message(self):
        if self.controller == "bmimic":
            if self.motion is not None and self.motion.finished:
                return "BMIMIC finished; physics paused, not active balance; restart to replay"
            return (
                "BMIMIC physics paused, not active balance; /simulation/start resumes motion; "
                f"authorized weights required at {self.policy_path}"
            )
        if self.policy_path is None:
            return "Model preview paused; select controller_config for active balance and motion"
        return f"Model preview paused; place authorized weights at {self.policy_path} for active balance and motion"

    def start_motion(self):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.controller != "bmimic":
            raise ValueError("/simulation/start requires a BMIMIC controller")
        if self.motion is not None and self.motion.finished:
            raise ValueError("BMIMIC finished; restart explicitly to replay")
        if not self.policy_path.is_file():
            raise ValueError(f"Place authorized weights at {self.policy_path}")
        self.motion_running = True

    def pause_motion(self):
        self.motion_running = False

    def ground_truth(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.pelvis_id, velocity, 0)
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        return {
            "time": self.data.time,
            "position": self.data.xpos[self.pelvis_id].tolist(),
            "quaternion_wxyz": self.data.xquat[self.pelvis_id].tolist(),
            "linear_velocity_world": velocity[3:].tolist(),
            "angular_velocity_body": (rotation.T @ velocity[:3]).tolist(),
            "height": float(self.data.xpos[self.pelvis_id, 2]),
            "tilt_rad": math.acos(float(np.clip(rotation[2, 2], -1, 1))),
            "contacts": self.data.ncon,
        }


def verify(sim):
    from aimdk_mujoco.gait import GaitMonitor

    report = {"passed": False, "phases": []}
    if sim.controller == "bmimic":
        report = {"passed": False, "controller": "bmimic", "completed_frames": 0}
        try:
            sim.start_motion()
            samples = []
            while sim.motion_running:
                sim.step()
                samples.append(sim.ground_truth())
                report["completed_frames"] = sim.policy_steps
            report.update(
                passed=True,
                min_height_m=min(sample["height"] for sample in samples),
                max_tilt_deg=math.degrees(max(sample["tilt_rad"] for sample in samples)),
                final_state=samples[-1],
                completion="Physics paused at end; no active balance or motion-quality claim",
            )
        except (ValueError, RuntimeError) as exc:
            report["error"] = str(exc)
        return report
    try:
        for name, duration, command in (
            ("stand", 5, (0.0, 0.0, 0.0)),
            ("forward", 10, (0.2, 0.0, 0.0)),
            ("stop", 5, (0.0, 0.0, 0.0)),
        ):
            start = sim.ground_truth()
            gait = GaitMonitor(sim) if name == "forward" else None
            samples = []
            for _ in range(int(duration / 0.02)):
                sim.step(command)
                samples.append(sim.ground_truth())
                if gait is not None:
                    gait.sample()
            end = samples[-1]
            displacement = np.subtract(end["position"], start["position"])
            phase = {
                "name": name,
                "duration_s": duration,
                "command": command,
                "displacement_m": displacement.tolist(),
                "min_height_m": min(s["height"] for s in samples),
                "max_tilt_deg": math.degrees(max(s["tilt_rad"] for s in samples)),
                "contact_fraction": sum(s["contacts"] > 0 for s in samples) / len(samples),
                "final_speed_m_s": float(np.linalg.norm(end["linear_velocity_world"][:2])),
                "last_second_speed_m_s": float(
                    np.mean([np.linalg.norm(s["linear_velocity_world"][:2]) for s in samples[-50:]])
                ),
            }
            report["phases"].append(phase)
            if gait is not None:
                phase["gait"] = gait.summary()
            if phase["min_height_m"] < 0.5 or phase["max_tilt_deg"] > 20:
                raise RuntimeError(f"{name}: unstable posture")
            if phase["contact_fraction"] < 0.8:
                raise RuntimeError(f"{name}: insufficient ground contact")
            if name == "forward" and (displacement[0] < 0.8 or abs(displacement[1]) > 0.5):
                raise RuntimeError("Insufficient forward displacement or excessive sideways drift")
            if name in ("stand", "stop") and phase["last_second_speed_m_s"] > 0.1:
                raise RuntimeError(f"{name}: failed to settle to zero velocity")
        if not report["phases"][1]["gait"]["passed"]:
            raise RuntimeError("Root translation passed, but foot gait failed: shallow shuffle or sliding")
        report["passed"] = True
    except RuntimeError as exc:
        report["error"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Directory containing prepared assets")
    parser.add_argument(
        "--controller-config", type=Path, help="Optional controller YAML; omitted: paused model preview"
    )
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--motion", help="Named motion in the BMIMIC controller YAML")
    parser.add_argument("--report", type=Path)
    options = parser.parse_args()
    report = verify(X2Simulation(options.assets, options.controller_config, options.motion))
    text = json.dumps(report, indent=2)
    print(text)
    if options.report:
        options.report.write_text(text + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
