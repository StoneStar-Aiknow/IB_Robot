"""Optional physics for the vendor mock: the official X2 MuJoCo model, pelvis welded.

Off by default: the mock is then the follower the contract tests are written
against (a command is reached at once). With a model path, the arm joints are
driven by a joint-space PD loop at 1 kHz, as torque motors with the URDF's rated
effort, and joint state is read back from the simulation: commands take time to
be reached and a pose held against gravity sags by the error the gains leave, as
the platform's does (0.05-0.19 rad on the right shoulder pitch, measured
2026-10-22). It is not the platform's controller and has no balance: the
pelvis is fixed, so it says nothing about whether the robot would stay upright.

The model is the vendor's own, MulanPSL-2.0 licensed, and is not vendored: the
``aimdk_fetch_x2_mjcf`` tool fetches the commit ``aimdk_description`` pins.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

# Joint-space gains per joint family (Nm/rad, Nm s/rad) and the rated torque clamp
# from the official URDF (<limit effort>).
GAINS = {"shoulder": (60.0, 3.0), "elbow": (40.0, 2.0), "wrist": (10.0, 0.4)}
EFFORT_NM = {"shoulder_pitch": 36.0, "shoulder_roll": 36.0, "shoulder_yaw": 24.0, "elbow": 24.0,
             "wrist_yaw": 24.0, "wrist_pitch": 4.8, "wrist_roll": 4.8}  # fmt: skip
STEP_S = 0.001
CACHE = Path.home() / ".cache" / "ibrobot" / "agibot_x2_urdf"
UPSTREAM = "https://github.com/AgibotTech/agibot_x2_urdf.git"
# The commit aimdk_description/manifest.yaml was synced from (a test holds them equal).
PINNED_COMMIT = "575cc6b988f976c23550e0db85aa1e5475d3652d"


class ArmPhysics:
    """A background simulation of the X2 arms tracking position targets."""

    def __init__(self, scene_path: str, arm_joints: list[str]) -> None:
        import mujoco  # noqa: PLC0415 - only when physics is requested

        self._mj = mujoco
        spec = mujoco.MjSpec.from_file(str(scene_path))
        for joint in list(spec.joints):
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                spec.delete(joint)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._arm = list(arm_joints)
        self._qadr = {n: int(self.model.joint(n).qposadr[0]) for n in self._arm}
        self._dadr = {n: int(self.model.joint(n).dofadr[0]) for n in self._arm}
        self._act = {}
        for a in range(self.model.nu):
            name = self.model.joint(self.model.actuator_trnid[a][0]).name
            self._act[name] = a
        self._others = [
            self.model.joint(j).name for j in range(self.model.njnt) if self.model.joint(j).name not in self._qadr
        ]
        self._lock = threading.Lock()
        self._targets: dict[str, float] = dict.fromkeys(self._arm, 0.0)
        self._fixed: dict[str, float] = dict.fromkeys(self._others, 0.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="x2_mock_physics", daemon=True)
        self._thread.start()

    @staticmethod
    def _family(name: str) -> str:
        return next(f for f in ("shoulder", "elbow", "wrist") if f in name)

    def set_targets(self, targets: dict[str, float]) -> None:
        with self._lock:
            for name, value in targets.items():
                if name in self._targets:
                    self._targets[name] = float(value)
                elif name in self._fixed:
                    self._fixed[name] = float(value)

    def place(self, positions: dict[str, float]) -> None:
        """Teleport joints (and make them the targets), as the follower's set_positions does."""
        with self._lock:
            for name, value in positions.items():
                if name in self._qadr:
                    self.data.qpos[self._qadr[name]] = float(value)
                    self.data.qvel[self._dadr[name]] = 0.0
                    self._targets[name] = float(value)
                elif name in self._fixed:
                    self._fixed[name] = float(value)

    def positions(self) -> dict[str, float]:
        with self._lock:
            out = {n: float(self.data.qpos[a]) for n, a in self._qadr.items()}
            out.update(self._fixed)
        return out

    def _loop(self) -> None:
        mj, m, d = self._mj, self.model, self.data
        next_tick = time.monotonic()
        while self._running:
            with self._lock:
                for _ in range(10):
                    for name, target in self._targets.items():
                        kp, kd = GAINS[self._family(name)]
                        effort = EFFORT_NM[name.split("_", 1)[1].removesuffix("_joint")]
                        tau = kp * (target - d.qpos[self._qadr[name]]) - kd * d.qvel[self._dadr[name]]
                        d.ctrl[self._act[name]] = max(-effort, min(effort, tau))
                    for name, value in self._fixed.items():
                        j = m.joint(name)
                        d.qpos[j.qposadr[0]] = value
                        d.qvel[j.dofadr[0]] = 0.0
                    mj.mj_step(m, d)
            next_tick += 10 * STEP_S
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)


def fetch_main(argv=None) -> int:
    """Fetch the pinned official model (MJCF + meshes) into the cache; print the scene path."""
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=fetch_main.__doc__)
    parser.add_argument("--commit", default=PINNED_COMMIT, help="default: the commit aimdk_description pins")
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args(argv)
    commit = args.commit
    root = Path(args.cache) / commit
    scene = root / "X2_URDF-v1.3.0" / "scene.xml"
    if not scene.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM, str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", "X2_URDF-v1.3.0", "LICENSE"], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", commit], check=True)
    print(scene)
    return 0
