"""RoboDojo platform adapter: the only place that knows RoboDojo's protocol and observation keys.

RoboDojo owns the main loop (platform pull). Its eval client (Isaac process, Python 3.11,
``robodojo_client/deploy.py``) calls the policy server hosted here once per tick:

    reset -> begin_episode -> (update_obs, get_action)* -> episode_end

and this adapter turns each call into the port of ``robot_runtime_sim.direct_control``: one
RoboDojo episode is one session of the engine, native observation keys become an
``ObservationFrame``, the engine's per-group targets become RoboDojo's joint action dict, IK
requests ride on ``get_action`` replies (RoboDojo's cuRobo is the only IK solver), and
RoboDojo's verdict is handed to the engine as the session's platform result, never into the
public session event. The wire protocol keeps RoboDojo's own word, ``episode_id``.

The adapter is named by this platform's embodiment (``robodojo_robot.embodiment``), not by
an adapter registry; the platform itself is registered in ``ibrobot.sim_platforms``.

Native observation keys (RoboDojo ``arx_x5``): ``state.<side>_arm_joint_state`` (6),
``state.<side>_ee_joint_state`` (gripper opening, 1 = open), ``state.<side>_ee_pose``
(link6 ``[x, y, z, qw, qx, qy, qz]`` in world), ``vision.<camera>.color``. ``<side>`` is the
``side`` of each motion group in the runtime profile.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

from robodojo_robot.overlay import CameraModel, draw_overlay
from robodojo_robot.policy_server import PolicyServerThread
from robot_runtime.path_subst import resolve_path
from robot_runtime_sim.direct_control.engine import SessionError, SimulationCore
from robot_runtime_sim.direct_control.ports import (
    CLOCK_PLATFORM_PULL,
    AdapterDescriptor,
    DescriptorError,
    IkReply,
    ObservationFrame,
    PlatformResult,
    descriptor_from_profile,
    platform_settings,
)

PLATFORM = "robodojo"  # the ibrobot.sim_platforms name and the profile's simulation.<platform> section
BODY_KEY = "_body"
WRITER = "robodojo_ws"
PROTOCOL_VERSION = "robodojo-ws-v2"
# Episode description fields the public session event may carry. Seeds, layout
# identity and camera calibration stay with the adapter (evaluation record / overlay only).
PUBLIC_EPISODE_FIELDS = ("run_id", "task", "instruction", "step_lim")


def _floats(values) -> list[float]:
    return [float(v) for v in (values.tolist() if hasattr(values, "tolist") else list(values))]


class RoboDojoCodec:
    """Native RoboDojo observation / action <-> the normalized port."""

    def __init__(self, descriptor: AdapterDescriptor, sides: Mapping[str, str], gripper_default: float = 1.0):
        self.descriptor = descriptor
        self.sides = dict(sides)  # group -> RoboDojo side prefix
        self.gripper_default = gripper_default

    def decode_observation(self, native: Mapping[str, Any], *, session_id: str, generation: int) -> ObservationFrame:
        state = native.get("state") or {}
        joints, grippers, ee = {}, {}, {}
        for group in self.descriptor.groups:
            side = self.sides[group.name]
            q = state.get(f"{side}_arm_joint_state")
            if q is None:
                raise SessionError(f"observation has no state.{side}_arm_joint_state")
            joints[group.name] = _floats(q)[: group.dof]
            if group.gripper is not None:
                g = state.get(f"{side}_ee_joint_state")
                grippers[group.name] = _floats(g)[0] if g is not None else self.gripper_default
            pose = state.get(f"{side}_ee_pose")
            if pose is not None:
                ee[group.name] = _floats(pose)[:7]
        vision = native.get("vision") or {}
        images = {
            cam: (vision.get(cam) or {}).get("color") for cam in self.descriptor.cameras if (vision.get(cam) or {})
        }
        meta = native.get(BODY_KEY) or {}
        return ObservationFrame(
            session_id=session_id,
            generation=generation,
            step_id=int(meta.get("tick", -1)),
            joints=joints,
            grippers=grippers,
            ee_poses=ee,
            images={k: v for k, v in images.items() if v is not None},
            wall_time=time.monotonic(),
        )

    def encode_action(self, targets: Mapping[str, Mapping[str, Any]]) -> dict[str, list[float]]:
        action: dict[str, list[float]] = {}
        for group, entry in targets.items():
            side = self.sides[group]
            action[f"{side}_arm_joint_state"] = [float(v) for v in entry["joints"]]
            if "gripper" in entry:
                action[f"{side}_ee_joint_state"] = [float(entry["gripper"])]
        return action


class RoboDojoPolicyModel:
    """The object RoboDojo's ``PolicyServer`` dispatches calls to (every public method is callable)."""

    def __init__(self, adapter: RoboDojoAdapter):
        self._adapter = adapter

    def reset(self) -> dict[str, Any]:
        return self._adapter.on_reset()

    def begin_episode(self, info: Mapping[str, Any]) -> dict[str, Any]:
        return self._adapter.on_begin_episode(info)

    def update_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        return self._adapter.on_update_obs(obs)

    def get_action(self, request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return self._adapter.on_get_action(request or {})

    def episode_end(self, info: Mapping[str, Any]) -> dict[str, Any]:
        return self._adapter.on_episode_end(info)

    def episode_interrupted(self, info: Mapping[str, Any]) -> dict[str, Any]:
        return self._adapter.on_episode_interrupted(info)


class RoboDojoAdapter:
    """Platform-pull adapter of the ``robodojo`` platform (``create_adapter`` is its factory)."""

    def __init__(self, profile: Mapping[str, Any], overrides: Mapping[str, Any] | None = None):
        sim = platform_settings(profile, PLATFORM)
        cfg = {**dict(sim.get("adapter_config") or {}), **dict(overrides or {})}
        self.descriptor = descriptor_from_profile(profile, platform=PLATFORM, clock_mode=CLOCK_PLATFORM_PULL)
        if self.descriptor.clock_mode != CLOCK_PLATFORM_PULL:
            # RoboDojo owns its main loop; this adapter has no host-step driver.
            raise DescriptorError(f"the robodojo adapter only drives platform_pull, not {self.descriptor.clock_mode!r}")
        groups = dict(profile.get("motion_groups") or {})
        self._sides = {name: str(spec.get("side", name)) for name, spec in groups.items()}
        self.codec = RoboDojoCodec(self.descriptor, self._sides, float(cfg.get("gripper_default", 1.0)))
        self._cfg = cfg
        self.overlay_cameras = tuple(str(c) for c in cfg.get("overlay_cameras", ()))
        self._overlay_axes = str(cfg.get("overlay_axes", "usd"))
        self._table_z = float(cfg.get("table_top_z", 0.0))
        self._silence_s = float(cfg.get("platform_silence_s", 120.0))
        self._core: SimulationCore | None = None
        self._server: PolicyServerThread | None = None
        self._private: dict[str, Any] = {}
        self._overlay_models: dict[str, CameraModel] = {}
        self._last_call = time.monotonic()
        self._watchdog_stop = threading.Event()
        self._logger = None

    # ------------------------------------------------------------------ lifecycle (runtime node)

    def bind(self, core: SimulationCore, logger=None) -> RoboDojoPolicyModel:
        """Attach the core as the only writer; returns the model the policy server dispatches to."""
        self._core = core
        self._logger = logger
        core.claim_writer(WRITER)
        return RoboDojoPolicyModel(self)

    def start(self, core: SimulationCore, logger) -> None:
        model = self.bind(core, logger)
        root = resolve_path(str(self._cfg.get("robodojo_root", "")))
        if not root:
            raise RuntimeError("robodojo adapter needs robodojo_root (set ROBODOJO_ROOT or adapter_overrides_json)")
        self._server = PolicyServerThread(
            model,
            robodojo_root=root,
            host=str(self._cfg.get("host", "127.0.0.1")),
            port=int(self._cfg.get("port", 34950)),
            logger=logger,
        )
        self._server.start()
        threading.Thread(target=self._watchdog, name="robodojo_watchdog", daemon=True).start()
        logger.info(f"RoboDojo policy server listening on {self._server.url}; waiting for the eval client")

    def stop(self) -> None:
        self._watchdog_stop.set()
        if self._server is not None:
            self._server.stop()

    def _watchdog(self) -> None:
        # The eval client polls get_action continuously, idle or not; a long silence while an
        # episode runs means the platform is gone and in-flight commands have unknown outcomes.
        while not self._watchdog_stop.wait(1.0):
            core = self._core
            if core is None or core.state not in ("READY", "RUNNING", "STOPPING"):
                continue
            silent = time.monotonic() - self._last_call
            if silent > self._silence_s:
                n = core.platform_lost(f"no call from the RoboDojo client for {silent:.0f} s")
                if self._logger is not None:
                    self._logger.error(f"RoboDojo client silent for {silent:.0f} s; {n} command(s) unknown")

    # ------------------------------------------------------------------ RoboDojo calls

    def _touch(self) -> SimulationCore:
        self._last_call = time.monotonic()
        assert self._core is not None
        return self._core

    def on_reset(self) -> dict[str, Any]:
        core = self._touch()
        ident = core.reset()
        self._private = {}
        self._overlay_models = {}
        # RoboDojo's word is episode_id; session_id is also kept for the first protocol's clients.
        return {**ident, "episode_id": ident["session_id"], "protocol_version": PROTOCOL_VERSION}

    def on_begin_episode(self, info: Mapping[str, Any]) -> dict[str, Any]:
        core = self._touch()
        info = dict(info)
        self._private = {k: v for k, v in info.items() if k not in PUBLIC_EPISODE_FIELDS}
        public = {k: info[k] for k in PUBLIC_EPISODE_FIELDS if k in info}
        ident = core.begin_session(public)
        return {**ident, "episode_id": ident["session_id"]}

    def _ident(self, meta: Mapping[str, Any]) -> tuple[Any, Any]:
        return meta.get("episode_id", meta.get("session_id")), meta.get("generation")

    def on_update_obs(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        core = self._touch()
        episode_id, generation = self._ident(obs.get(BODY_KEY) or {})
        core.check_session(episode_id, generation)
        frame = self.codec.decode_observation(obs, session_id=episode_id, generation=core.generation)
        core.on_observation(frame)
        return {"accepted": True, "tick": frame.step_id}

    def on_get_action(self, request: Mapping[str, Any]) -> dict[str, Any]:
        core = self._touch()
        episode_id, generation = self._ident(request)
        replies = [
            IkReply(
                request_id=int(r.get("id", -1)),
                ok=bool(r.get("ok")),
                joints=tuple(float(v) for v in r.get("joints") or ()),
                message=str(r.get("message", "")),
            )
            for r in request.get("replies") or []
        ]
        decision = core.next_tick(WRITER, episode_id, generation, replies)
        self._last_call = time.monotonic()
        requests = [
            {"id": r.request_id, "kind": "ik", "side": self._sides[r.group], "pose": list(r.pose)}
            for r in decision.ik_requests
        ]
        if decision.targets is None:
            return {"action": None, "requests": requests, "idle": not requests}
        return {
            "action": self.codec.encode_action(decision.targets),
            "requests": requests,
            "meta": {"command_id": decision.command_seq, "command_key": decision.command_id, "kind": decision.kind},
        }

    def on_episode_end(self, info: Mapping[str, Any]) -> dict[str, Any]:
        core = self._touch()
        episode_id, generation = self._ident(info)
        native = PlatformResult(
            native_success=None if info.get("success") is None else bool(info.get("success")),
            native_score=None if info.get("score") is None else float(info["score"]),
            terminated=True,
            truncated=info.get("failure_reason") == "timeout",
            native_reason=info.get("failure_reason"),
            take_action_cnt=info.get("take_action_cnt"),
            step_lim=info.get("step_lim"),
            protocol_version=PROTOCOL_VERSION,
            extra={
                "seed": self._private.get("seed"),
                "layout": self._private.get("layout"),
                "artifact": info.get("artifact"),
                "platform_run_id": info.get("run_id"),
            },
        )
        core.session_end({"session_id": episode_id, "generation": generation}, native)
        return {"accepted": True}

    def on_episode_interrupted(self, info: Mapping[str, Any]) -> dict[str, Any]:
        core = self._touch()
        episode_id, generation = self._ident(info)
        extra = {
            "seed": self._private.get("seed"),
            "layout": self._private.get("layout"),
            "artifact": info.get("artifact"),
            "platform_run_id": info.get("run_id"),
        }
        unknown = core.interrupt(
            {"session_id": episode_id, "generation": generation}, str(info.get("reason", "client interrupted")), extra
        )
        return {"accepted": True, "unknown_commands": unknown}

    # ------------------------------------------------------------------ overlay (calibration only)

    def overlay_images(self, frame: ObservationFrame, _session_info: Mapping[str, Any], tcp: Mapping[str, Any]):
        out = {}
        for cam in self.overlay_cameras:
            rgb = frame.images.get(cam)
            if rgb is None:
                continue
            model = self._overlay_models.get(cam)
            if model is None:
                calib = (self._private.get("camera_calibration") or {}).get(cam)
                if not calib:
                    continue
                try:
                    model = CameraModel.from_calibration(
                        calib, (int(rgb.shape[1]), int(rgb.shape[0])), axes=self._overlay_axes
                    )
                except Exception as exc:
                    if self._logger is not None:
                        self._logger.error(f"bad calibration for {cam}: {exc!r}", throttle_duration_sec=30.0)
                    continue
                self._overlay_models[cam] = model
            grippers = {self._sides[g]: pose[:3] for g, pose in tcp.items() if pose is not None}
            out[cam] = draw_overlay(rgb, model, table_z=self._table_z, grippers=grippers)
        return out


def create_adapter(profile: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> RoboDojoAdapter:
    return RoboDojoAdapter(profile, overrides)
