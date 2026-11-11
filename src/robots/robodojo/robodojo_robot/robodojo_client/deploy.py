"""RoboDojo eval-client side of the IB_Robot virtual body (runs inside the Isaac process).

RoboDojo imports ``XPolicyLab.policy.<name>.deploy.eval_one_episode`` and calls it once per
episode from ``run_eval``. The policy server behind ``model_client`` is the IB_Robot
``robodojo_runtime`` ROS node (robot_runtime_sim's direct-control runtime with ``robodojo_robot.adapter``). This loop:

* pushes the observation after every physics tick (and once at episode start),
* asks for the next tick: ``action`` = 14-dim joint targets to execute, or ``None`` = nothing
  to do (physics is NOT stepped, so the Agent's thinking time costs no episode ticks),
* answers IK requests with RoboDojo's own cuRobo solver (``robot_manager.solve_ik``), which
  only exists in this process; the replies travel with the next ``get_action``,
* reports the native verdict (``TASK_ENV.success``) with ``episode_end`` and writes the
  per-episode artifact (``artifacts.EpisodeRecorder``: every tick's action, every motion
  command, native success, failure reason). The runtime forwards the verdict to the
  evaluation channel only; the Agent never sees it.

Every call carries the runtime's ``episode_id`` and ``generation`` from ``reset``; a reply
or observation of an older generation is refused by the runtime. When a call is refused or
the runtime cannot be reached, the next action is unknown: the episode is interrupted at
once (no further ``take_action``), recorded as ``interrupted`` without a native verdict,
flagged unstable so RoboDojo does not score it, and the exception stops this seed.

No ROS import here: this file runs under RoboDojo's Python 3.11.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .artifacts import EpisodeRecorder, NativeOutcome

BODY_KEY = "_body"


# Once the runtime answered reset, a lost connection means the episode's runtime is gone: retry
# for at most this long (RoboDojo's own default waits 15 min for a cold-starting policy server).
RECONNECT_BUDGET_S = float(os.environ.get("IBROBOT_RUNTIME_RECONNECT_S", "30"))


def _bound_reconnect(model_client: Any, budget_s: float) -> bool:
    """Tighten RoboDojo's ws client retry budget for this session; False if it has no such config.

    Uses RoboDojo's own PolicyEvalClientConfig fields (read at each connect attempt); nothing in the
    RoboDojo checkout is changed.
    """
    config = getattr(getattr(model_client, "_client", None), "config", None)
    if config is None or not hasattr(config, "max_connect_seconds"):
        return False
    config.max_connect_seconds = budget_s
    delay = max(float(getattr(config, "connect_retry_delay_s", 5.0)), 0.1)
    config.max_connect_attempts = max(1, int(budget_s / delay) + 1)
    return True


def _checked(result: Any, call: str, key: str) -> Mapping[str, Any]:
    if not isinstance(result, Mapping) or key not in result:
        raise RuntimeError(f"virtual runtime {call} failed (got {result!r}); see the robodojo_runtime log")
    return result


def _as_env_action(action: Mapping[str, Sequence[float]]):
    import numpy as np

    return {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}


def _episode_identity(TASK_ENV, env_idx: int = 0) -> tuple[Any, dict[str, Any]]:
    seed = None
    seeds = getattr(TASK_ENV, "env_seeds", None)
    if seeds is not None:
        seed = int(seeds[env_idx])
    layout: dict[str, Any] = {"env_seed": seed, "eval_seed": getattr(TASK_ENV, "eval_seed", None)}
    try:
        layout["scene_layout"] = TASK_ENV.seed_manager.seed_info[seed]["scene_layout"]
    except Exception as exc:  # layout file is metadata only
        layout["scene_layout_error"] = repr(exc)
    return seed, layout


def _native_outcome(TASK_ENV, env_idx: int = 0) -> NativeOutcome:
    try:
        layout_valid = bool(TASK_ENV.scene_manager.layout_manager.layout_valid[env_idx])
    except Exception:
        layout_valid = True
    return NativeOutcome(
        success=bool(TASK_ENV.success[env_idx]),
        take_action_cnt=int(TASK_ENV.take_action_cnt[env_idx]),
        step_lim=int(TASK_ENV.step_lim),
        layout_valid=layout_valid,
        unstable=env_idx in getattr(TASK_ENV, "unstable_envs", set()),
    )


def solve_ik(TASK_ENV, request: Mapping[str, Any], env_idx: int = 0) -> dict[str, Any]:
    """Answer one IK request with RoboDojo's planner; never raises."""
    reply: dict[str, Any] = {"id": request.get("id"), "ok": False}
    try:
        manager = TASK_ENV.robot_manager
        robot = manager.get_robot_by_arm_name(f"{request['side']}_arm")
        result = manager.solve_ik(target_pose=[float(v) for v in request["pose"]], env_idx=env_idx, robot=robot)
        if not isinstance(result, Mapping) or result.get("status") != "Success":
            reply["message"] = str(result.get("status") if isinstance(result, Mapping) else result)
            return reply
        values = result["joint_value"]
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        flat = [float(v) for v in (values[0] if values and isinstance(values[0], list | tuple) else values)]
        if len(flat) < 6:
            reply["message"] = f"IK returned {len(flat)} joints"
            return reply
        reply.update(ok=True, joints=flat[:6])
    except Exception as exc:
        reply["message"] = f"IK error: {exc!r}"
    return reply


def _camera_calibration(TASK_ENV, env_idx: int = 0) -> dict[str, Any]:
    """Per camera: intrinsics K (for the spec resolution) and the camera-to-world pose at episode start.

    Sensor calibration only (what a real robot has from its camera calibration), no scene content.
    Wrist cameras move with the arm, so only fixed cameras' poses stay valid during the episode.
    """
    out: dict[str, Any] = {}
    try:
        manager = TASK_ENV.camera_manager
        from env.camera_manager.camera_manager import REAL_MAP

        for ith, name in enumerate(manager.camera_names[env_idx]):
            spec = REAL_MAP.get(manager.camera_config[name].camera.type) or {}
            k = manager.get_camera_intrinsics(ith, env_idx)
            if "resolution" not in spec or float(k[0][0]) == 1.0:
                continue  # no focal length known for this camera type
            out[str(name)] = {
                "K": [[float(v) for v in row] for row in k],
                "resolution": [int(v) for v in spec["resolution"]],
                "cam_to_world": [[float(v) for v in row] for row in manager.get_camera_extrinsics(ith, env_idx)],
            }
    except Exception as exc:  # calibration is an aid, never a reason to fail the episode
        out["error"] = repr(exc)
    return out


def _instruction(TASK_ENV, obs: Mapping[str, Any]) -> str:
    text = obs.get("instruction")
    if text:
        return str(text)
    return str(getattr(TASK_ENV, "instruction", "") or getattr(TASK_ENV, "task_name", ""))


def eval_one_episode(TASK_ENV, model_client, env_idx: int = 0) -> dict[str, Any]:
    run_id = os.environ.get("ROBODOJO_RUN_ID", "norunid")
    task = str(getattr(TASK_ENV, "task_name", "unknown_task"))
    out_dir = os.path.join(
        os.environ.get("IBROBOT_RUNTIME_ARTIFACT_DIR", os.path.join("eval_result", "ibrobot_virtual_runtime")),
        run_id,
        task,
    )
    seed, layout = _episode_identity(TASK_ENV, env_idx)
    first_obs = TASK_ENV.get_obs()
    episode_meta = {
        "run_id": run_id,
        "task": task,
        "seed": seed,
        "layout": layout,
        "instruction": _instruction(TASK_ENV, first_obs),
        "step_lim": int(TASK_ENV.step_lim),
        "camera_calibration": _camera_calibration(TASK_ENV, env_idx),
    }
    # reset() carries no payload over ws; the episode description goes with begin_episode.
    ident = _checked(model_client.call(func_name="reset"), "reset", "episode_id")
    bounded = _bound_reconnect(model_client, RECONNECT_BUDGET_S)
    info = _checked(model_client.call(func_name="begin_episode", obs=episode_meta), "begin_episode", "episode_id")
    episode = {"episode_id": info["episode_id"], "generation": ident["generation"]}
    session_id = info["episode_id"]
    if not bounded:
        print("[ibrobot_virtual_runtime] ws client has no retry config; reconnect stays RoboDojo's default")
    recorder = EpisodeRecorder(
        run_id=run_id,
        task=task,
        seed=seed,
        layout=layout,
        episode_id=f"seed{seed}_{session_id}",
        session_id=session_id,
    )

    obs = first_obs
    replies: list[dict[str, Any]] = []
    seen_commands: set[int] = set()
    push_obs = True
    idle_polls = 0
    t_episode = time.perf_counter()
    while not TASK_ENV.is_episode_end():
        tick = int(TASK_ENV.take_action_cnt[env_idx])
        try:
            if push_obs:
                obs[BODY_KEY] = {**episode, "tick": tick}
                _checked(model_client.call(func_name="update_obs", obs=obs), "update_obs", "accepted")
                push_obs = False
            res = _checked(
                model_client.call(func_name="get_action", obs={**episode, "tick": tick, "replies": replies}),
                "get_action",
                "action",
            )
        except Exception as exc:
            # The runtime refused this episode or cannot be reached: the action to take is unknown.
            # No more physics is stepped for it (no drain with a guessed hold).
            return _interrupt(
                TASK_ENV, model_client, recorder, episode, run_id, out_dir, env_idx, f"tick {tick}: {exc!r}", t_episode
            )
        replies = [solve_ik(TASK_ENV, req, env_idx) for req in res.get("requests") or []]
        action = res["action"]
        if action is None:
            idle_polls += 1
            continue  # nothing to execute: do not step physics
        meta = res.get("meta") or {}
        cid = meta.get("command_id")
        if cid is not None and cid not in seen_commands:
            seen_commands.add(cid)
            recorder.record_decision(cid, tick, kind=meta.get("kind"))
        recorder.record_tick(tick, action, decision_id=cid)
        TASK_ENV.take_action(_as_env_action(action))
        obs = TASK_ENV.get_obs()
        push_obs = True

    outcome = _native_outcome(TASK_ENV, env_idx)
    final_tick = int(TASK_ENV.take_action_cnt[env_idx])
    try:
        # Upload the observation after the last executed tick so the runtime's clock ends on the
        # platform's own count (the loop exits before pushing it).
        if push_obs:
            obs[BODY_KEY] = {**episode, "tick": final_tick}
            model_client.call(func_name="update_obs", obs=obs)
    except Exception as exc:
        print(f"[ibrobot_virtual_runtime] final observation not delivered: {exc!r}")
    artifact = recorder.finalize(outcome)
    artifact["idle_polls"] = idle_polls
    artifact["wall_s"] = round(time.perf_counter() - t_episode, 2)
    path = recorder.write(out_dir)
    try:
        model_client.call(
            func_name="episode_end",
            obs={
                **episode,
                "run_id": run_id,
                "success": bool(outcome.success),
                "failure_reason": artifact["failure_reason"],
                "take_action_cnt": outcome.take_action_cnt,
                "step_lim": outcome.step_lim,
                "artifact": os.path.abspath(path),
            },
        )
    except Exception as exc:
        print(f"[ibrobot_virtual_runtime] episode_end not delivered: {exc!r}")
    print(
        f"[ibrobot_virtual_runtime] {task} seed={seed} native_success={artifact['success']} "
        f"reason={artifact['failure_reason']} ticks={len(artifact['ticks'])} "
        f"commands={artifact['decision_count']} -> {path}"
    )
    return artifact


class EpisodeInterrupted(RuntimeError):
    """The runtime could not be asked for the next action; the episode was stopped, not judged."""


def _interrupt(TASK_ENV, model_client, recorder, episode, run_id, out_dir, env_idx, reason, t_episode):
    """Stop the episode without stepping: write an interrupted artifact, tell the runtime, abort.

    The env is flagged unstable so RoboDojo neither scores it nor keeps its video, and the raise
    makes RoboDojo's main loop drop this seed (no artificial native result).
    """
    print(f"[ibrobot_virtual_runtime] {reason}; episode interrupted, no further physics stepped")
    unstable = getattr(TASK_ENV, "unstable_envs", None)
    if isinstance(unstable, set):
        unstable.add(env_idx)
    artifact = recorder.finalize_interrupted(reason, take_action_cnt=int(TASK_ENV.take_action_cnt[env_idx]))
    artifact["wall_s"] = round(time.perf_counter() - t_episode, 2)
    path = recorder.write(out_dir)
    try:
        model_client.call(
            func_name="episode_interrupted",
            obs={**episode, "run_id": run_id, "reason": reason, "artifact": os.path.abspath(path)},
        )
    except Exception as exc:
        print(f"[ibrobot_virtual_runtime] interruption not delivered: {exc!r}")
    raise EpisodeInterrupted(reason)
