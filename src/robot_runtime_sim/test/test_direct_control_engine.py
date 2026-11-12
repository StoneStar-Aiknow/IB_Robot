"""Behaviour of the simulation core on two unrelated CPU simulators (V1-V4 of the design)."""

from __future__ import annotations

import dataclasses
import threading
import time

import pytest

from robot_runtime_sim.direct_control.engine import (
    OUTCOME_CANCELLED,
    OUTCOME_PLANNING_FAILED,
    OUTCOME_REJECTED,
    OUTCOME_SUCCEEDED,
    OUTCOME_UNKNOWN,
    STATE_FAILED,
    STATE_FINALIZED,
    STATE_READY,
    STATE_RESETTING,
    STATE_UNKNOWN,
    Command,
    SessionError,
    SimulationCore,
    WriterConflict,
)
from robot_runtime_sim.direct_control.host_step import HostStepLoop
from robot_runtime_sim.direct_control.ports import (
    DescriptorError,
    IkReply,
    ObservationFrame,
    PlatformResult,
    descriptor_from_profile,
)
from robot_runtime_sim.testing.direct_control_fakes import (
    FAST,
    DualArmCodec,
    DualArmEnv,
    PullPlatform,
    SingleArmCodec,
    SingleArmEnv,
    dual_arm_descriptor,
    single_arm_descriptor,
    submit_async,
    wait_for,
)


def pull_rig(step_delay_s: float = 0.0, **descriptor_overrides):
    events, native = [], []
    core = SimulationCore(
        dual_arm_descriptor(**descriptor_overrides),
        FAST,
        on_event=lambda kind, info: events.append((kind, info)),
        on_platform_result=native.append,
    )
    env = DualArmEnv(core.descriptor)
    platform = PullPlatform(core, DualArmCodec(core.descriptor), env, step_delay_s)
    return core, env, platform, events, native


# ---------------------------------------------------------------- descriptor (fail closed)


def test_descriptor_refuses_pose_without_ik_and_unknown_clock_modes():
    with pytest.raises(DescriptorError):
        dual_arm_descriptor(ik="none")
    with pytest.raises(DescriptorError):
        dual_arm_descriptor(clock_mode="realtime")


def test_descriptor_from_profile_takes_groups_dof_and_dt_from_the_profile():
    profile = {
        "motion_groups": {"arm": {"joints": [f"j{i}" for i in range(7)], "tcp_offset": [0, 0, 0.1]}},
        "simulation": {"p": {"dt_s": 0.02, "supported_commands": ["joint"]}},  # the platform's section
        "cameras": ["front"],
        "pose_frame": "base",
    }
    descriptor = descriptor_from_profile(profile, platform="p", clock_mode="host_step")
    assert descriptor.group("arm").dof == 7
    assert descriptor.dt_s == 0.02 and descriptor.cameras == ("front",)
    with pytest.raises(DescriptorError):
        descriptor_from_profile({"motion_groups": profile["motion_groups"]}, platform="p", clock_mode="host_step")


def test_unsupported_command_kind_is_rejected_not_approximated():
    core = SimulationCore(single_arm_descriptor(supported_commands=frozenset({"joint"})), FAST)
    result = core.submit(Command(kind="pose", group="arm", pose=[0.1, 0.1, 0.1, 1, 0, 0, 0]), 1.0)
    assert result["outcome"] == OUTCOME_REJECTED and "does not support 'pose'" in result["message"]
    result = core.submit(Command(kind="gripper", group="arm", opening=0.0), 1.0)
    assert result["outcome"] == OUTCOME_REJECTED


# ---------------------------------------------------------------- idle / cancel


def test_idle_does_not_advance_simulated_time():
    core, env, platform, _events, _native = pull_rig()
    platform.start()
    assert wait_for(lambda: platform.idle_polls >= 5)
    platform.join()
    assert env.tick == 0
    assert core.clock()["sim_time_s"] == 0.0


def _run_host_session(loop):
    import threading

    box = {}
    thread = threading.Thread(target=lambda: box.setdefault("r", loop.run_session({"task": "toy"})), daemon=True)
    thread.start()
    return thread, box


def test_stop_cancels_a_running_move_and_no_further_tick_is_submitted():
    core, env, platform, _events, _native = pull_rig(step_delay_s=0.01)
    platform.start()
    thread, box = submit_async(core, Command(kind="joint", group="a", joints=[1.2] * 6))  # 19 ticks (FAST cap 20)
    assert wait_for(lambda: env.tick >= 3)
    assert core.stop("operator stop") == 1
    thread.join(timeout=2.0)
    assert box["result"]["outcome"] == OUTCOME_CANCELLED
    assert 3 <= box["result"]["ticks_executed"] < 20
    ticks_at_cancel = env.tick
    assert wait_for(lambda: platform.idle_polls >= 3)
    assert env.tick <= ticks_at_cancel + 1  # at most the tick already handed over
    platform.join()


def test_wall_deadline_is_independent_of_simulated_time():
    core, _env, platform, _events, _native = pull_rig()
    # The platform never asks for a tick: the move is queued but never runs.
    result = core.submit(Command(kind="joint", group="a", joints=[0.4] * 6), 0.2)
    assert "before it ran" in result["message"]
    assert core.state == STATE_READY  # nothing was handed over: still safe
    del platform


# ---------------------------------------------------------------- generations, reset, stragglers


def test_reset_drains_commands_and_refuses_the_old_generation():
    core, env, platform, _events, _native = pull_rig()
    old = (platform.session_id, platform.generation)
    thread, box = submit_async(core, Command(kind="joint", group="a", joints=[0.5] * 6))
    assert wait_for(lambda: core._queue)  # queued, no tick yet
    ident = core.reset()
    thread.join(timeout=2.0)
    assert box["result"]["outcome"] == OUTCOME_CANCELLED
    assert ident["generation"] == old[1] + 1 and core.state == STATE_RESETTING
    stale = DualArmCodec(core.descriptor).decode_observation(env.native(), session_id=old[0], generation=old[1])
    with pytest.raises(SessionError):
        core.on_observation(stale)
    with pytest.raises(SessionError):
        core.next_tick(PullPlatform.WRITER, old[0], old[1])
    # Motion is refused until the new generation's first observation arrives.
    result = core.submit(Command(kind="joint", group="a", joints=[0.5] * 6), 1.0)
    assert result["outcome"] == OUTCOME_REJECTED and "resetting" in result["message"]


def test_late_ik_reply_from_a_previous_session_is_dropped():
    core, env, platform, _events, _native = pull_rig()
    thread, box = submit_async(core, Command(kind="pose", group="a", pose=[0.2, 0.3, 0.4, 1, 0, 0, 0]))
    assert wait_for(lambda: core._queue)
    decision = core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)
    (request,) = decision.ik_requests
    ident = core.reset()
    thread.join(timeout=2.0)
    assert box["result"]["outcome"] == OUTCOME_CANCELLED
    core.begin_session({"task": "toy"})
    core.on_observation(
        DualArmCodec(core.descriptor).decode_observation(
            env.native(), session_id=ident["session_id"], generation=ident["generation"]
        )
    )
    before = core.stale_messages
    decision = core.next_tick(
        PullPlatform.WRITER, ident["session_id"], ident["generation"], [IkReply(request.request_id, True, (1, 1, 1))]
    )
    assert decision.targets is None and core.stale_messages == before + 1


def test_duplicate_command_id_executes_once():
    core, env, platform, _events, _native = pull_rig()
    platform.start()
    first = Command(kind="joint", group="a", joints=[0.3] * 6, command_id="req-7/step-0")
    t1, b1 = submit_async(core, first)
    again = Command(kind="joint", group="a", joints=[0.3] * 6, command_id="req-7/step-0")
    t2, b2 = submit_async(core, again)
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    assert b1["result"]["outcome"] == b2["result"]["outcome"] == OUTCOME_SUCCEEDED
    assert b1["result"]["ticks"] == b2["result"]["ticks"]
    ticks_after = env.tick
    other = core.submit(Command(kind="joint", group="a", joints=[0.9] * 6, command_id="req-7/step-0"), 1.0)
    assert other["outcome"] == OUTCOME_REJECTED and "different content" in other["message"]
    assert env.tick == ticks_after
    platform.join()


def test_unknown_outcome_blocks_further_motion_until_reset():
    core, _env, platform, _events, _native = pull_rig()
    # The platform takes the pose's IK request and then never answers.
    thread, box = submit_async(core, Command(kind="pose", group="a", pose=[0.2, 0.3, 0.4, 1, 0, 0, 0]), 0.3)
    assert wait_for(lambda: core._queue)
    decision = core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)
    assert decision.ik_requests
    thread.join(timeout=2.0)
    assert box["result"]["outcome"] == OUTCOME_UNKNOWN
    assert core.state == STATE_UNKNOWN
    refused = core.submit(Command(kind="joint", group="a", joints=[0.1] * 6), 1.0)
    assert refused["outcome"] == OUTCOME_REJECTED and "unknown outcome" in refused["message"]
    # Unknown outcome: no physical tick at all, not even a hold, until the evaluator decides.
    idle = core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)
    assert idle.targets is None
    resumed = core.resume()  # an operator resume does not clear UNKNOWN
    assert (
        resumed is None
        and core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation).targets is None
    )
    # The evaluator may explicitly run out the platform's clock with hold ticks.
    assert core.finish()
    hold = core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)
    assert hold.kind == "hold" and hold.targets is not None
    core.reset()
    assert core.state == STATE_RESETTING


def test_platform_loss_with_a_command_in_flight_is_unknown():
    core, env, platform, _events, _native = pull_rig()
    thread, box = submit_async(core, Command(kind="joint", group="a", joints=[0.8] * 6))
    assert wait_for(lambda: core._queue)
    decision = core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)
    assert decision.targets is not None
    assert core.platform_lost("ws closed") == 1
    thread.join(timeout=2.0)
    assert box["result"]["outcome"] == OUTCOME_UNKNOWN and core.state == STATE_UNKNOWN


def test_only_one_writer_drives_the_tick_loop():
    core, env, platform, _events, _native = pull_rig()
    with pytest.raises(WriterConflict):
        HostStepLoop(core, DualArmCodec(core.descriptor), env)
    with pytest.raises(WriterConflict):
        core.next_tick("someone_else", platform.session_id, platform.generation)


# ---------------------------------------------------------------- motion on different bodies


def test_seven_dof_joint_move_and_wrong_dof_refusal():
    core = SimulationCore(single_arm_descriptor(), FAST)
    env = SingleArmEnv(core.descriptor, step_lim=100)
    loop = HostStepLoop(core, SingleArmCodec(core.descriptor), env)
    thread, _box = _run_host_session(loop)
    assert wait_for(lambda: core.state == STATE_READY)
    target = [0.4, -0.2, 0.3, 0.1, 0.0, 0.2, -0.1]
    result = core.submit(Command(kind="joint", group="arm", joints=target), 5.0)
    assert result["outcome"] == OUTCOME_SUCCEEDED and env.q["arm"] == pytest.approx(target)
    bad = core.submit(Command(kind="joint", group="arm", joints=[0.0] * 6), 1.0)
    assert bad["outcome"] == OUTCOME_REJECTED and "7 joint values" in bad["message"]
    loop.abort()
    thread.join(timeout=5.0)


def test_pose_move_through_platform_ik_and_planning_failure():
    core, env, platform, _events, _native = pull_rig()
    platform.start()
    ok = core.submit(Command(kind="pose", group="b", pose=[0.25, 0.35, 0.45, 1, 0, 0, 0]), 5.0)
    assert ok["outcome"] == OUTCOME_SUCCEEDED and ok["position_error_m"] < 1e-6
    assert env.q["b"][:3] == pytest.approx([0.25, 0.35, 0.45]) and env.q["a"] == [0.1] * 6
    bad = core.submit(Command(kind="pose", group="a", pose=[9.0, 0.0, 0.0, 1, 0, 0, 0]), 5.0)
    assert bad["outcome"] == OUTCOME_PLANNING_FAILED
    platform.join()


def test_pull_and_step_clock_modes_produce_the_same_ticks_for_the_same_commands():
    def commands():
        return [
            Command(kind="joint", group="arm", joints=[0.4, 0.2, 0.3, 0, 0, 0, 0]),
            Command(kind="pose", group="arm", pose=[0.1, 0.1, 0.2, 1, 0, 0, 0]),
        ]

    # host step
    core = SimulationCore(single_arm_descriptor(), FAST)
    step_env = SingleArmEnv(core.descriptor, step_lim=200)
    loop = HostStepLoop(core, SingleArmCodec(core.descriptor), step_env)
    thread, _box = _run_host_session(loop)
    assert wait_for(lambda: core.state == STATE_READY)
    step_results = [core.submit(c, 5.0) for c in commands()]
    loop.abort()
    thread.join(timeout=5.0)

    # platform pull with the same body
    core2 = SimulationCore(single_arm_descriptor(clock_mode="platform_pull"), FAST)
    pull_env = SingleArmEnv(core2.descriptor, step_lim=200)
    platform = PullPlatform(core2, SingleArmCodec(core2.descriptor), pull_env).start()
    pull_results = [core2.submit(c, 5.0) for c in commands()]
    platform.join()

    assert [r["outcome"] for r in step_results] == [r["outcome"] for r in pull_results] == [OUTCOME_SUCCEEDED] * 2
    assert [r["ticks"] for r in step_results] == [r["ticks"] for r in pull_results]
    assert step_env.steps == pull_env.steps


# ---------------------------------------------------------------- session end: verdict stays on the evaluation side


def test_native_verdict_goes_to_the_evaluation_channel_only():
    core, env, platform, events, native = pull_rig()
    core.session_end(
        {"session_id": platform.session_id, "generation": platform.generation},
        PlatformResult(native_success=True, native_score=1.0, native_reason=None, take_action_cnt=12),
    )
    assert core.state == STATE_FINALIZED
    assert native == [
        {
            "session_id": platform.session_id,
            "generation": platform.generation,
            "native_success": True,
            "native_score": 1.0,
            "terminated": True,
            "truncated": False,
            "failure_reason": None,
            "take_action_cnt": 12,
            "step_lim": None,
            "protocol_version": "",
            "interrupted": False,
        }
    ]
    for _kind, info in events:
        assert not {"native_success", "native_score", "success", "reward", "failure_reason"} & set(info)
    with pytest.raises(SessionError):
        core.on_observation(
            ObservationFrame(platform.session_id, platform.generation, 99, joints={"a": [0] * 6, "b": [0] * 6})
        )


# ---------------------------------------------------------------- review findings (21:38): regressions


def test_a_request_of_an_earlier_generation_never_runs_in_the_next_session():
    core, env, platform, _events, _native = pull_rig()
    old_generation = platform.generation
    ident = core.reset()
    core.begin_session({"task": "toy"})
    core.on_observation(
        DualArmCodec(core.descriptor).decode_observation(
            env.native(), session_id=ident["session_id"], generation=ident["generation"]
        )
    )
    late = core.submit(
        Command(kind="joint", group="a", joints=[0.5] * 6, generation=old_generation, command_id="old-request"), 1.0
    )
    assert late["outcome"] == OUTCOME_REJECTED and "not executed" in late["message"]
    decision = core.next_tick(PullPlatform.WRITER, ident["session_id"], ident["generation"])
    assert decision.targets is None and env.tick == 0


def test_the_engine_has_no_admission_rule_but_a_caller_gate_can_hold_motion_back():
    core = SimulationCore(dual_arm_descriptor(), FAST)
    env = DualArmEnv(core.descriptor)
    platform = PullPlatform(core, DualArmCodec(core.descriptor), env).start()
    # No gate: a fresh session takes motion as soon as its first observation is in.
    assert core.submit(Command(kind="joint", group="a", joints=[0.3] * 6), 5.0)["outcome"] == OUTCOME_SUCCEEDED
    opened: set[int] = set()
    seen: list[int] = []

    def gate(generation: int) -> str:
        seen.append(generation)
        return "" if generation in opened else "held back by the caller's gate"

    core.set_motion_gate(gate)
    refused = core.submit(Command(kind="joint", group="a", joints=[0.1] * 6), 1.0)
    assert refused["outcome"] == OUTCOME_REJECTED and "caller's gate" in refused["message"]
    assert seen[-1] == core.generation  # the gate decides per generation
    opened.add(core.generation)
    assert core.submit(Command(kind="joint", group="a", joints=[0.1] * 6), 5.0)["outcome"] == OUTCOME_SUCCEEDED
    core.set_motion_gate(None)
    platform.join()


def test_simulated_time_moves_only_with_observations_and_never_goes_back_at_a_reset():
    clocks: list[float] = []
    core = SimulationCore(dual_arm_descriptor(), FAST, on_clock=clocks.append)
    dt = core.descriptor.dt_s
    env = DualArmEnv(core.descriptor)
    platform = PullPlatform(core, DualArmCodec(core.descriptor), env).start()
    assert core.submit(Command(kind="joint", group="a", joints=[0.3] * 6), 5.0)["outcome"] == OUTCOME_SUCCEEDED
    first = core.sim_time_s()
    assert env.tick > 0 and first == pytest.approx(env.tick * dt)
    time.sleep(0.2)
    assert core.sim_time_s() == first  # idle: no step, the simulated clock stands still
    platform.join()
    env2 = DualArmEnv(core.descriptor)
    platform2 = PullPlatform(core, DualArmCodec(core.descriptor), env2).start()  # reset: a new session at tick 0
    assert core.sim_time_s() == pytest.approx(first)  # not back to zero
    assert core.submit(Command(kind="joint", group="a", joints=[0.1] * 6), 5.0)["outcome"] == OUTCOME_SUCCEEDED
    assert core.sim_time_s() == pytest.approx(first + env2.tick * dt)
    platform2.join()
    assert clocks and clocks == sorted(clocks) and clocks[-1] == pytest.approx(core.sim_time_s())


def test_a_tick_request_that_waited_across_a_reset_is_refused():
    import threading

    core, env, platform, _events, _native = pull_rig()
    old = (platform.session_id, platform.generation)
    box = {}

    def stale_poll():
        try:
            box["decision"] = SimulationCore.next_tick(core, PullPlatform.WRITER, *old)
        except SessionError as exc:
            box["error"] = exc

    core.cfg = dataclasses.replace(core.cfg, idle_wait_s=1.0)  # FAST is shared: do not mutate it
    thread = threading.Thread(target=stale_poll)
    thread.start()
    time.sleep(0.1)
    ident = core.reset()
    core.begin_session({"task": "toy"})
    core.on_observation(
        DualArmCodec(core.descriptor).decode_observation(
            env.native(), session_id=ident["session_id"], generation=ident["generation"]
        )
    )
    submit_async(core, Command(kind="joint", group="a", joints=[0.4] * 6, command_id="new-command"))
    thread.join(timeout=3.0)
    assert "error" in box and "decision" not in box


def test_cancelling_one_command_withdraws_it_and_holds_its_group():
    core, env, platform, _events, _native = pull_rig(step_delay_s=0.01)
    platform.start()
    queued_first = Command(kind="joint", group="a", joints=[1.2] * 6)  # 19 ticks (FAST cap 20)
    t1, b1 = submit_async(core, queued_first)
    early = Command(kind="gripper", group="a", opening=0.0)
    t2, b2 = submit_async(core, early)
    assert wait_for(lambda: early in core._queue)
    assert core.cancel(early)  # cancelled before it ran: never queued for a tick
    t2.join(timeout=2.0)
    assert b2["result"]["outcome"] == OUTCOME_CANCELLED and "ticks_executed" not in b2["result"]
    assert wait_for(lambda: env.tick >= 2)
    assert core.cancel(queued_first)  # cancelled while running
    t1.join(timeout=2.0)
    assert b1["result"]["outcome"] == OUTCOME_CANCELLED and b1["result"]["ticks_executed"] < 20
    ticks = env.tick
    assert wait_for(lambda: platform.idle_polls >= 3)
    assert env.tick <= ticks + 1 and env.grip["a"] == 1.0
    assert not core.cancel(queued_first)  # already finished
    platform.join()


def test_stop_after_finish_stops_ticking_until_resumed():
    core, env, platform, _events, _native = pull_rig(step_delay_s=0.005)
    platform.start()
    assert core.finish()
    assert wait_for(lambda: env.tick >= 3)
    core.stop("operator stop")
    time.sleep(0.05)
    at_stop = env.tick
    time.sleep(0.3)
    assert env.tick <= at_stop + 1  # at most the tick already handed over
    assert core.halted and core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation).targets is None
    refused = core.submit(Command(kind="joint", group="a", joints=[0.2] * 6), 0.5)
    assert refused["outcome"] == OUTCOME_REJECTED and "stopped" in refused["message"]
    core.resume()
    assert core.submit(Command(kind="joint", group="a", joints=[0.2] * 6), 5.0)["outcome"] == OUTCOME_SUCCEEDED
    platform.join()


def test_a_blocked_rotation_or_missing_pose_feedback_is_not_reached():
    class NoTurn(DualArmEnv):
        def flange(self, group):  # the wrist never turns: orientation stays identity
            q = self.q[group]
            return [q[0], q[1], q[2], 1.0, 0.0, 0.0, 0.0]

    core2 = SimulationCore(dual_arm_descriptor(), FAST)
    env2 = NoTurn(core2.descriptor)
    p2 = PullPlatform(core2, DualArmCodec(core2.descriptor), env2).start()
    # same position, target turned by pi about z
    result = core2.submit(Command(kind="pose", group="a", pose=[0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 1.0]), 5.0)
    assert result["outcome"] == "EXECUTION_FAILED" and "orientation" in result["message"]
    assert result["position_error_m"] < 1e-9 and result["orientation_error_rad"] > 3.0
    p2.join()

    class Blind(DualArmCodec):
        def decode_observation(self, native, *, session_id, generation):
            frame = super().decode_observation(native, session_id=session_id, generation=generation)
            return ObservationFrame(frame.session_id, frame.generation, frame.step_id, frame.joints, frame.grippers)

    core3 = SimulationCore(dual_arm_descriptor(), FAST)
    p3 = PullPlatform(core3, Blind(core3.descriptor), DualArmEnv(core3.descriptor)).start()
    blind = core3.submit(Command(kind="pose", group="a", pose=[0.1, 0.1, 0.1, 1.0, 0.0, 0.0, 0.0]), 5.0)
    assert blind["outcome"] == "EXECUTION_FAILED" and "no end-effector pose" in blind["message"]
    p3.join()


# ---------------------------------------------------------------- alignment review (F1/F2/F5)


def test_cancel_by_id_withdraws_only_that_command():
    core, env, platform, _events, _native = pull_rig(step_delay_s=0.01)
    platform.start()
    thread, box = submit_async(core, Command(kind="joint", group="a", joints=[1.2] * 6, command_id="task:step0:move"))
    assert wait_for(lambda: env.tick >= 3)
    cancelled, result = core.cancel_by_id("task:step0:move")
    at_ack = env.tick
    thread.join(timeout=2.0)
    assert cancelled and result["outcome"] == OUTCOME_CANCELLED and box["result"]["outcome"] == OUTCOME_CANCELLED
    time.sleep(0.2)
    assert env.tick <= at_ack + 1
    again, done = core.cancel_by_id("task:step0:move")
    assert not again and done["outcome"] == OUTCOME_CANCELLED
    assert core.cancel_by_id("")[0] is False  # nothing to cancel without an id
    platform.join()


def test_an_interrupted_session_halts_and_ends_without_a_platform_result():
    core, env, platform, events, native = pull_rig()
    thread, box = submit_async(core, Command(kind="joint", group="a", joints=[0.7] * 6))
    assert wait_for(lambda: core._queue)
    core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)  # handed over
    unknown = core.interrupt({"session_id": platform.session_id, "generation": platform.generation}, "client lost")
    thread.join(timeout=2.0)
    assert unknown == 1 and box["result"]["outcome"] == OUTCOME_UNKNOWN and core.halted
    (end,) = [info for kind, info in events if kind == "session_end"]
    assert end["interrupted"] is True and native[0]["native_success"] is None
    assert core.platform_lost("again") == 0 and len(native) == 1  # published once
    # the client's own interruption report after an abort is accepted, not an error
    assert core.interrupt({"session_id": platform.session_id, "generation": platform.generation}, "late") == 0
    with pytest.raises(SessionError):
        core.next_tick(PullPlatform.WRITER, platform.session_id, platform.generation)


def test_inconsistent_descriptors_and_profiles_refuse_to_start():
    from robot_runtime_sim.direct_control.ports import GroupSpec, ik_request_refusal, profile_mismatches

    for bad in (
        {"dt_s": float("inf")},
        {"dt_s": float("nan")},
        {"pose_frame": ""},
        {"groups": (GroupSpec(name="a", joints=("j",), tcp_offset=(0.0, 0.1)),)},
        {"groups": (GroupSpec(name="a", joints=("j",), tcp_offset=(0.0, float("nan"), 0.0)),)},
        {"groups": (GroupSpec(name="a", joints=("j", "k"), joint_wrap_limits=(1.0,)),)},
    ):
        with pytest.raises(DescriptorError):
            dual_arm_descriptor(**bad)
    descriptor = dual_arm_descriptor()
    profile = {
        "capabilities": {
            "motion.move_to_pose": {"groups": ["a", "b"]},
            "motion.ik": {},
            "gripper.1d": {"count": 2},
        },
        "simulation": {"fake_pull": {"clock_mode": "platform_pull"}},
    }
    assert profile_mismatches(profile, descriptor) == []
    assert profile_mismatches({**profile, "simulation": {"fake_pull": {"clock_mode": "host_step"}}}, descriptor)
    # another platform's section says nothing about this one
    assert profile_mismatches({**profile, "simulation": {"other": {"clock_mode": "host_step"}}}, descriptor) == []
    no_ik = {"capabilities": {k: v for k, v in profile["capabilities"].items() if k != "motion.ik"}}
    assert any("motion.ik" in p for p in profile_mismatches(no_ik, descriptor))
    # ComputeIk: unsupported requests are refused, never approximated
    ok = dict(frame_id="", seed_names=[], orientation_tolerance=0.0, timeout_s=0.0)
    assert ik_request_refusal(descriptor, **ok) == ""
    assert ik_request_refusal(descriptor, **{**ok, "frame_id": "world"}) == ""
    assert "not the runtime pose frame" in ik_request_refusal(descriptor, **{**ok, "frame_id": "base_link"})
    assert "seed" in ik_request_refusal(descriptor, **{**ok, "seed_names": ["j1"]})
    assert ik_request_refusal(descriptor, **{**ok, "orientation_tolerance": 0.01})
    tolerant = dual_arm_descriptor(ik_orientation_tolerance_rad=0.05)
    assert ik_request_refusal(tolerant, **{**ok, "orientation_tolerance": 0.1}) == ""
    assert "tighter" in ik_request_refusal(tolerant, **{**ok, "orientation_tolerance": 0.01})
    assert ik_request_refusal(descriptor, **{**ok, "timeout_s": float("nan")})


def test_a_cancel_that_overtakes_its_command_leaves_a_tombstone():
    """R3-1: cancel before the runtime registered the command -> the late command runs 0 ticks."""
    core, env, platform, _events, _native = pull_rig(step_delay_s=0.01)
    platform.start()
    cancelled, result = core.cancel_by_id("late:step0:move", "task cancelled")
    assert cancelled and result["outcome"] == "NOT_REGISTERED" and result["ticks_executed"] == 0
    late = core.submit(Command(kind="joint", group="a", joints=[0.5] * 6, command_id="late:step0:move"), 2.0)
    assert late["outcome"] == OUTCOME_CANCELLED and late["ticks_executed"] == 0
    time.sleep(0.2)
    assert env.tick == 0
    # an ordinary id is unaffected; a new generation forgets the tombstone
    assert core.submit(Command(kind="joint", group="a", joints=[0.3] * 6, command_id="other"), 5.0)["outcome"] == (
        OUTCOME_SUCCEEDED
    )
    platform.join()
    core.reset()
    assert "late:step0:move" not in core._tombstones


# ---------------------------------------------------------------- review round 2 (F2, F4, F5)


def test_a_command_cancelled_before_its_submit_is_never_queued_or_ticked():
    """F2: a cancel watcher that wins the race against submit: CANCELLED, no target, no tick."""
    core, env, platform, _events, _native = pull_rig(step_delay_s=0.01)
    platform.start()
    cmd = Command(kind="gripper", group="a", opening=0.0)
    assert core.cancel(cmd, "gripper goal cancelled")  # the watcher ran first: cmd not yet registered
    result = core.submit(cmd, 2.0)
    assert result["outcome"] == OUTCOME_CANCELLED
    assert cmd not in core._queue and core._active is None and not cmd.command_id
    assert wait_for(lambda: platform.idle_polls >= 3)
    assert env.tick == 0 and env.grip["a"] == 1.0
    platform.join()


def test_an_ended_command_left_in_the_queue_is_skipped_not_ticked():
    """F2 (defensive): whatever ended while queued never reaches the platform."""
    core, env, platform, _events, _native = pull_rig()
    ended = Command(kind="gripper", group="a", opening=0.0, seq=99, command_id="ended")
    ended.finish(OUTCOME_CANCELLED, "ended elsewhere")
    with core._cv:
        core._queue.append(ended)
    decision = core.next_tick(platform.WRITER, platform.session_id, platform.generation)
    assert decision.targets is None and not core._queue and core._active is None
    assert env.tick == 0


def test_a_move_that_needs_more_ticks_than_allowed_is_refused_not_sped_up():
    """F4: max_motion_ticks bounds a move; it never compresses one (which would raise its speed)."""
    core, env, platform, _events, _native = pull_rig()
    platform.start()
    # 1.9 rad at FAST speed (0.06 rad/tick) needs 32 ticks; the cap is 20.
    slow = core.submit(Command(kind="joint", group="a", joints=[2.0] * 6), 2.0)
    assert slow["outcome"] == OUTCOME_REJECTED and "needs 32 ticks" in slow["message"]
    assert "max_motion_ticks=20" in slow["message"] and slow["ticks_executed"] == 0
    # a pose: refused after its IK reply, still before any physical tick
    far = core.submit(Command(kind="pose", group="b", pose=[2.1, 0.1, 0.1, 1, 0, 0, 0], velocity_scaling=0.5), 2.0)
    assert far["outcome"] == OUTCOME_REJECTED and "max_motion_ticks" in far["message"]
    assert env.tick == 0
    # within the budget: no tick moves faster than max_joint_speed * velocity_scaling
    ok = core.submit(Command(kind="joint", group="a", joints=[0.5] * 6, velocity_scaling=0.5), 5.0)
    assert ok["outcome"] == OUTCOME_SUCCEEDED and env.q["a"] == pytest.approx([0.5] * 6)
    limit = FAST.max_joint_speed * 0.5 * core.descriptor.dt_s
    path = [[0.1] * 6] + [step["a"]["joints"] for step in env.steps]
    assert (
        max(abs(b - a) for prev, cur in zip(path, path[1:], strict=False) for a, b in zip(prev, cur, strict=True))
        <= limit + 1e-9
    )
    platform.join()


class _FailingStepEnv(SingleArmEnv):
    """The host-step toy whose reset, IK or step call raises (after two good steps)."""

    def __init__(self, descriptor, fail_on: str):
        super().__init__(descriptor, step_lim=1000)
        self.fail_on = fail_on

    def reset(self, seed=None):
        if self.fail_on == "reset":
            raise RuntimeError("reset failed")
        return super().reset(seed)

    def solve_ik(self, request):
        if self.fail_on == "ik":
            raise RuntimeError("ik failed")
        return super().solve_ik(request)

    def step(self, native_action):
        if self.fail_on == "step" and self.tick >= 2:
            raise RuntimeError("step failed")
        return super().step(native_action)


@pytest.mark.parametrize("fail_on", ["reset", "ik", "step"])
def test_a_failing_host_step_platform_call_interrupts_its_session_once(fail_on):
    """F5: in-flight work ends at once, one interrupted end, no verdict, no later physics tick."""
    native = []
    core = SimulationCore(single_arm_descriptor(), FAST, on_platform_result=native.append)
    env = _FailingStepEnv(core.descriptor, fail_on)
    loop = HostStepLoop(core, SingleArmCodec(core.descriptor), env)
    box = {}

    def run():
        try:
            loop.run_session({"task": "toy"})
        except RuntimeError as exc:
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    if fail_on != "reset":
        assert wait_for(lambda: core.state == STATE_READY)
        if fail_on == "ik":
            cmd = Command(kind="pose", group="arm", pose=[0.3, 0.2, 0.1, 1, 0, 0, 0])
        else:
            cmd = Command(kind="joint", group="arm", joints=[0.5] * 7)
        started = time.monotonic()
        result = core.submit(cmd, 5.0)
        assert time.monotonic() - started < 2.0  # ended by the interrupt, not by its wall deadline
        assert result["outcome"] == OUTCOME_UNKNOWN  # handed to the platform when the call failed
    thread.join(timeout=5.0)
    assert not thread.is_alive() and f"{fail_on} failed" in str(box["error"])
    assert len(native) == 1 and native[0]["interrupted"] is True and native[0]["native_success"] is None
    assert core.state in (STATE_FAILED, STATE_UNKNOWN)
    steps = env.tick
    time.sleep(0.1)
    assert env.tick == steps


def test_a_stale_host_step_driver_does_not_interrupt_the_next_session():
    """F5: an old driver's failure never ends a newer generation."""
    native = []
    core = SimulationCore(single_arm_descriptor(), FAST, on_platform_result=native.append)
    loop = HostStepLoop(core, SingleArmCodec(core.descriptor), SingleArmEnv(core.descriptor))
    old = core.reset()
    current = core.reset()
    core.begin_session({"task": "toy"})
    loop._interrupt({"session_id": old["session_id"], "generation": old["generation"]}, RuntimeError("late"))
    assert core.state == STATE_RESETTING and core.generation == current["generation"] and native == []
