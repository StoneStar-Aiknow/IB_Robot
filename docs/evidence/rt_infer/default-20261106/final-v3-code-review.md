# RT-Infer default integration code review

## Result

- `codeQualityStatus`: `CLEAR`
- `recommendation`: `APPROVE`
- Scope reviewed: current dirty implementation in `/tmp/IB_Robot-pr581-oee-summit`, with emphasis on default owner lifecycle, wrapper execution, Ascend readiness, periodic admission, and asynchronous teardown. This review intentionally excludes the in-progress final-v3 performance matrix.

## Evidence inspected

- The current default board profile selects `2026.11.06` at `src/robot_config/config/robots/aimdk_x2_magicmirror_board.yaml:41-48`. Its manifest-pinned payload is installed as `robot_config` share data by `src/robot_config/setup.py:19-56`; the resolver rejects incomplete source or installed bundles at `src/robot_config/robot_config/rt_infer_bundle.py:46-105`.
- Owner startup acquires a nonblocking lease, rechecks idleness, confirms a live fixed scheduler identity, and deliberately leaves unproven starts untouched at `src/robot_config/robot_config/launch_builders/rt_infer_scheduler.py:195-235`, `390-506`.
- A supervisor exit outside launch shutdown raises into `LaunchService`; the installed launch implementation catches that exception, emits launch shutdown, and keeps dispatching shutdown work. The owner registers this behaviour at `rt_infer_scheduler.py:565-608`.
- Teardown only signals token- and pid/start-tick-verified clients, drains them, then stops the pinned scheduler. Any verification ambiguity returns failure without widening the kill set: `src/robot_config/robot_config/rt_infer_scheduler_supervisor.py:82-148`, `266-349`.
- The wrapper executes the real `module:function` in a child process, records its identity token atomically, and preserves client ownership through shutdown at `src/inference_service/inference_service/rt_infer_client.py:86-128`, `131-207`.
- PEAR admission validates the owner/session/scheduler identity and leaves mapping disarm/removal to the owner: `src/inference_service/inference_service/rt_infer_periodic.py:95-154`, `156-206`; Ascend sessions mark readiness only after three successful executions per deployment role at `src/inference_service/inference_service/model_sessions/ascend.py:293-351`.
- The mutable-state repair uses component-wise `O_NOFOLLOW` directory traversal, lock inode and directory identity verification, and only ignores disappearance of the three exact atomic-publish temporary names at `third_party/vendor/rt-infer/2026.11.06/scripts/base_control.py:144-233`.
- Test evidence: `/tmp/rtinfer-default-finish/colcon-controlfix.log` records `83` selected `robot_config`, `31` selected `inference_service`, and `1` selected `visual_interaction` RT-Infer tests passing. It includes the state race/security tests at log lines 103-128. The earlier `/tmp/rtinfer-default-finish/state-security-unit.log` failure predates the current state-security test and the later colcon run (19:26 versus 19:29-19:30 on November 6, 2026), so it is not evidence against the reviewed current tree. `git diff --check` was clean.

## Skill-perspective check

Unavailable. Neither `remove-ai-slops` nor `programming` was installed or exposed in this session; their stated criteria were applied directly. The reviewed production code and tests do not show deletion-only coverage, prompt-string tests, tautological assertions, implementation-constant mirroring as the primary assertion, untyped escape hatches, or unnecessary parsing/normalization for this goal. The tests exercise observable subprocess invocation, cross-process atomic rename, ownership refusal, state integrity, and launch teardown. No violation of either perspective was found.

## Findings

### CRITICAL

None.

### HIGH

None.

### MEDIUM

None.

### LOW

None.

## Residual validation boundary

The final-v3 board performance matrix was still running and was intentionally excluded from this code-quality decision. Its completion is required for performance acceptance, but it is not a code-review blocker for the current lifecycle, safety, and test changes.
