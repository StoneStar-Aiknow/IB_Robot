# Lightweight Skill Package + Registry Design

Status: Revised Draft
Date: 2026-08-04

## 1. Revision Summary

This revision applies the review feedback to the earlier hot-reload proposal.
It corrects five structural problems:

1. `registry_epoch` is introduced so process restarts do not reuse the same version namespace.
2. `SkillRegistry` is separated from the process-local execution coordinator.
3. Snapshot sync uses exact version identity, not only `min_generation`.
4. Skill requests are bound to an expected registry version.
5. Reload uses a staged immutable catalog and atomic swap, not live-tree partial reads.

It also fixes the manifest examples so they match the current validator contract:

- `when_to_use` stays a list.
- `motion_scope` stays a list.
- `do_not_use` stays a list of objects.
- `rule_entry` and `requires_motion_params` stay explicit fields.

This is a design document only. It does not change runtime code.

## 2. Context

The current IB-Robot runtime still carries skill definitions inline in `robot_config`
YAML and injects them as startup JSON into `skill_executor`, `safety_guard`, and
the planners. That keeps skills and hardware facts in one file, but it also makes
skill iteration expensive and forces full-node restart for simple YAML changes.

The target direction remains:

- `robot_config` owns stable robot hardware facts.
- A dedicated skill catalog owns skill definitions and profiles.
- Runtime nodes consume immutable snapshots, not ad hoc shared globals.

The revised design below keeps that direction, but removes the versioning and
restart gaps from the previous draft.

## 3. Goals

1. Split skill data from robot hardware data.
2. Keep skill definitions reloadable without changing control-mode or joint SSOT.
3. Make every consumer observe the same catalog version identity.
4. Fail closed on reload, mismatch, or unknown snapshot state.
5. Preserve active executions across reloads while preventing new goals from mixing versions.
6. Make production packaging atomic and deterministic.

## 4. Non-goals

1. Third-party skill marketplace or online installation.
2. Runtime loading of new Python or C++ executor code.
3. Hot reload of ROS message/service/action definitions.
4. Hot reload of robot joint limits, controller names, named poses, or workspace bounds.
5. Automatic crash recovery of in-flight actions across process restart.

## 5. Architecture Boundary

The boundary is revised as follows:

```text
robot_config  ----->  skill_catalog  ----->  skill_library
      ^                    ^                      ^
      |                    |                      |
      |                    |                      +--> safety_guard / planners / CLI consume snapshots
      |                    +--> pure data compiler and registry
      +--> hardware facts and robot context
```

Responsibilities:

- `robot_config`: robot name, joints, control modes, named poses, workspace limits,
  timeout policy, and robot-specific execution context.
- `skill_catalog`: manifest/profile loading, schema validation, canonical digest,
  snapshot compilation, and source discovery.
- `skill_library`: runtime execution coordinator, admission control, bundle swap,
  and action dispatch.
- `safety_guard`, planners, CLI: snapshot consumers only.

`skill_catalog` must not import `robot_config` or `skill_library`. It receives a
plain `SkillRobotContext` object from the caller.

## 6. Version Model

### 6.1 Identity Fields

Catalog identity is a tuple:

```text
(registry_epoch, generation, registry_digest)
```

Definitions:

- `registry_epoch`: UUID generated when the runtime owner activates a catalog.
- `generation`: monotonic `uint64` within one epoch.
- `registry_digest`: hash of the immutable internal snapshot.
- `capability_digest`: hash of the public capability view only.

### 6.2 Why Generation Alone Is Not Enough

If the process restarts and generation starts at `1` again, consumers that still
hold a higher generation will treat the new snapshot as stale forever. That makes
restart recovery impossible.

Using an epoch solves this:

- A process restart creates a new `registry_epoch`.
- Consumers reject older epochs explicitly.
- Version comparison is always explicit, never implicit.

### 6.3 Retention Rule

The registry may evict historical snapshots only after no active execution retains
that snapshot identity. If the runtime crashes, the process-local active state is
gone and the system must come back in fail-closed mode.

No silent resumption is promised after crash.

## 7. Package Layout

The package layout stays logical, not a plugin marketplace:

```text
src/skill_catalog/
├── package.xml
├── setup.py
├── skill_catalog/
│   ├── models.py
│   ├── source.py
│   ├── compiler.py
│   ├── validator.py
│   ├── digest.py
│   └── registry.py
├── config/
│   ├── schemas/
│   ├── profiles/
│   └── skills/
└── test/
```

Runtime state does not live in this package.

## 8. Skill File Contract

### 8.1 `manifest.yaml`

The manifest is the skill SSOT. The document contract must match the current
validator, not a simplified textual sketch.

Example:

```yaml
schema_version: 1
name: wave_hello
version: 1.0.0

description:
  summary: Perform a casual side-to-side greeting wave.
  category: social_greeting
  when_to_use:
    - greet a person
    - acknowledge a user
  aliases_zh: [挥手, 打招呼, 再见]
  aliases_en: [wave, say hello]
  motion_scope: [arm]
  intensity: moderate
  duration_sec_estimate: 3.0
  requires_motion_params: false
  rule_entry: true
  do_not_use:
    - condition: workspace is obstructed
      instead_use: inspect_scene

capability:
  schema_version: 1
  summary: Perform a greeting wave.
  domain: social
  moves_robot: true
  required_control_mode: moveit_planning
  parameters:
    type: object
    additionalProperties: false
    properties: {}
    required: []
  recovery_policy: safe_pose

implementations:
  so101_single_arm: implementations/so101_single_arm.yaml
```

Rules:

1. `name` must match the directory name.
2. `when_to_use` must be a non-empty list.
3. `motion_scope` must be a list.
4. `do_not_use` must be a list of `{condition, instead_use}` objects.
5. `rule_entry` is part of the schema because the rule parser uses it.
6. `SKILL.md` is documentation only and must not drive runtime behavior.

### 8.2 Implementation File

The implementation file is robot-variant specific:

```yaml
schema_version: 1
kind: primitive_sequence
robot: so101_single_arm

initial_gripper_state: closed
timeout_sec: 20.0

primitive_sequence:
  - primitive_name: move_to_joint_positions
    joint_positions:
      "1": 0.02
      "2": 0.54
      "3": -0.82
      "4": -0.18
      "5": 0.02
    duration_sec: 2.0
```

`workspace_limits` inside a skill implementation are local preconditions only.
They must be validated against the hard workspace and joint bounds from
`robot_config`.

### 8.3 Profile File

The profile selects which skills are enabled for one robot variant:

```yaml
schema_version: 1
name: so101_single_arm
robot_name: so101_single_arm

enabled_skills:
  - name: inspect_scene
    implementation: so101_5dof_single_arm
    planner_visible: true
  - name: wave_hello
    implementation: so101_5dof_single_arm
    planner_visible: true
```

The profile does not own joint limits, controllers, camera topics, or pose data.

The implementation identifier describes a stable kinematic or execution variant,
not a deployment config name. For example, `so101_single_arm` and
`so101_rtp_distributed` should be able to share `so101_5dof_single_arm` instead of
duplicating identical motion YAML.

The profile is authoritative for the enabled and planner-visible skill sets.
`planner_visible` defaults to `true`; it may be set to `false` for skills that are
executable through direct APIs but should not be emitted by planners.

## 9. Compiler and Validation

### 9.1 Compiler Inputs

The compiler takes only plain data:

```python
@dataclass(frozen=True)
class SkillRobotContext:
    robot_name: str
    named_pose_names: frozenset[str]
    arm_joint_names: tuple[str, ...]
    joint_limits: Mapping[str, Mapping[str, float]]
    workspace_limits: Mapping[str, tuple[float, float]]
    required_control_mode: str
    timeout_policy: Mapping[str, float]
    allowed_skill_executors: frozenset[str]
```

### 9.2 Compile Order

1. Load profile.
2. Load catalog files from an immutable source root.
3. Validate manifest and implementation schemas.
4. Check profile references and duplicates.
5. Expand trajectory templates.
6. Validate pose, joint, executor, and primitive references against `SkillRobotContext`.
7. Build capability view.
8. Canonicalize JSON and compute digests.
9. Freeze the result into an immutable snapshot.

### 9.3 Source Read Rule

Source reads must be transactional:

- Production reads from an immutable release directory.
- Development reads from a staging root and verifies the source fingerprint before commit.
- If the tree mutates during a read, the compile must fail or retry, never mix files from two revisions.

### 9.4 Authority Rule

`robot_config.loader` constructs `SkillRobotContext` and resolves the selected
profile name, but it does not own runtime generation and does not perform a second
authoritative compile.

The runtime owner compiles the authoritative bundle once. Offline CLI validation
may call the same compiler, but offline results have no authoritative epoch or
generation.

## 10. Runtime Model

### 10.1 Registry vs Coordinator

`SkillRegistry` is pure catalog state. It does not own:

- active execution lease
- in-flight admission ledger
- process-wide busy state
- executor cancel cleanup

Those belong to `ExecutionCoordinator` in `skill_library`.

This split matters because reload must not let a new policy accidentally ignore
an existing root execution lease.

### 10.2 Runtime Bundle

The runtime bundle contains immutable data only:

```python
@dataclass(frozen=True)
class SkillRuntimeBundle:
    registry_epoch: str
    generation: int
    snapshot: SkillSnapshot
    capability_view: Mapping[str, Any]
    skill_requirements: Mapping[str, Any]
    parameter_schemas: Mapping[str, Any]
```

The bundle does not contain the active lease or request ledger.

### 10.3 Reload Transaction

Reload is a staged swap:

1. Acquire reload mutex.
2. Build a candidate snapshot from the current catalog source.
3. Validate the candidate completely.
4. Build candidate runtime bundle inputs.
5. Compare candidate digest with current digest.
6. If unchanged, return no-op.
7. If changed, atomically swap the current bundle pointer.
8. Increment generation inside the current epoch.
9. Publish a successful event only after the swap is visible.

Old bundles remain valid for active goals until those goals finish.

### 10.4 Restart Semantics

On restart, the runtime owner creates a new epoch and a new coordinator.
The system does not assume it can reconstruct in-flight state from memory.

Operationally:

- new motion is rejected until the fresh bundle is loaded
- stale consumers must resync to the new epoch
- crash recovery is a separate feature, not an implied behavior

## 11. ROS Interfaces

### 11.1 Reload Service

`std_srvs/Trigger` is not enough. Reload needs structured output, so the design
uses a custom service.

Production deployment must disable the service by default or restrict it to an
operator policy. The service always reloads the configured source root and must
not accept an arbitrary path from the request.

Suggested service:

```text
# ReloadSkillCatalog.srv
string request_id
bool force
---
bool success
string registry_epoch
uint64 generation
string registry_digest
string capability_digest
string error_code
string message
string[] changed_skills
```

### 11.2 Snapshot Service

Snapshot requests must support exact version lookup.

```text
# GetSkillSnapshot.srv
string registry_epoch
uint64 generation
---
bool success
string registry_epoch
uint64 generation
string registry_digest
string capability_digest
string snapshot_json
string message
```

`generation == 0` means current bundle. Any other value means exact match.

### 11.3 Registry Event

```text
# SkillRegistryEvent.msg
string registry_epoch
uint64 old_generation
uint64 new_generation
string registry_digest
string capability_digest
string[] changed_skills
```

The event is success-only and informs late subscribers which epoch they should
query.

### 11.4 Status and Validation

`GetSkillGatewayStatus.srv` should expose:

- `registry_epoch`
- `registry_generation`
- `registry_digest`
- `config_digest` / `capability_digest`

`ValidateSkill.srv` and `SkillCommand.action` should carry the expected registry
identity so preflight and admission are tied to the same version.

Suggested request fields:

- `expected_registry_epoch`
- `expected_registry_generation`
- `expected_registry_digest`

If the admission sees a different identity, it fails closed with a structured
version mismatch code.

`TaskCommand.msg` should carry the same expected identity so plans queued between
planner and executor cannot silently switch to a newer catalog. A temporary
compatibility path may place it in `context_json`, but the target interface should
use typed fields.

## 12. Consumer Behavior

### 12.1 Safety Guard

safety_guard must keep a local snapshot and reject validation when the incoming
request version does not match its own snapshot.

### 12.2 Planners and CLI

Planners and CLI should:

1. read the current gateway status
2. validate the public capability digest
3. attach the version identity to the final request
4. reject execution if the version changed between preflight and admission

All planner paths must also filter output through the snapshot enabled set. This
includes the hardcoded observation, recovery, gripper, and relative-motion branches
in the rule parser, not only dynamically loaded aliases.

### 12.3 Active Tasks

An active task keeps the bundle captured at admission time.
It must finalize against the same bundle and the same coordinator lease token.

The runtime may reject new goals during reload, but it must not switch an active
goal to a new bundle mid-flight.

## 13. Packaging and Install Layout

Production loading must not rely on mutable source-tree paths.

Recommended layout:

```text
<install>/share/skill_catalog/releases/<catalog_digest>/...
<install>/share/skill_catalog/current -> releases/<catalog_digest>
```

Rules:

1. Source and release layouts are immutable once activated.
2. The active pointer changes atomically.
3. `setup.py` must recursively install YAML, JSON schema, and `SKILL.md`.
4. A non-source install must be able to load the same catalog content.

Adding or deleting a file in the source tree is not automatically a production
hot reload. Production sees the change only after a complete immutable release is
deployed and the active pointer is atomically switched.

## 14. Implementation Impact

| Area | Required change |
|---|---|
| `src/skill_catalog/**` | Add compiler, schemas, sources, snapshot models, and tests. |
| `src/robot_config/robot_config/loader.py` | Build robot context and enforce legacy/profile mutual exclusion; do not own runtime generation. |
| `src/robot_config/robot_config/config.py` | Add typed profile and implementation-variant fields. |
| `so101_single_arm.yaml` | Remove inline templates after migration and select a profile. |
| `so101_handeye_realsense_grasp.yaml` | Migrate its delegated grasp skill profile. |
| `so101_rtp_distributed.yaml` | Select the shared SO101 kinematic implementation instead of duplicating templates. |
| `embodied_bringup` | Stop treating startup template JSON as the profile-mode authority. |
| `skill_library` | Add bundle swap, shared `ExecutionCoordinator`, exact version admission, and restart handling. |
| `safety_guard` | Add exact snapshot sync and epoch/generation validation. |
| `embodied_common.command_parser` | Gate hardcoded and alias-based outputs by the snapshot enabled set. |
| `embodied_agent` and `vlm_task_planner` | Consume dynamic enabled/planner-visible skills and carry request version identity. |
| `robot_skill_cli` | Bind preflight, validation, and execution to one snapshot identity. |
| `ibrobot_msgs` | Add reload/snapshot/event interfaces and version fields on task, validation, status, and action contracts. |
| Package README and tests | Update SSOT, interfaces, install layout, and concurrency expectations. |

Architecture governance files should be updated only if they exist in the target
baseline. The current `IB_Robot_0803` workspace does not contain the referenced
`.agents/architecture/` rule tree, so migration must not assume that gate is
available.

## 15. Migration Plan

### Stage 0: Governance

- update ADR and architecture rules
- freeze schema v1
- agree on epoch/generation/digest semantics

### Stage 1: Pure Compiler

- add `skill_catalog` package
- move validation logic out of runtime nodes
- keep current runtime behavior unchanged

### Stage 2: Catalog Files

- move skills into catalog directories
- keep current skill set and outputs byte-for-byte equivalent
- preserve legacy inline path only as temporary compatibility

### Stage 3: Versioned Runtime

- add exact snapshot service
- add registry event
- add version-bound requests
- switch `skill_library` to bundle swap

### Stage 4: Remove Legacy Inline Skill SSOT

- remove inline `embodied.skill_templates`
- remove unversioned request paths
- remove startup-only static JSON as the authoritative source

## 16. Test Plan

### Compiler Tests

- manifest field type mismatches
- profile references to missing skills
- duplicate skill names and aliases
- unsupported primitive or executor
- invalid joint or pose references
- exact digest stability
- schema examples matching validator types
- profile-enabled and planner-visible set derivation
- stable implementation reuse across deployment config names

### Runtime Tests

- reload no-op when digest is unchanged
- reload success increments generation in the same epoch
- restart creates a new epoch
- stale epoch is rejected
- active goal survives reload with its captured bundle
- new goal cannot mix old preflight with new admission

### Concurrency Tests

- concurrent reload calls serialize
- reload and active goal do not deadlock
- coordinator lease prevents double root execution
- snapshot service returns exact version or a structured miss
- rule parser cannot emit a skill removed from the current profile

### Packaging Tests

- install space can load catalog files
- source-tree mutation during compile is rejected or retried
- release pointer swap is atomic

## 17. Risks

| Risk | Mitigation |
|---|---|
| epoch restart confusion | carry epoch on every event and request |
| stale consumer split-brain | exact snapshot lookup and fail closed |
| reload interfering with active task | coordinator separated from registry |
| partial file updates | staged immutable release root |
| schema drift | manifest examples must match current validators |

## 18. Open Questions

1. Should the production reload service be operator-only or disabled by default?
2. Should history retention keep only active generations or a small bounded cache?
3. Should the catalog source be `skill_catalog` or a differently named package?

The rest of the design assumes the answers do not change the version model,
coordinator split, or exact snapshot protocol.
