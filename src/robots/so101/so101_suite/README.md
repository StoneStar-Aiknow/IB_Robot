# so101_suite

SO-101 robot suite: robot-specific grasp geometry and wrist guards consumed by the generic manipulation pipeline through configuration-declared providers.

## Responsibility

- **Gripper mesh geometry** (`gripper_geometry.py`): SO-101 STL loading, convex hull, jaw motion, tabletop clearance, batch collision metrics
- **Joint-5 wrist guards** (`wrist_guard.py`): branch continuity, closing-axis correction, within-limit checks, retry logic

## Prohibited

- Importing from `manipulation_execution` (avoids a circular dependency; the suite is self-contained)
- Generic grasp math (lives in `manipulation_execution.geometry`)

## Design

Per design D8 and openspec change `cleanup-manipulation-so101-providers`: the
generic manipulation pipeline resolves robot-specific geometry/guards through
configuration-declared providers and carries no robot default. The two
features declare their requirements separately:

- the orientation guard (joint-5 constraints) needs
  `target_gripper.ik_orientation_guard.wrist_guard_provider`;
- the tabletop filter needs `target_geometry.grasp_geometry_provider` plus the
  mesh location fields `mesh_package` / `mesh_directory`.

With a feature enabled and its provider missing, node startup fails fast
naming the configuration field instead of starting a degraded pipeline. The
`so101_suite.*` module names above are this robot's example: any module
satisfying the provider contract works.

The suite is a **dependency leaf**: no dependency of any category on
`manipulation_execution` or any other IB-Robot generic package, so a robot
runtime (`lekiwi_robot` exec_depends on it) installs it without the
manipulation/inference closure, and `colcon list --packages-up-to so101_suite`
lists no other IB-Robot package.

The quaternion semantics preserve the deleted legacy guard: input quaternions
are normalised and a zero quaternion raises `ValueError`.

## Testing

Direct unit tests in `test/` (registered for `colcon test` through
`extras_require={"test": ["pytest"]}` in `setup.py`):
- `test_gripper_geometry.py` — synthetic-STL mesh tests (headless, no repo meshes needed), quaternion normalization, batch/scalar parity, plus real-mesh parity when `so101_description` meshes are present
- `test_wrist_guard.py` — joint-5 branch continuity, closing-axis correction, quaternion normalization and zero-quaternion rejection, retry, HOME limits
- The provider-contract tests live in the composition root,
  `embodied_bringup/test/test_so101_provider_contract.py`: the consumer owns
  its contract tuples, this suite must not test-import the generic package,
  and the generic package must not test-import this suite

The generic pipeline (`manipulation_execution`) declares no dependency of any
category on this package and resolves it at runtime through
configuration-declared providers; the generic core stays buildable, testable
and launchable without any robot suite.
