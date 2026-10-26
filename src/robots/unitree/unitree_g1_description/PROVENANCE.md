# Provenance

Vendored from Unitree Robotics, BSD 3-Clause License (see LICENSE).

- Upstream repository: https://github.com/unitreerobotics/unitree_ros
- Upstream commit: ccfc6fd8430a17ba3dacef9a1e2faf64ff3b0aee ("rename for r1-7a")
- Upstream path: robots/g1_description
- Downloaded: 2026-09-17 (local clone under `/unitree/`, gitignored)

## Vendored subset

- `urdf/g1_29dof_with_hand_rev_1_0.urdf` — the 29-DoF + Dex3-hand model.
  Per the upstream README this corresponds to `mode_machine = 5` (wrist motor
  4010); the real machine's mode must be confirmed before the runtime declares
  this configuration (openspec unitree-g1-runtime-migration design D10).
- `meshes/` — exactly the 49 STL files the URDF references; nothing else from
  upstream `meshes/` is copied.

## Local modifications (IB_Robot)

- Mesh paths rewritten from the upstream relative `meshes/<file>.STL` to
  `package://unitree_g1_description/meshes/<file>.STL` (87 references) so
  robot_state_publisher / RViz resolve them from the installed package. The
  URDF is otherwise byte-identical to upstream.
- Packaging (`package.xml`, `CMakeLists.txt`) written for this repository; no
  upstream launch/config files are vendored.

Upstream updates are an explicit re-vendor operation recorded in this file.
