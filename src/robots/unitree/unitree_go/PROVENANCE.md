# Provenance

Vendored from Unitree Robotics, BSD 3-Clause License (see LICENSE).

- Upstream repository: https://github.com/unitreerobotics/unitree_ros2
- Upstream commit: 668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
- Upstream path: cyclonedds_ws/src/unitree/unitree_go
- Downloaded: 2026-09-17 (local clone under `/unitree/`, gitignored)

## Local modifications (IB_Robot, ROS 2 Humble adaptation)

- `CMakeLists.txt`: removed `rosidl_generator_dds_idl` and the
  `rosidl_generate_dds_interfaces` / `add_dependencies` Connext IDL block —
  that generator was removed after ROS 2 Galactic. `rosidl_default_generators`
  (already a dependency) provides the Python/C/C++ typesupports used here.
- `CMakeLists.txt`: disabled `ament_cmake_copyright` / `ament_cmake_cpplint`
  linters — upstream source files carry no copyright headers and must stay
  byte-identical to upstream.
- `package.xml`: removed the empty `<?xml-model href=""?>` declaration and
  replaced the TODO description.

Upstream updates are an explicit re-vendor operation recorded in this file.
