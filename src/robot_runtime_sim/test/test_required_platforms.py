"""The first-party platforms register and stay import-light (D10).

Two guarantees the open registry rests on:

- ``gazebo`` and ``mujoco`` are discoverable through the REAL
  ``ibrobot.sim_platforms`` entry points (no injection seam), with the specs
  this package ships;
- importing ``robot_runtime_sim.platforms`` stays light: building an
  interface description imports a spec module, so it must never pull in
  ``launch``, ``sim_models`` or the ``mujoco`` Python package. Checked in a
  subprocess so the importing interpreter is pristine (this test process
  already imported most of them).
"""

from __future__ import annotations

import subprocess
import sys

from robot_runtime_sim.platforms import GAZEBO_SPEC, MUJOCO_SPEC


def test_gazebo_and_mujoco_are_registered_through_the_real_entry_points():
    from robot_runtime.sim_platforms import clear_sim_platform_cache, discover_sim_platforms

    clear_sim_platform_cache()
    try:
        discovered = discover_sim_platforms()  # real importlib.metadata discovery
    finally:
        clear_sim_platform_cache()
    assert discovered.get("gazebo") is GAZEBO_SPEC, f"gazebo not discovered: {sorted(discovered)}"
    assert discovered.get("mujoco") is MUJOCO_SPEC, f"mujoco not discovered: {sorted(discovered)}"


def test_the_first_party_driver_sets_match_the_platform_modules():
    import robot_runtime_sim.gazebo as gazebo
    import robot_runtime_sim.mujoco as mujoco

    assert GAZEBO_SPEC.sensor_traits.lidar_sim_drivers == gazebo.LIDAR_SIM_DRIVERS
    assert GAZEBO_SPEC.sensor_traits.integrated_imu_drivers == gazebo.LIDAR_INTEGRATED_IMU_DRIVERS
    assert MUJOCO_SPEC.sensor_traits.lidar_sim_drivers == mujoco.MUJOCO_LIDAR_SIM_DRIVERS
    assert MUJOCO_SPEC.sensor_traits.integrated_imu_drivers == gazebo.LIDAR_INTEGRATED_IMU_DRIVERS


def test_importing_the_spec_module_pulls_in_nothing_heavy():
    code = (
        "import sys\n"
        "import robot_runtime_sim.platforms\n"
        "heavy = [name for name in ('launch', 'sim_models', 'mujoco') if name in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    heavy = result.stdout.strip()
    assert not heavy, f"robot_runtime_sim.platforms imported {heavy}"
