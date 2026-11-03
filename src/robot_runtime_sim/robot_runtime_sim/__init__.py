"""Simulation platform providers for the robot runtime (sim-platform-registry D6).

The Gazebo and MuJoCo platform specs (``platforms.py``) register through the
``ibrobot.sim_platforms`` entry points; their default embodiments
(``gazebo_embodiment.py`` / ``mujoco_embodiment.py``) hold the description
renderers and launch compositions moved out of ``robot_runtime.launch_support``.
The runtime contract layer discovers both only through the registry.
"""
