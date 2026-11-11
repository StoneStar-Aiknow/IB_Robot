"""The ``robodojo`` simulation platform spec (``ibrobot.sim_platforms`` entry point).

Import-light by contract (sim-platform-registry D1): building an interface description
imports this module, so it uses only the registry's types. Everything heavy is a lazy
``module:attr`` reference.

RoboDojo's Isaac Sim process owns the world, the scene and the task, so there is no world
hook (``world=None``): generic launch starts nothing world-side. The robot has no controller
on IB-Robot's side; the embodiment starts the direct-control runtime with this platform's
adapter. Default sensor traits: the profile declares the cameras exactly as published.
"""

from __future__ import annotations

from robot_runtime.sim_platforms import SimPlatform, SimSensorTraits

ROBODOJO_SPEC = SimPlatform(
    name="robodojo",
    package="robodojo_robot",
    embodiment="robodojo_robot.embodiment:ROBODOJO_EMBODIMENT",
    world=None,
    scene_task=None,
    sensor_traits=SimSensorTraits(),
)
