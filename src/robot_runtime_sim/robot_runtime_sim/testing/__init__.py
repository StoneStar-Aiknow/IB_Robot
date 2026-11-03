"""Test-support API of the simulation platform providers (openspec D14.1/D6).

``sim_environment`` and ``sim_ground_truth`` moved here from
``robot_runtime.testing`` together with the platforms they probe: they name
simulator packages and the MuJoCo platform module, so a hardware-only
``robot_runtime`` deployment never needs them. Everything under
``robot_runtime_sim.testing`` exists so repository tests can observe a
running simulation the same way; none of it is part of any runtime launch
path. Importing a module here must start no process and create no ROS
entity (D4: nothing ground-truth on the graph).
"""
