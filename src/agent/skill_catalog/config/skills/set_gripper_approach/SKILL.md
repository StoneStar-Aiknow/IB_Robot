# set_gripper_approach

Turn the gripper to one of the robot's named approach orientations (for example pointing
straight down) while the grasp point stays where it is. The orientations are declared by the
robot configuration (`embodied.execution.gripper_approach.presets`), not by this skill.

The name travels in the request's `approach` field; safety_guard rejects names the robot does
not declare and any target whose orientation differs from the declared one.
