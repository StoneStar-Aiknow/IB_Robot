# free_pose

Move the arms to a pose the user describes in their own words.

Runs through the `pose_composer` executor: `target_name` carries the user's
description unchanged. The pose composer has a language model describe each
arm (segment directions, or the hand at a named body point), solves the joint
angles on the official X2 model inside the runtime's joint travel and clear of
the body, plans a collision-free path from the measured pose, and executes it
through the runtime's `joint.trajectory` action. With the Agent, the pose is
composed while the plan is prepared, so the presented pose is the one that
runs. A stop request holds the arms where they are.
