# Command-side reference of a managed arm, published every control tick
# (motion.arm.reference).
#
# For monitoring, lost-tracking feedback and recording. Clients take the
# clutch origin from the StartArmSession response, not from this topic.
# Session state (active mode, stop latch) is in RuntimeStatus.
# Not published during a joint-stream session, which has no Cartesian reference.

# stamp: control tick time; frame_id: base frame.
std_msgs/Header header
# End-effector target the runtime is currently tracking.
geometry_msgs/Pose reference
# m; distance between the reference and the measured end-effector position.
float64 position_error_m
# rad; NaN when orientation is not tracked (position_only).
float64 orientation_error_rad
# False when the commanded pose has stopped short of the reference: the target
# is outside the workspace, or the joint limits block the arm's current branch.
# Tracking lag while the command still closes the gap does not clear it.
bool tracking
# Why the last target was rejected or is not tracked; empty otherwise.
string reason
