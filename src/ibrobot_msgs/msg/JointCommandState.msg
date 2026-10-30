# What a robot's joint controller is commanding now, as the controller reports it.
#
# Measured joint states say where the joints are; this says where the controller
# is driving them and how stiffly. A position-controlled arm settles away from its
# target under load (by torque / stiffness), so the two differ, and an executor
# that takes over the stream must start from this target, not from the measured
# position, or the arm moves at takeover. It is also how an executor sees the
# controller clamp or replace the targets it sends.
#
# One entry per joint; the arrays are parallel to name. stamp is the
# controller's own time for this command where it reports one.

std_msgs/Header header
string[] name
# rad
float64[] position
# The controller's position gain and damping for the joint, in its own units.
float64[] stiffness
float64[] damping
