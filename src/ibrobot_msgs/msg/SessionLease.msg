# Keepalive for a managed teleoperation session (motion.arm.session_lease).
#
# The runtime drops a lease stamped at or before the current session start, so
# a lease from an earlier session cannot keep a new one alive. The sender must
# share the runtime's clock domain.

std_msgs/Header header
