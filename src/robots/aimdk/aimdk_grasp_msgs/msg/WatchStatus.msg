# The X2 grasp head's target watch, published while a session is armed
# (aimdk_grasp.grasp_head -> aimdk_grasp.remote_head), a few times a second
# and at once when the target moved.
#
# Every field is the head's own: times are measured on the head's clock and
# never compared with the receiver's. The receiver judges freshness from when
# it received each status and from fresh_frames advancing.

# The session the robot-side pipeline named when it armed the watch; a status
# of any other session (an earlier goal, a replay) is not about this one.
string session_id
# 1, 2, ... within the session, in publishing order.
uint64 sequence

# Armed; no watched step (the approach) has started yet.
uint8 WAITING=0
# The newest fresh frame was compared and the target reads where it was planned.
uint8 STABLE=1
# The frames voted that the target moved (latched for the session).
uint8 MOVED=2
# Frames arrive but cannot decide (the hand covers the target, no camera
# shift could be measured, image-time feedback unhealthy): see unknown_s.
uint8 UNKNOWN=3
# The watch cannot look: it did not start, its thread stopped, or no fresh
# image for longer than the head's frame timeout. See reason.
uint8 UNAVAILABLE=4
# The watched steps are over (the close started): the hand is meant to touch
# the target now, so nothing is compared any more.
uint8 ENDED=5
uint8 state

# The execution step the head last heard of.
string phase
# Frames compared or found undecidable so far: each one is a newer image than
# the one before, captured while a watched step ran.
uint32 fresh_frames
# The newest of those images (the head's camera clock); 0 before the first.
int64 last_image_stamp_ns
# UNKNOWN: how long no frame has decided, on the head's clock.
float64 unknown_s
string reason
