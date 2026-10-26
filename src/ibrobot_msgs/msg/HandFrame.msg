# Per-frame hand keypoints of the actor, one message at most per camera frame.
#
# header.stamp carries the source camera frame's observation time, the same
# clock and value as PearFrame, so the retargeter merges hands into the arm's
# window by it. Same layout as video2x2_runtime hri_pear_msgs/HandFrame (570c53c).

std_msgs/Header header
# Monotonic per node run, from 0.
int64 frame_id
# header.stamp flattened.
int64 timestamp_ns

# The person's left and right hands.
HandObservation left
HandObservation right

# The person box used: the newest PearFrame not later than this frame.
# person_valid false: no usable box (none yet, too old, or a fallback crop);
# the frame is not inferred and both hands are undetected.
bool person_valid
float64[4] person_bbox_xyxy
float32 person_bbox_age_ms
int32 hands_in_frame

# Diagnostics, not on the data path.
float32 decode_ms
float32 infer_ms
float32 latency_ms
