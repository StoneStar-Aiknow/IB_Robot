# Per-frame SMPL-X / FLAME parameters regressed by the PEAR parameter network.
#
# One message per accepted PEAR response, published continuously by the
# resident head-camera stream. This is a live stream, not a recording:
# rejected frames (service failure, empty, non-finite or wrong-width output)
# are dropped and counted on the stream's status topic rather than published.
#
# The array field names match the PearParameterPredict response exactly, and
# header.stamp carries the source camera frame's observation time, so a
# retargeting consumer gets the same (stamp, smplx_pose_raw) pair that the
# windowed imitation capture records.

std_msgs/Header header

# Contract tag for consumers that may see more than one producer generation.
string schema

# Monotonic per node run, counting published frames (not frames received).
uint64 sequence

# "yolox" when the crop came from a person detection, "center_fallback" when
# there is no actor (no qualifying person, no detection result yet, or the
# newest result is stale) and PEAR was handed a centred square crop instead.
# A consumer cannot tell the two apart from the pose alone.
string source

# The actor this crop belongs to, as chosen for the same frame and published
# on the stream's ibrobot_msgs/ActorState topic; -1 for a center_fallback
# frame, which has no actor.
int32 actor_id

# Person box in source-image pixels, [x_min, y_min, x_max, y_max].
float32[4] bbox

# Detection confidence; exactly 0.0 for a center_fallback frame.
float32 confidence

# Inference latency as reported by the PEAR service.
float32 inference_time_ms

# 52 packed 6D rotations: 0 root, 1-21 body, 22-36 left hand, 37-51 right hand.
float32[] smplx_pose_raw
float32[] smplx_scale
float32[] smplx_shape
float32[] smplx_expression
float32[] flame_pose
float32[] flame_shape
float32[] flame_expression
float32[] camera_raw
