# A container detected on a pick's capture, handed to the following place with
# the held object (ClaimHeldObject). The place reuses the mask only when its own
# capture shows the same camera and an unchanged container region; otherwise it
# detects again. Empty query: nothing was prefetched.
string query
int64 stamp_ns
# Row-major 4x4 base_from_camera and 3x3 intrinsics of the capture.
float64[] base_from_camera
float64[] intrinsics
# PNG mask over the whole image (manipulation_execution pick prefetch).
sensor_msgs/CompressedImage mask
# The container region of the capture: [x0, y0, x1, y1) pixel box, its RGB (rgb8)
# and depth in metres (32FC1), compared with the place's own capture.
int32[] roi
sensor_msgs/Image rgb
sensor_msgs/Image depth
