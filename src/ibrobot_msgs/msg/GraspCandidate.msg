# Single 6-DOF grasp candidate
std_msgs/Header header

# 4x4 homogeneous transformation matrix (row-major, SE(3))
float32[16] pose_matrix

# Discriminator confidence score [0, 1]
float32 confidence

# Whether this grasp passed collision filtering
bool collision_free

# Robot-agnostic target geometry estimated from the segmented point cloud.
# Width is measured along this candidate's source gripper closing axis.
# A width_quality of 0.0 means no reliable width was estimated.
float32 target_width_m
float32 target_width_quality
float32[3] width_axis_camera

# Robust target extent along width_axis_camera, relative to pose_matrix's
# translation. The min side is the side opposite width_axis_camera. Consumers
# can combine these offsets with target-specific asymmetric gripper geometry.
float32 target_width_min_offset_m
float32 target_width_max_offset_m
