std_msgs/Header header

string label
float32 confidence

# Bounding box in pixel coordinates [x_min, y_min, x_max, y_max]
float32[4] bbox

# Binary segmentation mask (mono8, same size as input image)
sensor_msgs/Image mask

# 3D centroid in camera optical frame (meters)
geometry_msgs/Point centroid_xyz

# Volume centroid of the convex hull enclosing the object's 3D points (meters).
# Approximates the filled-volume center of mass; closer to the physical center
# than centroid_xyz (visible-surface mean) for convex objects.
geometry_msgs/Point volume_centroid_xyz

# Convex hull volume (m^3) of the object's 3D points
float32 volume_m3

# Number of valid points in the masked region
int32 point_count
