std_msgs/Header header

string label
float32 confidence

# Bounding box in pixel coordinates [x_min, y_min, x_max, y_max]
float32[4] bbox

# Binary segmentation mask (mono8, same size as input image)
sensor_msgs/Image mask

# 3D centroid in camera optical frame (meters)
geometry_msgs/Point centroid_xyz

# Number of valid points in the masked region
int32 point_count
