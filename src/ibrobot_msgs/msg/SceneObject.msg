string object_id
string label
float32 confidence

# 2D bounding box in image pixel coordinates.
uint32 bbox_x
uint32 bbox_y
uint32 bbox_width
uint32 bbox_height

# Optional 3D pose. An empty header.frame_id means unavailable.
geometry_msgs/PoseStamped pose

string source
string attributes_json
