string object_id
uint64 object_version
string label
string[] query_labels
float32 confidence

# Fused object pose in the persistent semantic-map frame.
geometry_msgs/PoseWithCovarianceStamped pose

# Axis-aligned extent in pose.header.frame_id, in meters.
geometry_msgs/Vector3 size

builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
uint32 observation_count
uint32 point_count
bool active
string state
bool action_ready
string readiness_reason
string map_version
string observation_source_frame
string model_versions_json
string caption
string pointcloud_ref

# Extensible metadata. Image embeddings are intentionally kept private.
string attributes_json
