std_msgs/Header header

uint8 ACQUIRING=0
uint8 TRACKING=1
uint8 SEARCHING=2
uint8 LOST=3
uint8 STOPPED=4

uint8 MOTION_UNKNOWN=0
uint8 STATIONARY=1
uint8 MOVING=2

string session_id
string object_id
uint8 lifecycle_state
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
float32[4] bbox
bool measured
bool prediction_only
bool actionable
uint8 motion_state
float32 ego_compensated_speed
float32 confidence
builtin_interfaces/Time last_seen
string state_reason
