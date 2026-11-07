# Durable snapshot; ownership is changed only by ClaimHeldObject/ReturnHeldObject.
std_msgs/Header header
string custody_id
bool holding
string state
string arm_side
string target_query
string owner
# Root task of the pick that took custody; only that task's place may claim it.
string root_task_id
string reason_code
float64 radius_m
float64[] points_in_palm
string effector_state_json
float64 lift_m
string[] joint_names
float64[] joint_positions
float64[] home_positions
string observation_id
builtin_interfaces/Time lifted_at
# Handover pose back to home_positions as pick drove it (reversed), row-major; empty if unknown.
float64[] return_positions
float64[] return_times
