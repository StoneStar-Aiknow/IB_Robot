# Custody of what one arm holds: the single state source the Gateway admits
# motion from. Published latched by the executor that owns custody and
# persisted before every publish. State names, owners, reason codes and who may
# act in each state: embodied_common.held_custody_contract.
# Ownership changes only through ClaimHeldObject / ReturnHeldObject; an
# interrupted pick is reported here too (no separate record).
std_msgs/Header header
uint32 schema_version
# Never empty; a fresh one for a custody the owner cannot match (e.g. after a restart).
string custody_id
# Generated when the custody owner process starts. Requests and claim tokens
# from another epoch are refused.
string custody_epoch
# EMPTY | HELD | RECOVERING | UNKNOWN. Only fresh empty-hand feedback makes EMPTY
# (reason DROPPED: it fell, the Gateway returns home first). HELD is safe: the arm
# admits only body motions and a place of this object, from any root task, and
# the Gateway puts it back only after idling (NO_PUT_BACK: refused, stays held).
# UNKNOWN stops this arm only and admits on it only the operator's open-hand and
# return-home commands. The owner republishes every CUSTODY_REPUBLISH_S; the
# Gateway reads an older message as UNKNOWN.
string state
# pick | place while HELD; the last known one, or empty, while UNKNOWN.
string owner
# Custody is per arm: one message per arm side.
string arm_side
string target_query
# Allowed per state: embodied_common.held_custody_contract.STATE_REASONS.
string reason_code
# True while a goal of the owner's own (a pick, a put-back) runs on this arm:
# the arm admits nothing else, and a Gateway that starts waits for it to end.
bool goal_active
# Held geometry: points in the palm frame (xyz triples) are primary; radius_m
# is derived from them for consumers that need a bound.
float64[] points_in_palm
float64 radius_m
string effector_state_json
float64 lift_m
string[] joint_names
float64[] joint_positions
float64[] home_positions
# Arm joints at the verified grasp contact; empty when unknown (no put-back).
float64[] contact_positions
string observation_id
builtin_interfaces/Time lifted_at
# Handover pose back to home_positions as pick drove it (reversed), row-major;
# empty when unknown. Never replayed without a check on the current scene.
float64[] return_positions
float64[] return_times
