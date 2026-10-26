# One step of a grasp plan (manipulation_execution grasp_pipeline.contracts.Step),
# as the X2 grasp head returns it to the X2 grasp pipeline.

uint8 MOVE=0
uint8 EFFECT=1
uint8 HOLD=2
uint8 kind
string name

# MOVE: an arm segment. A TRAJECTORY has N >= 2 waypoints with times from
# zero; a TARGET has two rows (the planned start and the target) and no times.
uint8 TRAJECTORY=0
uint8 TARGET=1
uint8 segment_kind
string[] joint_names
# N x len(joint_names), row-major
float64[] positions
float64[] times
# Ask the planner to revise the rest of the plan once this move has run.
bool refine_after

# EFFECT: "preshape", "close" or "release"
string action

# HOLD
float64 seconds
