# TaskCommand.msg - Main task data carrier across the embodied AI task entry,
# planner and execution path.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.5.
#
# dispatch_binding carries the full task/version envelope (task_id moved here).
# Before planning, raw TaskCommands may have empty identity and empty
# workflow_steps; planned TaskCommands must carry non-empty typed steps,
# complete version binding and a correct workflow_digest, while
# root_lease_nonce and dispatch_nonce remain empty. Task executor must use
# each WorkflowStep's own parameters and must not override all steps with the
# top-level target_name/place_name/motion_* hints. context_json only carries
# non-execution planner/perception annotations: it must not carry registry
# identity, task budget, skill_sequence or other strongly typed contract
# fields.

DispatchBinding dispatch_binding
string source
string raw_command
string task_type
WorkflowStep[] workflow_steps
string target_name
string place_name
string motion_direction
float32 motion_distance
uint8 priority
float32 timeout_sec
string context_json
