# TaskStatus.msg - Task execution status, published by task_entry_node /
# task_executor_node to /embodied/task_status.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.5.
# schema_version is 1 in v1. Action result, validation response and status
# response return the actually-used or observed registry identity.

uint32 schema_version
string task_id
string state
bool success
string current_skill
string[] completed_skills
string error_code
string message
bool recoverable
bool replan_requested
string actual_registry_epoch
uint64 actual_registry_generation
string actual_registry_digest
string provenance_digest
