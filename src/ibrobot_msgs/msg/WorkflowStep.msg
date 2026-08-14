# WorkflowStep.msg - One typed step of a planned Workflow or Agent plan.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.5.
#
# skill_name is a compatibility field name; it may reference a catalog entry
# with semantic_level=atomic_operator or skill. A step does not carry its own
# registry identity: it inherits the exact snapshot of the enclosing planned
# TaskCommand.dispatch_binding. timeout_sec <= 0 uses the entry default;
# positive values are still bounded by the Skill cap and shared TaskBudget.
# schema_version is 1 in v1.

uint32 schema_version
string skill_name
string target_name
string container_name
string place_name
string motion_direction
float32 motion_distance
float32 timeout_sec
