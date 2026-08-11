# SkillCapabilityStatus.msg - Gateway non-blocking readiness snapshot for one
# public high-level capability (Atomic Operator or Skill).
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.5.
# schema_version is 1 in v1. semantic_level is read by the Gateway from the
# captured snapshot; callers must not self-report the level.

uint32 schema_version
string name
string semantic_level
bool planner_visible
bool ready
string reason
string required_control_mode
