# TaskBudget.msg - Immutable task deadline envelope shared by every dispatch
# inside one root Task/Workflow. Owned by the task entry boundary; planners,
# task executor, SkillCommand, delegated executor and primitive dispatch must
# propagate it verbatim.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 10.7 and 11.5.
# schema_version is 1 in v1.

uint32 schema_version
builtin_interfaces/Time started_at
builtin_interfaces/Time deadline
