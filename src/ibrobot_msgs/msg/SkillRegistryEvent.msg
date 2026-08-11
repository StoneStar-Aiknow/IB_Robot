# SkillRegistryEvent.msg - Published only after a successful reload to tell
# late-joining subscribers which epoch/generation to query.
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.4.
#
# QoS is fixed: RELIABLE, TRANSIENT_LOCAL, KEEP_LAST, depth 1. changed_skills
# compares full skill package manifest, selected implementation, enabled state
# and planner-visible state; it may be empty when only profile, robot context
# or source documentation (e.g. SKILL.md) changed.
# schema_version is 1 in v1.

uint32 schema_version
string registry_epoch
uint64 old_generation
uint64 new_generation
string registry_digest
string capability_digest
string source_release_digest
string provenance_digest
string profile_name
string[] changed_skills
