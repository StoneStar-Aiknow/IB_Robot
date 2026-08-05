# SkillDiagnostic.msg - Structured diagnostic emitted by all catalog compile
# entry points (compiler, reload, validate, agent plan/validate/confirm).
#
# See docs/lightweight_skill_package_registry_design_zh.md section 11.1.
#
# schema_version is 1 in v1; source_relative_path is relative to the release
# root and empty when there is no associated file or field_path. Diagnostics
# are sorted by source_relative_path, error_code, field_path, message.

uint8 ERROR=1
uint8 WARNING=2

uint32 schema_version
uint8 severity
string error_code
string source_relative_path
string field_path
string message
