# Skill-independent facts. schema_version=0 means unavailable/legacy.
# These fields describe execution; suggested_action never authorizes motion.
uint32 schema_version
string phase
string phase_state
string failed_phase
string reason_code
string recovery_state
uint32 retry_count
string suggested_action
