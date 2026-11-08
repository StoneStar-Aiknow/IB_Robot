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
# Version 2 physical facts; empty/default strings normalize to UNKNOWN only
# through version-aware consumers. v1 never asserts placement facts.
# release_status: UNKNOWN | NOT_RELEASED | RELEASED
string release_status
# verification_status: UNKNOWN | NOT_RUN | SUCCESS | FAILED | UNCERTAIN
string verification_status
bool place_succeeded
