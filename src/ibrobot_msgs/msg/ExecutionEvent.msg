# Authoritative asynchronous projection of one Gateway-owned execution.
# Events are ordered per execution_id by sequence.  canonical_steps describes
# the frozen public Program; it never contains private catalog templates.

uint32 schema_version
string execution_id
string request_id
string caller_scope
string robot_id
uint64 sequence
string state
string program_digest
WorkflowStep[] canonical_steps
uint32 completed_step_count
string error_code
string message
builtin_interfaces/Time created_at
