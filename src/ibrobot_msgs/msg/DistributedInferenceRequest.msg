uint8 OP_INFER=1
uint8 OP_RESET=2
uint8 OP_CANCEL=3

uint32 protocol_version
uint8 operation
string pipeline_id
string request_id
string target_request_id
string session_id
uint64 session_generation
string deployment_fingerprint
# Absolute UTC/system-clock deadline. Zero means no explicit deadline.
builtin_interfaces/Time deadline
string prompt
VariantsList tensors
