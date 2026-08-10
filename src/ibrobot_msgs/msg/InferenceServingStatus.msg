# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# control_modes.<mode>.inference.scheduler.enable=true. Do not use on the legacy inference path.
# Product-session serving status published by each monolithic scheduled pipeline.
# Distinct from InferencePipelineStatus, which carries distributed transport handshake only.
uint8 STARTING=1
uint8 IDLE=2
uint8 RESETTING=3
uint8 ACTIVE=4
uint8 FAILED=5
uint8 CLOSING=6
std_msgs/Header header
string<=63 pipeline_id
string<=36 boot_id
uint64 sequence
uint8 state
string<=36 product_session_id
uint64 product_session_generation
builtin_interfaces/Time lease_expires_at
string<=64 deployment_fingerprint
string<=64 runtime_policy_fingerprint
string<=64 pipeline_compatibility_fingerprint
string<=128 configured_hardware_resource_id
string<=128 runtime_hardware_resource_id
uint32 hardware_priority_levels
ibrobot_msgs/InferenceWorkCapacity[<=2] capacities
ibrobot_msgs/ScheduledInferenceError error
