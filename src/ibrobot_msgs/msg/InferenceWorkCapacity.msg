# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# control_modes.<mode>.inference.scheduler.enable=true. Do not use on the legacy inference path.
# Per work-class public capacity of a pipeline, echoed from robot_config + live counts.
uint8 SESSION_CONTROL=1
uint8 ACTION_GENERATION=2
uint8 work_class
uint32 max_in_flight
uint32 current_in_flight
bool accepting_requests
