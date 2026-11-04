# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# Hardware policy inference with robot.runtime_scheduler.enable=true only.
# Simulation and benchmark use the direct DispatchInfer interface.
# Per work-class public capacity of a pipeline, echoed from robot_config + live counts.
uint8 SESSION_CONTROL=1
uint8 ACTION_GENERATION=2
uint8 work_class
uint32 max_in_flight
uint32 current_in_flight
bool accepting_requests
