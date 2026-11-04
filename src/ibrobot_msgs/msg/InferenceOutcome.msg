# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# Hardware policy inference with robot.runtime_scheduler.enable=true only.
# Simulation and benchmark use the direct DispatchInfer interface.
# Terminal certainty for a scheduled inference request.
# UNSPECIFIED is the ROS default only; any terminal Result using it is a protocol error.
uint8 UNSPECIFIED=0
# No session generation, processor, backend or action side effect; reservation rolled back.
uint8 NOT_STARTED=1
# Definite terminal reached; does not imply business success.
uint8 COMPLETED=2
# Downstream terminal certainty could not be determined (lost acceptance/result).
uint8 UNKNOWN=3
uint8 value
