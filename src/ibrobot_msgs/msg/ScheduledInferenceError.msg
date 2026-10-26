# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# Hardware policy inference with robot.runtime_scheduler.enable=true only.
# Simulation and benchmark use the direct DispatchInfer interface.
# Bounded scheduled-path error. Replaces the legacy unbounded InferenceError on scheduled IDL.
string<=64 code
string<=1024 message
bool recoverable
string<=64 stage
string<=8192 details_json
