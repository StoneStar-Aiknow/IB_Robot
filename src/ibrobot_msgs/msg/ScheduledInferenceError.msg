# AI-AGENT NOTICE: Scheduled-only interface. Use only when
# control_modes.<mode>.inference.scheduler.enable=true. Do not use on the legacy inference path.
# Bounded scheduled-path error. Replaces the legacy unbounded InferenceError on scheduled IDL.
string<=64 code
string<=1024 message
bool recoverable
string<=64 stage
string<=8192 details_json
