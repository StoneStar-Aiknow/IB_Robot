# mirror_stop

Stop mirroring and hand the robot back to idle.

Calls the X2 mirror session's `~/set_session` with `false` through the
`mirror_session` executor. The session winds the current round down and
releases the runtime to idle. If the runtime is already idle there is nothing
to stop: the Gateway answers success without switching modes or calling the
session.

## Scope in this version

Mirroring is a resident state started by `mirror_start`, not a running task.
Nothing interrupts it from inside:

- Generic stop phrases ("停", "停下", "stop") and Agent cancel find no active
  task and leave the mirror running.
- `mirror_stop` is the only way to end mirroring. Say it explicitly, e.g.
  "停止照镜子".

## Emergency stop

The runtime stop service (`/runtime/stop`) is the operator emergency stop. It
latches the runtime; the session closes its output and drops to WAIT. While
the stop is latched the Gateway refuses every skill, including `mirror_stop`.
Re-arm by requesting `SetRuntimeMode(idle)` on `/runtime/set_mode`, which
clears the latch; the session stays in WAIT until the next `mirror_start`.
