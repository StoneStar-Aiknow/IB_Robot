# return_to_task_pose

Reach an exact joint posture named in the deployment's local task-pose file.
The target is never interpreted as free text. Fresh measured feedback must
remain within the configured tolerance for the stable interval. Unknown names,
changed snapshots, unsafe paths, cancellation or missing feedback fail closed.
Without `hands` the operation preserves head and hand settings; it never enters zero torque.

A pose may also declare `hands: {left: open, right: open}`. The arms return first,
planned from the measured state; then the hands open left before right, and the
arms and hands are confirmed together. This only runs inside the Gateway task
that owns the execution, needs the pose's `unloaded_operation_approved` (nothing
held), and refuses before any motion when a hand has no verified collision
model or the arm planner cannot cover the current hand shape. A pose without
`hands` behaves exactly as before.
