# aero_hand_hardware

`aero_hand_hardware` owns the Aero Hand serial SDK and exposes a ROS 2 command/state boundary.
It is intentionally independent of `ros2_control`; the hand is not part of the robot URDF,
MoveIt model, controller manager, or the arm `/joint_states` stream.

## Interfaces

- Command: `/aero_hand_right/commands` (`std_msgs/msg/Float64MultiArray`, seven radians)
- State: `/aero_hand_right/joint_states` (`sensor_msgs/msg/JointState`, seven radians)
- Executable: `ros2 run aero_hand_hardware aero_hand_node`

The subscriber only caches commands. A 50 Hz timer publishes the cached command to the driver,
while state readback defaults to 20 Hz. A stale command holds the last hardware position.

All serial I/O runs on a dedicated driver thread that exclusively owns the port. The ROS timer
never blocks on serial: `set_joint_positions()` queues the latest command and returns, and
`get_joint_positions()` serves the most recent cached readback.

Crucially, the readback is also non-blocking *inside* the I/O thread. The Aero protocol is
half-duplex, but `CTRL_POS` is fire-and-forget with no ACK, so only reads wait on bytes. Rather
than calling the SDK's blocking `get_joint_positions_compact()`, the driver issues the `GET_POS`
request itself and polls for the 16-byte reply across later cycles using `in_waiting`. A hand
that answers slowly — or never — therefore costs zero command cadence. This matters because the
SDK's `read()` uses a 10 ms inter-byte timeout that a partially answered frame can stretch well
past one control period; left synchronous, it would cap the real command rate at roughly
`1 / read_latency` (about 6 Hz for a 160 ms reply) even with commands queued at 50 Hz.

A request that goes unanswered for `read_reply_timeout` (default 0.3 s) is abandoned and counted,
so a dead readback never wedges the loop. Queued commands coalesce: if the I/O thread is busy,
only the newest command is sent, so a backlog never replays stale poses. Readback older than
`state_timeout` (default 0.5 s) raises `TimeoutError` rather than serving a stale position.
`read_failure_count` and `write_failure_count` expose cumulative I/O errors for diagnostics.

Mock mode stays fully synchronous and spawns no thread, keeping hardware-free runs deterministic.

Real mode is strict: missing SDK, serial device, or connection errors stop node startup. Set
`mock:=true` explicitly for deterministic hardware-free operation.

The ROS boundary always uses radians. `AeroHandNode` converts commands to degrees immediately
before `set_joint_positions()` and converts compact SDK readback back to radians. No other layer
performs an Aero unit conversion.

Emergency stop suppresses commands (or sends the configured safe pose once), while state readback
continues so diagnostics and recording retain the actual hand position.

The first three values follow the SDK compact representation: CMC abduction, CMC flexion, and
combined MCP/IP contraction. They are mechanism coordinates, not named human gestures such as
"opposition". Human-skeleton retargeting belongs to `robot_teleop`; this driver does not infer poses.

## Configuration

The main launch owns the node through `robot_config`:

```yaml
auxiliary_actuators:
  aero_hand_right:
    type: aero_hand
    active_control_modes: [teleop]
    mock: false
    port: /dev/ttyACM0
    joint_names: [right_thumb_cmc_abd, right_thumb_cmc_flex, right_thumb_mcp_ip,
                  right_index, right_middle, right_ring, right_pinky]
    command_topic: /aero_hand_right/commands
    joint_state_topic: /aero_hand_right/joint_states
```

For real hardware, `robot_config` also requires every listed joint in the active
safety `joint_limits`. Launch passes ordered `command_lower_limits` and
`command_upper_limits` to this node. The hardware boundary clamps commands again,
even when a publisher bypasses `robot_teleop`.

The E-stop latch exists in both the ROS node and the driver I/O thread. Activating
it clears queued motion and waits for any write already holding the serial lock;
after the latch returns, only the configured safe pose may write until release.

`aero-open-sdk==0.1.0.dev1` is installed through `requirements/hardware.txt`. The Aero Hand is
not represented in URDF and its state topic is intentionally separate from `/joint_states`.

The serial user must have access to the device. Add the user to `dialout` for persistent access,
or grant a temporary ACL during bring-up:

```bash
sudo usermod -aG dialout "$USER"
sudo setfacl -m "u:$USER:rw" /dev/ttyACM0
```

Device names can change after reconnect. Confirm the Aero Hand port with `udevadm info` before
starting real mode; the reference hardware used during bring-up enumerated as `/dev/ttyACM0`.
