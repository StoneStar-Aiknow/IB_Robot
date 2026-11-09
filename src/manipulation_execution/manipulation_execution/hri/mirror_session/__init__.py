"""X2 mirror session: the single output gate and the state skeleton.

The runtime/Gateway ownership rules this session follows — takeover of the
``policy_stream`` the Gateway already switched to, revocation, and the one
release every exit path shares, which never requests idle over a stop latch —
live in ``manipulation_execution.shared.runtime_ownership`` (design D5 of the
package-layout change), shared with the grasp pipeline's runtime stream.

Modules:

``core``      the WAIT / STANDBY / MIRRORING skeleton and the 50 Hz output
              gate. The transitions are fixed; every decision inside them is
              a hook.
``hooks``     the extension points for the mirror behaviour. Every default
              returns the most conservative value and implements no mirror
              behaviour.
``endpoints`` interface lookup by ID from the bound runtime description.
``inputs``    the perception messages, collected into one ``SessionInputs``
              per tick.
``mock_playback``  a window pipeline that plays the mock node's recorded
              animation through the gate, so the data flow runs end to end.
``node``      the ROS adapter (``mirror_session_node`` / the
              ``mirror_session_release`` one-shot).

Nothing in this package imports ROS except ``node``, which adapts the
ROS-free modules above to rclpy; the rest can be exercised with a fake
runtime.
"""
