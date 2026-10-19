"""X2 mirror session: runtime ownership, the single output gate and the state skeleton.

Modules:

``runtime``   the runtime/Gateway boundary. Takeover of the ``policy_stream``
              the Gateway already switched to, revocation, and the one
              release every exit path shares, which never requests idle over
              a stop latch.
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

Nothing in this package imports ROS, so all of it can be exercised with a fake
runtime. ``mirror_session_node`` adapts it to ROS.
"""
