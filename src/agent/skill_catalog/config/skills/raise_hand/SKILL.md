# raise_hand

Raise one hand, as when volunteering or asking for attention.

Runs the robot runtime's `raise_hand` named motion through the
`runtime_named_motion` executor. While the runtime replays gestures on the
arm stream (`stream` mode) a stop request halts the arm where it is, and
"continue" resumes the gesture from that point; otherwise the motion cannot be
cancelled once started and only the runtime stop service interrupts it.
