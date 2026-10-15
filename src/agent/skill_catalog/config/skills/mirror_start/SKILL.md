# mirror_start

Start mirroring the person in front of the robot with the upper body.

Calls the X2 mirror session's `~/set_session` with `true` through the
`mirror_session` executor. The Gateway first switches the runtime to the
skill's control mode (`model_inference` → `policy_stream`); the session takes
that mode over and waits for a person. While it holds the mode, other motion
skills are refused until `mirror_stop`.

If the start fails after the Gateway switched the runtime out of idle (the
session refused, did not answer in time, or the call was canceled), or the
session is not running at all, the Gateway switches the runtime back to idle
before reporting the failure. It never does so while the runtime stop is
latched.

Once started, mirroring runs until `mirror_stop`; generic stop phrases do not
end it in this version.
