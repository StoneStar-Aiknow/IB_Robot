# Process Management and Logs

## Ubuntu / openEuler

- Foreground launch is preferred while testing.
- ROS logs normally live under `~/.ros/log/`.
- Use repository cleanup tooling only from the source workspace and only when stale ROS processes
  are confirmed.

## OpenHarmony

- Foreground SSH is preferred for diagnosis because HDC PTY output may truncate long logs.
- There is no systemd. For background execution, use a pidfile and writable log path:

```sh
LOG=/data/local/tmp/ibrobot-launch.log
PIDFILE=/data/local/tmp/ibrobot-launch.pid
ros2 launch <package> <launch-file> <args> >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
```

- Check with `kill -0 "$(cat "$PIDFILE")"`; stop with `kill`, wait, then use `kill -9` only if
  necessary. Remove the pidfile after shutdown.
- Do not use `systemctl`, `pkill -f` as the primary owner check, or `ps | grep <full-command>`.
