# Cross-host clock calibration

`ibrobot-clock` records a Linux PHC and host clocks without adjusting any clock.
LinuxPTP (`ptp4l`, `pmc`) owns hardware synchronization; this tool records its
quality and maps timestamps offline. To apply the mapping to CTF/logs and reuse
the analyzer on a common reference axis, see [trace_merge.md](trace_merge.md).

For the tested Ubuntu/Ascend310P configuration, permissions, synchronization
checks, simultaneous capture and cleanup, follow the
[two-host PTP runbook](ptp_two_host_runbook.md).

## Reference and prerequisites

- Use a dedicated PTP domain with one pinned grandmaster. Both ports must use
  hardware timestamping and expose a PHC via `ethtool -T INTERFACE`.
- Run `ptp4l` separately with a working network configuration. The collector
  needs read access to `/dev/ptpN` and access to the local PMC Unix socket.
- Do not run `phc2sys` to alter system realtime for this workflow. The reference
  is the experiment's grandmaster PHC, **not UTC**. Existing NTP may keep running.
- Use a new shared session ID after restarting or stepping the grandmaster.
  A clock identity alone does not identify its time epoch. Do not combine
  separate sessions, host reboots, or discontinuous reference epochs.
- This implementation supports a single PTP port per daemon. Multiple-port,
  partial, or duplicate PMC replies fail closed.

The collector requires `linuxptp` and `ethtool` installed on the host. Neither
the CLI nor the offline API installs packages, changes permissions, opens
network listeners, starts a PTP daemon, or sets the system/PHC time.

## Capture

From each initialized workspace, after PTP is running:

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-clock capture \
  --phc /dev/ptp0 --interface enp129s0 \
  --pmc-socket /tmp/experiment-ptp.sock --domain 97 \
  --grandmaster c85309.fffe.a775b6 --session-id experiment-unique-id \
  --duration 120 --rate 50 --output /absolute/path/host-clock.jsonl
```

Replace interface, PHC, socket and grandmaster with actual values on each host.
The session ID, domain and grandmaster must agree across hosts. Capture must
span the trace interval; extrapolation outside the capture is not permitted.
Durations are bounded to 600 seconds and rates to 1-100 Hz. Output creation is
exclusive: an existing capture is never overwritten. Ctrl-C preserves samples
already written. A completed collection does not imply successful lock: inspect
the qualified/invalid interval counts.

Each sample selects the shortest of nine complete read windows. Realtime,
monotonic raw, boottime and PHC reads lie within the recorded monotonic bracket.
Sampling uses read-only `clock_gettime` on the PHC descriptor. Status polling is
on a separate thread with a bounded subprocess timeout so a stalled `pmc` does
not silently stop clock sampling. Raw PMC replies and errors are preserved.

JSONL contains a versioned header with machine ID, boot ID, PHC/interface,
reference identity, policy assumptions and tool provenance, followed by raw
status records and samples with their associated status snapshot.

## Quality policy

Default policy is recorded in every capture:

| Parameter | Default |
|---|---:|
| Per-host mapped uncertainty ceiling | 50 microseconds |
| Combined difference ceiling | 100 microseconds |
| Maximum adjacent sample gap | 100 milliseconds |
| Maximum status / last sync age | 2.5 seconds |
| Distinct qualified status updates before admission | 3 |
| Clock rate envelope | 1000 ppm |
| Sync error growth allowance | 10 ppm |
| Hardware/PHY timestamp bias allowance | 1 microsecond per host |

The conditional estimate includes the full read window, slave master-offset
residual, full mean path delay as an asymmetry allowance, sync-age growth,
timestamp bias and bounded-rate interpolation uncertainty. Both endpoint
uncertainties are added for a cross-host difference. No negative delay is
clamped into a valid positive measurement.

These are **engineering assumptions, not certified accuracy bounds**. A servo
residual alone cannot reveal constant path asymmetry or PHY timestamp bias.
The asymmetry allowance assumes non-negative physical path delays, and the
rate envelopes must hold between observations. Unobserved discontinuities or
larger hardware bias invalidate that estimate. Validate assumptions against
the actual platform and retain independent held-out measurements; use external
timing instrumentation for an absolute accuracy certification.

Mappings reject wrong hardware/interface/reference, non-locked or stale PTP,
excess uncertainty, sample gaps, host/boot mismatch, observed clock steps,
suspend discontinuities and ambiguous realtime epochs. They require fresh
status updates to settle after an observed fault. Only valid adjacent samples
can be interpolated. An observed reference step between qualified samples
invalidates the rest of that capture; start a new reference session. Original
timestamps are retained in the capture.

## Offline API

```python
from ibrobot_tracing.clock_calibration import load_calibration, reference_difference

edge = load_calibration("edge-clock.jsonl")
cloud = load_calibration("cloud-clock.jsonl")
print(edge.summary())

# Supply identities from the event/trace provenance, not an assumed checkout.
start = edge.map_time(edge_event_ns, clock="monotonic_ns",
                      machine_id=edge_event_machine_id, boot_id=edge_event_boot_id)
end = cloud.map_time(cloud_event_ns, clock="monotonic_ns",
                     machine_id=cloud_event_machine_id, boot_id=cloud_event_boot_id)
delta = reference_difference(start, end, edge, cloud)
if delta.valid:
    print(delta.timestamp_ns, delta.uncertainty_ns)
else:
    print(delta.reason)
```

Supported source clocks are `monotonic_ns`, `monotonic_raw_ns`, `realtime_ns`.
Epoch-sized integer arithmetic avoids float rounding of nanoseconds. A signed
delta is a time difference, not necessarily a network delay: event boundaries
and request/frame association must still be correct. The API only accepts
datasets up to 64 MiB. It rejects non-monotonic capture order rather than sorting
reboots or discontinuities into a fictitious continuous clock.

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-clock inspect /absolute/path/host-clock.jsonl
```

The [trace merge workflow](trace_merge.md) consumes capture provenance and this
mapping to create a calibrated analysis archive. Transport metrics still need
explicit integration; running this collector alone does not correct arbitrary
cross-host timestamp subtraction in business code.
