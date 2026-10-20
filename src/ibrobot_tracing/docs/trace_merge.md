# Calibrated multi-host trace merge

The merge workflow consumes each host's original CTF/log plus a clock capture
covering the same run. It preserves raw timestamps and provenance, projects
qualified events onto one PTP reference, and retains rejected events outside
that timeline. Existing CLI analysis and query/projection APIs can read the
resulting `.ibtrace` archive.

## 1. Record fresh traces with provenance

Use the [two-host PTP runbook](ptp_two_host_runbook.md) to start hardware PTP and
clock collection. On **every emitting process**, set the same unique session
ID used by `ibrobot-clock capture --session-id`:

```bash
source .shrc_local
export IB_TRACE_ENABLED=1
export IB_TRACE_LOG_LEVEL=INFO
export IB_TRACE_SESSION_ID=unique-shared-experiment-id
```

Start the local LTTng session and calibration collector before the business
workload. Start the workload only after both collectors cover qualified PTP
intervals; finish the workload before stopping either collector. Record each
host's CTF and its own calibration file. Collection commands and permissions
are documented in the clock runbook; no NTP or system-time changes are required.

The emitter attaches hostname, machine ID, boot ID, PID, native TID, a process
instance ID, session ID and a process-local event sequence when identity is
available. Construct enabled emitters with `get_trace_emitter()` (or
`TraceEmitter`) during process setup, before the workload: this is where host
identity is read and cached. Construction has setup cost; emission only reads
the cache and never acquires the identity-initialization lock or reads files.
Disabled emitters do neither. Existing JSON/logging/clock costs remain; this is
not a zero-overhead or hard-real-time guarantee for tracing as a whole.

Concurrent or reentrant construction does not wait for identity setup. While
setup is busy, or after a nonfatal setup failure, local records are emitted
without provenance; calibrated merge quarantines them as
`missing_event_provenance`. Emission never retries setup. A later explicit
enabled-emitter construction may retry a failed setup. Missing/unreadable or
invalid identity files yield empty fields, cached for that process, and likewise
prevent calibrated merging without stopping local recording.

After `fork`, the child clears the inherited identity without reading files or
generating IDs in the fork hook. Inherited emitters produce local-only records
until the child explicitly constructs an enabled emitter during its own setup;
then all emitters in that process share the new identity and sequence. Do this
before child business work starts. The parent's identity/sequence is unchanged.
No business request IDs, payloads, clocks or transport interfaces are modified.

`IB_TRACE_SESSION_ID` is validated and cached with process identity at setup:
use a non-empty string of at most 256 characters, without surrounding whitespace.
Changing the environment during a run does not relabel that process's events.
Invalid/unset session values become empty; local recording continues and merge
reports the missing field. Before recording, check `/etc/machine-id`,
`/proc/sys/kernel/random/boot_id` and the shared session on both hosts. Missing
identity is never guessed from hostname or silently matched by boot ID alone.
The runbook covers its stated Ubuntu/310P Linux environments, not all boards.

`event_sequence` identifies attempted emission order, not acknowledged delivery.
Invalid timestamps are rejected before allocating it; size limits, logger filters
and downstream capture loss can still leave gaps. Do not infer network packet
loss from gaps. PID/TID queries remain to preserve process/thread correctness;
identity-cache lookup is not a promise that the entire tracer has no system calls.

`timestamp_ns` remains the authoritative realtime event boundary. The parser
also preserves the record's `monotonic_ns`. For deferred events with an explicit
timestamp, monotonic may identify the later emission rather than the original
boundary; the merger therefore maps realtime, not a substituted emission time.
An old capture without provenance or a matching session remains usable for
local analysis, but calibrated merge will quarantine its events. It never
invents host/boot identity from an import path or the current analysis host.

## 2. Merge the original sources

Copy both CTF directories and calibration JSONL files to the analysis machine:

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-trace merge \
  --input /absolute/edge-ctf /absolute/edge-clock.jsonl \
  --input /absolute/cloud-ctf /absolute/cloud-clock.jsonl \
  --output /absolute/run.ibtrace
```

`--input` can be repeated 2–32 times, for original CTF directories or IBTRACE1
log files. All clock captures must agree on session/domain/grandmaster. Output
creation is exclusive. `--max-events` bounds total input events (default and
maximum one million); exceeding the limit fails instead of silently presenting
a truncated run as complete. A failed write may leave a partial file, which
cannot be loaded without a complete matching footer; choose a new output path.

The archive is versioned JSONL: header, calibrated and uncalibrated records,
then a count-checked footer. It preserves input path, reader, calibration identity,
policy, SHA-256, original event fields/timestamp/clock, local monotonic time,
mapping segment and estimated uncertainty. It is not a rewritten CTF file.
Per-record and archive read limits are 1 MiB and 512 MiB. Trace input is
observational evidence, not an authenticated assertion of host identity.

Readers bind each calibrated event to its source ID, calibration hash and
source policy; a custom per-endpoint uncertainty ceiling survives round-trip.
Legacy v1 archives without a `sources` section retain the original 50 µs ceiling,
never a guessed larger budget. Incomplete/mismatched source policy entries fail.
The default two-endpoint difference budget (100 µs) is a separate quantity.

Only documented absolute boundaries are converted: the primary timestamp and
`first_action_execute.publish_end_ns`. Arbitrary user `*_ns` fields and measured
`duration_ns` are not interpreted or rewritten. The original fields remain in
`time_mapping.original_fields`.

Span/parent IDs are namespaced by machine, boot and process instance so equal
IDs in different processes do not alias. Request and flow IDs remain unchanged
to correlate cross-host work. Exact repeats of the same emitted identity with
identical payload and calibration are deduplicated; conflicting repeats fail.
Duplicate flow endpoints with distinct emission identities are ambiguous and
do not get a numeric duration.

## 3. Analyze and inspect rejected events

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-trace summary /absolute/run.ibtrace
ros2 run ibrobot_tracing ibrobot-trace timeline /absolute/run.ibtrace --request-id REQUEST_ID
ros2 run ibrobot_tracing ibrobot-trace flows /absolute/run.ibtrace --request-id REQUEST_ID
ros2 run ibrobot_tracing ibrobot-trace critical-path /absolute/run.ibtrace \
  --request-id REQUEST_ID --format json
ros2 run ibrobot_tracing ibrobot-trace events /absolute/run.ibtrace --uncalibrated
ros2 run ibrobot_tracing ibrobot-trace export /absolute/run.ibtrace --format json
```

`.ibtrace` selects the merged reader automatically. Use `--source-kind merged`
for other extensions. Normal analysis JSON is a report, not an input archive;
use the merge output to reopen the full dataset. Export includes rejected
events and their reasons. Timeline queries include only qualified events;
component lanes distinguish host/process and carry mapping metadata. Text
timeline prints host/PID and clock error while `events`/JSON expose full details.

Missing provenance additionally reports `missing_fields` on rejected records and
`missing_provenance_by_source` in merge metadata. Inspect these for missing
machine/boot/process/session/sequence rather than treating all rejected events
as media loss. Incomplete spans retain the originating process lane as well.

The merged request summary is conservative: if any event in a request cannot
be calibrated, that request is excluded from common-clock request metrics and
is reported with `clock_status=uncalibrated_events`. This avoids a failed or
missing occurrence being hidden behind another successful occurrence. Valid
local spans elsewhere remain available; analyze the original source separately
when investigating local behavior in a rejected calibration interval.

Local span duration statistics continue to use recorded monotonic `duration_ns`.
For merged timeline/span-profile/critical-path geometry, both calibrated
endpoints define the reference interval. A span crossing an observed calibration
segment discontinuity has no complete common-clock geometry. Cross-host flow
and request-boundary estimates expose the sum of endpoint error budgets and
reject incompatible references or estimates above 100 µs. Negative results are
flagged, not clamped. A difference smaller than its uncertainty does not resolve
the temporal ordering of the endpoints.

Calibration quality remains a conditional engineering estimate, not certified
absolute accuracy. Flow duration includes everything between the chosen
application points, such as serialization/logging, transport and callback
dispatch. It must not be labeled pure network propagation time.

## 4. Programmatic use

```python
from pathlib import Path
from ibrobot_tracing.analysis import AnalysisRequest, AnalysisService
from ibrobot_tracing.merging import merge_traces, write_merged_trace

dataset = merge_traces([
    (Path("edge-ctf"), Path("edge-clock.jsonl")),
    (Path("cloud-ctf"), Path("cloud-clock.jsonl")),
])
write_merged_trace(Path("run.ibtrace"), dataset)
result = AnalysisService().analyze(AnalysisRequest(Path("run.ibtrace")))
print(result.coverage["clock_calibration"])
for flow in result.flows:
    print(flow.status, flow.duration_ms, flow.uncertainty_ns)
```

Existing query/projection and JSON/JS-safe serialization are reused. The Web
workbench can discover `.ibtrace` files in its configured trace roots and retain
clock provenance in API responses; see the
[Web setup guide](../../../tools/ibrobot_tracing_web/README.md).
This is generic archive compatibility and does not require scenario-specific
instrumentation or derived events.

## 5. Scope

This change adds generic trace identity/merge infrastructure. It does not add
the full LIBERO observation-to-step-completion business points, change the
RTP latency field, or provide LTTng Live. Existing benchmark coverage can still
report `total_ms` or execution boundaries as missing. Those gaps remain visible
until the corresponding instrumentation is added.
