# Calibrated RTP access-unit timing

This is an **offline consumer** of journals already produced by an external
tool or a historical experiment. The current repository does not include an
RTP journal producer: setting `IB_RTP_METRICS_PATH` or
`IB_RTP_METRICS_SESSION_ID` does not enable recording in its sender/receiver.
No bounded writer queue, background writer or media-path hooks are provided by
this feature. Adding a producer requires a separately reviewed runtime change.

The analyzer applies the existing [clock calibration](clock_calibration.md) to
local files. It does not modify `DistributedCloudService`, its
`network_latency_ns` calculation, RTP packets, DDS contracts, business tracing
points or LIBERO configuration. Periodic video-status values may refer to
different frames and different host clocks; they cannot substitute for the
same-access-unit journal below, even if their subtraction is positive.

## Input journal contract

Each UTF-8 JSONL file describes one emitting process. Every line must end in a
newline and contain a JSON object. The header is:

```json
{"kind":"rtp_metrics","schema_version":1,"capture_session":"run-1","machine_id":"host-a","boot_id":"boot-a","process_instance":"process-a","clock":"monotonic_ns"}
```

The four identity strings must be non-empty. `capture_session` must equal the
paired calibration's session ID; machine/boot identity must match that host's
clock capture. Use the [two-host runbook](ptp_two_host_runbook.md) for calibration
requirements. Timestamps must be readings of the producing host's monotonic
clock in nanoseconds, never borrowed from the other host.

Each data row has `kind` (`send`, `receive` or `decode_call`) and the identity
fields in this table; both ends must describe the same actual access unit:

| Fields | Type / constraint |
|---|---|
| `pipeline_id`, `session_id`, `observation_key`, `stream_id` | Non-empty strings; `session_id` is the protocol session, distinct from `capture_session` |
| `session_generation` | Integer in 1..2^63−1 |
| `ssrc`, `rtp_timestamp` | Integers in 0..2^32−1 |
| `last_sequence` | Integer in 0..65535 |
| `first_sequence`, `packet_count` | Required for send/receive; 0..65535 and 1..65536 respectively |

Required time fields are non-negative signed-64-bit integers (booleans are not
integers here): `send_start_ns/send_end_ns`, `receive_ns/reassembly_end_ns`, or
`decode_start_ns/decode_end_ns`, according to `kind`. Optional encode fields
are described in the metrics section below. Include a final
`{"kind":"end","dropped_records":0}` row with the actual non-negative dropped
diagnostic record count. Missing footer or positive drops produce a warning;
records after a footer are rejected. Do not infer media packet loss from
missing diagnostic rows. Test fixtures in `test/test_rtp_metrics.py` illustrate
the reader contract; they are not a production capture tool or performance data.

## Match and calculate

Once externally recorded journals are available, copy both journals and their corresponding clock JSONL files
to the analysis host. Both calibrations must cover the relevant frames and use
the same PTP reference session/domain/grandmaster:

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-rtp-metrics \
  --input /absolute/edge-journal.jsonl /absolute/edge-clock.jsonl \
  --input /absolute/cloud-journal.jsonl /absolute/cloud-clock.jsonl \
  --output /absolute/rtp-report.json
```

Equivalent entry: `python3 -m ibrobot_tracing.rtp_metrics`. The offline API is
`analyze_rtp_metrics([(journal_path, calibration_path), ...])`. Accepts 2–32
input pairs, up to 100,000 records in total; each journal is limited to 64 MiB.
Duplicate input paths, malformed identities and incompatible reference sessions
fail. The report retains source hashes, calibration policy, original records,
valid measurements, unavailable reasons and capture warnings.

The join key is `(pipeline, session, generation, observation, stream, SSRC,
RTP timestamp, last sequence)`. A pair also requires identical first sequence
and packet count. There must be exactly one send and one receive. Repeated or
ambiguous identities, incomplete access units and unmatched records are not
given a delivery duration. The sender capture timestamp is retained for
diagnosis, but equality with the receiver's quantized capture estimate is not
used as the wire identity.

| Metric | Exact boundary |
|---|---|
| `frame_delivery_ns` | Sender starts packetization → receiver enqueues the complete access unit's marker packet |
| `uncertainty_ns` | Sum of both mapped endpoint uncertainties |
| `send_call_ns` | Sender packetization start → completion of its UDP send calls |
| `receive_queue_and_reassembly_ns` | Marker packet enqueue → completed depacketization |
| `encode_call_ns` | Encode call that produced this access unit, only when input/output capture identity matches |
| `decode_call_ns` | Decode submission/drain call for this access unit; not a claim about delayed output-frame latency |

`frame_delivery_ns` is an application-level **frame delivery interval**, not
wire-only network propagation. It includes packetization and sending all
packets, network/OS handling and receive-thread scheduling. We use send start
rather than send completion because the receiver may already have the final
packet before the sender thread takes its post-send timestamp. `send_call_ns`
overlaps frame delivery and must not be added again to that metric.

An external producer must preserve these measurement boundaries, including
receiver enqueue before queue wait. For a batched receive path, document if
`receive_ns` is batch completion/enqueue rather than individual packet arrival;
the analyzer cannot infer or verify a producer's actual instrumentation point.
`decode_call_ns` measures a submission/drain call, not delayed output residence
time. Optional `encode_call_ns` requires matching `encode_input_capture_ns` and
`capture_timestamp_ns` and valid positive `encode_start_ns/encode_end_ns`.
Producer-side input/output identity tracking is outside this offline tool.

Mapping rejects missing/stale calibration, wrong boot, different segments or
combined uncertainty exceeding 100 microseconds. Negative or positive values
not distinguishable from zero within the error budget are explicitly
`negative_or_unresolved_order`, not clamped into a valid latency. Validity is
the same conditional engineering estimate used by clock calibration, not
absolute accuracy certification. Runtime inference does not depend on this
analyzer or its calibration inputs. No business transport configuration change
is needed to analyze existing files.
