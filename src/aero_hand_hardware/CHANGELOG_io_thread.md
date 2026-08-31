# Aero Hand Driver: Serial I/O Thread Fix

## Problem

The Aero Hand node ran all SDK operations synchronously in a single 50 Hz ROS timer:

1. Write command to serial
2. Every 2.5 cycles (20 Hz), read joint state from serial — **blocks up to 160 ms**
3. Continue

`pyserial` reads with `timeout=0.01` (10 ms inter-byte), but `read(16)` resets the timer
on every received byte. A partially-answered frame (e.g., 3 bytes arrive, then silence)
blocks far longer than 10 ms. Empirically measured worst-case on this hardware: **160 ms**,
versus a 20 ms control budget.

Result: **8 dropped control cycles every 2.5 cycles** → visible periodic stuttering.

## Solution

Two separate blocking problems had to be fixed; the first alone is not sufficient.

### 1. The ROS timer must not block

All serial I/O moved to a dedicated driver thread that exclusively owns the port:

- `set_joint_positions()` queues a command and returns immediately
- `get_joint_positions()` reads the most recent cached state and returns immediately

This stops the ROS executor from stalling, but on its own it does **not** fix the hand.

### 2. The I/O thread must not block either

A first attempt kept the loop as `write → blocking read`. That still gated write cadence on
read latency: commands enqueued instantly but only reached the SDK every 160 ms, i.e. a real
control rate near **6 Hz**. Enqueue latency looked perfect while the hand still stuttered.

The fix relies on a protocol detail: `CTRL_POS` is fire-and-forget with no ACK, so only reads
wait on bytes. The driver now issues the `GET_POS` request itself and polls for the 16-byte
reply on subsequent cycles via `ser.in_waiting`, decoding it inline. Writes proceed at full
rate while a reply is outstanding. An unanswered request is abandoned after
`read_reply_timeout` (default 0.3 s) and counted in `read_failure_count`.

E-stop safe poses use a `blocking=True` path to preserve synchronous error reporting: the write
holds a second lock (`_port_lock`) that the I/O loop also takes, so it waits at most one cycle.

## Changes

### `aero_hand_driver.py`

- Added `command_frequency`, `state_frequency`, `state_timeout`, `read_reply_timeout` params
- `connect()` spawns `_io_thread` (daemon, unless mock)
- `set_joint_positions(positions, *, blocking=False)` queues or blocks
- `get_joint_positions()` returns cached state or raises `TimeoutError` if stale
- `_request_state()` / `_collect_state_reply()` implement the non-blocking request/poll split
- Added `read_failure_count` / `write_failure_count` properties for diagnostics
- `disconnect()` signals the thread and joins with 2s timeout

### `aero_hand_node.py`

- Passes `command_frequency` and `state_frequency` to driver at construction
- E-stop safe pose uses `set_joint_positions(..., blocking=True)`

### `test_aero_hand_driver.py`

- Updated safe-pose test to expect `blocking=True` signature
- `_SlowHand` now emulates the half-duplex protocol (request, delayed reply via `in_waiting`)
  and records `write_times`, so tests can assert on cadence at the SDK rather than at the API
- Added 8 tests, including the two that matter most:
  - `test_commands_reach_the_sdk_at_full_rate_despite_a_slow_readback` — asserts ≥35 writes/s
    and worst inter-write gap <60 ms with a 160 ms readback
  - `test_unanswered_reads_are_abandoned_without_stalling_writes`
  - `test_state_readback_still_succeeds_while_commands_stream`
  - plus enqueue-latency, coalescing, staleness, and shutdown coverage

The driver test suite covers these behaviors.

## Verification

Synthetic benchmark against a 160 ms blocking readback:

```
OLD synchronous worst control cycle:   160.2 ms  (budget 20.0 ms)
NEW threaded  worst set_joint_positions: 0.027 ms
```

Enqueue latency alone was misleading — the decisive check is write cadence at the SDK, now
covered by `test_commands_reach_the_sdk_at_full_rate_despite_a_slow_readback`.

## Backward Compatibility

Mock mode stays fully synchronous — no thread, no nondeterminism.

Real-hardware API is unchanged except for the new optional `blocking` kwarg. Existing callers
that never specified `blocking` continue to queue as before; only the node's E-stop path uses
the new blocking mode.

Nodes using the old driver will fail to construct with `TypeError: unexpected keyword argument
'command_frequency'` — this is intentional; the threading change is not drop-in compatible
with external driver instantiation that doesn't pass frequencies.

## Known Limitations

- State readback frequency is fixed at driver construction; dynamic adjustment requires reconnect
- `read_reply_timeout` defaults to 0.3 s. Because the poll is non-blocking, waiting longer is
  nearly free, so this is deliberately generous — a hand answering in 160 ms must not be cut off.
  `state_timeout` (0.5 s) must stay above it or state will always read as stale.
- `_request_state()` / `_collect_state_reply()` reimplement the SDK's `get_actuations()` framing
  against module-private names (`GET_POS`, `_UINT16_MAX`, `actuation_*_limits`). An SDK upgrade
  that changes the wire format or these internals will break readback. The command path still
  goes through the public `set_joint_positions()`, so commands are unaffected. Pin the SDK
  version and re-run the driver tests after any bump.
- Thread join timeout is hard-coded 2 s; if `disconnect()` lands during a hung read, it exits anyway

## Deployment Notes

No YAML or launch changes required — frequencies are already in the config and were previously
passed only to the node's timer. Now they're also forwarded to the driver at construction.

After deployment, monitor `driver.read_failure_count` and `driver.write_failure_count` via
a diagnostics aggregator or periodic log. A rising read failure count suggests either:
1. Serial noise/cable issues
2. `read_reply_timeout` too tight for the device response latency

Note that read failures no longer imply command stuttering — that decoupling is the point of
this change. Commands and state health should now be diagnosed independently.
