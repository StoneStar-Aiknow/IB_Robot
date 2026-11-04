# unitree_g1_adapter

ROS-free Unitree G1 SDK adapter over the official
[`unitree_sdk2_python`](https://github.com/unitreerobotics/unitree_sdk2_python).
It is the single dependency boundary between IB_Robot and the vendor SDK: the
only module that imports a vendor symbol is `unitree_g1_adapter/vendor.py`, and
everything else consumes its injectable surface, so this package imports and
tests cleanly on hosts where the vendor SDK is not installed.

## Responsibility

- Vendor import guarding with setup guidance
  (`UnitreeSdkUnavailableError`); the G1 runtime fails closed when the SDK is
  absent instead of pretending the vendor tier is reachable.
- The `UnitreeSdk` facade: the vendor channel factory is a process-wide
  singleton, so initialization is once-per-process, a conflicting second call
  raises instead of being silently ignored, and every channel/client created
  through the facade is tracked for `close()`.
- Thin channel wrappers: `StateFeed` (newest-sample cache with a monotonic
  stamp; the vendor calls handlers on its own DDS thread) and `CommandWriter`
  (plain-dict slot frames built through verified converters, or ready IDL
  samples; the vendor `Write` boolean is passed through).
- Typed RPC client wrappers (`loco` / `arm_action` / `audio` /
  `motion_switcher`) that normalize the vendor's mixed return shapes onto one
  `SdkCallResult` with an explicit `timed_out` sentinel (vendor code 3104).
- Transport-level converters for the `unitree_hg` IDL (`LowCmd_` fixed 35
  slots, `HandCmd_` unbounded slots), refusing non-finite motor values.

## Prohibited

- ROS dependencies (no rclpy, no message packages — the package has an
  intentionally empty runtime dependency list).
- Re-implementing vendor RPC/transport behavior: identity correlation,
  timeouts and the no-resend rule belong to the official client
  (`ClientBase`); the adapter only maps outcomes.
- Robot semantics: topic names, joint index mappings, control weights, mode/
  stop policies and Dex3/Revo2 hand specifics live in the G1 runtime (PR 2).

## Vendor provenance

- Source: `unitree_sdk2_python` @ `814556d15970dd2ecf1c9984e845ca02ab07e206`
  (upstream has no release tags; the pin is the reviewed revision).
- License: BSD-3-Clause (upstream `LICENSE`).
- Installed by `scripts/setup/install_unitree_sdk.sh` from the gitcode mirror
  (`https://gitcode.com/unitreerobotics/unitree_sdk2_python.git`, same mirror
  convention as `feetech_sdk`'s FTServo fetch) into the workspace venv's
  `.cache/unitree_sdk2_python` (the `pointnet2_ops` cache convention: a
  non-PyPI Python package installed `--no-deps` caches under the venv it is
  installed into), then `pip install --no-deps
  -e`. The editable install is required: upstream `setup.py` declares no
  `package_data`, so a regular wheel build drops the bundled
  `unitree_sdk2py/utils/lib/crc_*.so` native libraries; the installer's
  verification checks for them. `cyclonedds==0.10.2` (upstream's own exact
  pin) is installed alongside; its Linux wheels are x86_64-only, which is why
  the installer reports every other architecture as unsupported and skips.
- Known upstream quirk carried verbatim: `AudioClient.TtsMaker` sends a
  constant `index` (`tts_index += tts_index` never leaves 0).

## Testing

```bash
source .shrc_local
colcon test --merge-install --packages-select unitree_g1_adapter
```

Unit tests run against an injectable fake vendor surface (no DDS, no
network, no device). Wherever the pinned SDK is installed, the same suite
also executes the real-vendor converter smoke (the tests that skip here) —
`unitree_sdk2py` imports only; no participant is created.
