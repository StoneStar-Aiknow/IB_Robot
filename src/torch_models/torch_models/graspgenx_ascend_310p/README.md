# GraspGen-X Ascend310P

This directory owns the repository-native GraspGen-X implementation for
Ascend310P. It holds the Ascend PTV3 encoder, the generator with its offline
diffusion loop, the loader for the custom Ascend C operators the encoder calls,
and the assembly that puts the three together on an NPU.

The deployment is selected by the runtime identity:

```text
interface:  tensor_model
model_type: graspgenx
backend:    torch
device:     npu
```

`backend` is `torch`, not `ascend`: the PTV3 encoder has no compiled form, so
the model runs in this Torch process and the compiled diffusion loop runs
inside it, reading and writing Torch's own device memory. The bundle declares
that graph as a sha256-verified artifact with an empty `execution` list, which
is what lets a Torch deployment carry it — the framework verifies and resolves
the file, and this package drives it.

## What runs where

Not everything is on the device, and that is deliberate rather than unfinished:

| Stage | Where | Why |
|---|---|---|
| Serialization sort, per-stage pooling reduce, embedding hash | host | measured faster there; the shapes are irregular and data-dependent |
| CPE, attention, FFN | NPU, FP16 | `graspgenx_subm` plus `torch_npu` fused operators |
| 20-step diffusion loop | NPU, one offline model | ~1500 operator launches become one execute |
| Discriminator | NPU | shares the encoders' coordinate work through the request scope |

One request encodes the same point cloud twice, for the generator and the
discriminator. `shared_geometry` is a per-request, per-thread scope that lets
the second reuse the first's coordinate-derived work; `GraspGen.forward` enters
it through the `request_scope` seam that patch 0003 of the upstream wheel adds.

## Prerequisites

Three things must be in place before a deployment loads, and none of them is
carried by the model bundle.

**The upstream wheel must be the audited `1.0.0+ibrobot.2` build.** Patches
0002-0004 in `third_party/patches/graspgenx/b942909/` open the seams this
package needs: an explicit `initial_noise` (on a 310P `torch.randn(device="npu")`
segfaults the process), an injectable generator class and request scope, and
per-request tensors placed on the device the weights are on rather than with
`.cuda()`. `scripts/setup.sh` installs it and asserts the version.

**The custom operators must be built on the host that runs them**, by
`scripts/setup/install_graspgenx_ascend_ops.sh`. Neither product of that build
is portable: the operator package is compiled for a CANN release and the
`ascend310p` compute unit, and the Torch bridge links the installed Torch's ABI.
See `src/torch_models/torch_models/graspgenx_ascend_310p/custom_ops/kernels/manifest.yaml`.

**The two products go to different places, and that is load-bearing.** The
Torch bridges land in the per-host cache; the CANN vendor packages are
installed into the CANN tree's own `opp/vendors/`, because that is the only
location CANN reuses their packaged binary kernels from. So the build needs
write access there, and `--root` moves only the bridges:

```bash
root=~/.cache/ibrobot/graspgenx_ascend_ops   # or wherever --root built them
export IBROBOT_GRASPGENX_ASCEND_OPS="${root}"
# and deliberately NOT ASCEND_CUSTOM_OPP_PATH -- see below
```

`IBROBOT_GRASPGENX_ASCEND_OPS` is the only place the bridge location is written
down: this package reads it, and a launch that has to carry it forwards the
variable rather than recomputing a path. Export it even when the bridges sit in
the cache default, or the launch has nothing to forward and the operators fail
to load.

**Do not set `ASCEND_CUSTOM_OPP_PATH` for these packages.** It selects CANN's
external-install branch, which registers their operator-info, proto and tiling
stores but *not* the binary store: GE then recompiles every operator once per
process. On an idle 310P that is 17.9 s, 11.2 s and 12.4 s against 6.4 s, 3.4 s
and 3.4 s when the same kernels are found through `opp/vendors`, and the full
first served request is 44.7 s against 12.3 s. A value inherited from
elsewhere is left alone, because it may name someone else's package.

## Shape and version contracts

The offline diffusion loop is compiled for **one grasp count and one step
count**, and for one CANN release and SoC. Three checks exist because the
failure it prevents is silent rather than loud — the generator declines a graph
it cannot use and runs the eager loop instead, which looks only like the
deployment being inexplicably slow:

- the packager refuses a graph whose metadata disagrees with the bundle's
  `num_grasps`;
- the loader refuses a bundle whose `torch_npu` deployment declares no
  `diffusion_loop` artifact, and one whose graph does not match the grasp and
  step counts;
- `assemble` refuses an Ascend SKU other than Ascend310P.

Both halves are built for one CANN release, and the build pins its toolkit
explicitly rather than following `latest`: a host with several CANN installs
would otherwise produce an operator package or a graph that nothing has tested
against the runtime serving it. Build both on the board — compiling the loop
graph there takes about five minutes. A graph from a newer toolkit is not
automatically wrong, but it needs its own on-device comparison against one the
board built before it can be trusted; the 310P's `compatible_version` stops at
C21. Read a graph's provenance with
`strings diffusion_loop.om | grep -A1 atc_cmdline` rather than assuming which
toolkit produced it.

`adapter.json` must declare `num_grasps: 400` and `topk: 400` to match the
compiled graph. 400 is also what the X2 pipeline requests — its sweep over
200, 400 and 800 settled there — so the two agree by measurement rather than
by coincidence. Asking for fewer does not make a request faster: the eager
loop's cost barely depends on the grasp count, so a request the graph does not
cover costs about 315 ms against 196 ms for one it does (measured before the
2026-10-26 recompile; the steady figure has moved, see Measured behaviour, but
the eager path was not re-measured, so the pair is kept as the comparison it was). The adapter therefore
pins the draw to the compiled count whenever the deployment carries a graph,
and lets `max_grasps` bound only what comes back.

## Packaging a bundle

```bash
source .shrc_local
python -m perception_service.package_graspgenx_torch_bundle \
  --bundle-root models/graspgenx_omnihand_left_x2 \
  --deployment torch_npu --device npu \
  --num-grasps 400 --topk 400 \
  --diffusion-loop /path/to/loop_g400.om
```

The graph's metadata JSON must sit beside it. One bundle holds both
deployments: `torch_cuda` keeps the upstream CUDA path untouched, and only the
`(tensor_model, graspgenx, generate_grasps)` identity on `npu` resolves this
one.

## Measured behaviour

Re-measured 2026-10-26 after the environment was put on CANN 8.1.RC1 alone (`latest/`
had been mixing 8.1 and 8.3) and the loop was re-exported and recompiled on the board.
The figures below supersede the earlier ones; where they disagree, these were measured
on the current artifact.

The operating point is now the checkpoint's own budget, 3500 points: the adapter
resamples a served cloud to `data.num_points`, as upstream's batched entry point does.
Through the ROS service on the X2 bottle frame, 400 grasps:

```text
load                 72 s      once per process, not per request
first request       6.1 s
steady state        304 ms
```

Measured on an **idle** board: no other stack running, load average at its
14-15 baseline, and `/proc/sys/kernel/random/boot_id` checked before and after
each run so that a reboot could not be mistaken for a result. The GE arm was
rebuilt from the pre-aclnn sources into a separate operator root and measured
under the same conditions, five requests per run:

```text
                      GE / OpCommand        aclnn
first request            44654.3 ms       6138.7 ms
steady state     302.5 ms (299-310)   303.7 ms (298-308)
load                        72.0 s          66.9 s
```

So the first request drops **44.7 s to 6.1 s**, and **steady state does not
regress**: 1.2 ms apart with the two intervals almost entirely overlapping.

That last point corrects this file's own earlier figures, which are worth
recording because the mistake is easy to repeat. An initial pair of runs gave
GE 321.1 ms and aclnn 330.5 ms and was read as a 9 ms regression caused by the
bridges building ACL descriptors and running tiling per call; a later aclnn run
gave 342.0 ms with a 29 ms spread. All three were taken while another stack
held the NPU, at load averages from 17.7 to 19.4. On an idle board the two
paths are indistinguishable, so that "regression" was contention, measured
against a baseline taken under *different* contention. A steady-state figure
from this board is only meaningful with the load average and the other stacks
recorded beside it.

One real defect was found while chasing that phantom. The stream handed to the
launch closure came from the no-argument `NPUStream::stream()`, which calls
`MakeSureQueueEmpty()` first -- so asking for the stream drained the task
queue once per operator call, giving back exactly the host-side overlap
`RunOpApi` exists to preserve. The code now uses `stream(false)`. A
per-operator benchmark cannot see this, because it synchronises around every
call: it measures 0.39/0.41/0.78 ms with the drain removed against
0.43/0.38/0.78 ms with it.

Per operator, first call against steady, in a process that does nothing else:

```text
                 GE / OpCommand      aclnn        steady GE -> aclnn
GridEncode            17375.8 ms   1379.7 ms      0.29 -> 0.39 ms
BuildSubmMap          11119.0 ms      2.3 ms      0.40 -> 0.41 ms
SubmConv3d            11428.6 ms      2.1 ms      0.59 -> 0.78 ms
```

The 1.4 s left on the first operator is not compilation: it is a
once-per-process aclnn and vendor-package initialisation, which is why the two
operators after it cost ~2 ms. That cost is paid by whichever operator runs
first, not by each of them.

The same frame and bundle through `torch_cuda` on an RTX 3090, driven by the same
harness, take **159 ms** steady with a 481 ms first request, so the board is 1.9x of
CUDA on steady state and now within 13x on the first request rather than 93x. The
2048-point figures below were taken before the aclnn bridges and have not been
re-measured: the board gave 293 ms steady and a 62 s first request through the model
session, against the 196 ms and ~9 s recorded before that -- so the recompiled
`--precision_mode=force_fp32` loop is slower than its predecessor on both counts, and
the earlier claim of 1.6x of CUDA no longer holds. Of that 62 s, ~40 s was the operator
compilation the bridges no longer pay.

**Rotations are not the weak point, and the recorded ~2e-3 cannot be reproduced.**
Orthonormality measures 4.4e-07 to 5.3e-07 on the board against 3.8e-07 to 4.5e-07 on
CUDA. The ~2e-3 this file used to report is not reproducible on either board path: on one
cloud in one process, the compiled FP32 loop gives 4.856e-07 and the **eager** path, with
the same FP16 attention, FFN and CPE, gives 4.849e-07. So that figure's provenance is
unknown, and three explanations for it are ruled out by measurement rather than argument:

* not the FP16 attention/FFN/CPE — the eager path has them and measures 4.8e-07;
* not the offline loop — removing it changes the figure by 7e-10;
* not the 2026-10-26 re-export and recompile — old and new `diffusion_loop.json` are
  identical apart from the added `generator_binding` (same 1723 nodes, 366 initializers,
  same shapes), and the compile log shows `--precision_mode=force_fp32` both times.

Note that `_ORTHONORMAL_TOLERANCE` in the adapter, and the served-request check, are 2e-3
— sized for the old figure and now four orders of magnitude loose. They are a filter
against a mis-transposed or quantised graph, so leaving them wide is deliberate, but they
cannot be read as evidence about this path's accuracy.

**Translation and ranking were compared on device, with a same-device control.** On one
real segmented cloud resampled to 3500 points, `verify_graspgenx_pose_parity.py --cloud`
gives a cross-device grasp-centroid distance of 8.8-17.5 mm and a nearest-neighbour
median of 15.5-16.8 mm, both inside the 8.5-27.3 mm and 16.1-17.9 mm bands that four
CUDA seeds produce among themselves -- so no translation deviation attributable to the
device is detectable at this level. `verify_graspgenx_confidence_ranking.py`, which
scores one fixed pose set on both devices so the confidences are paired, gives a
cross-device Spearman of 0.99956-0.99990 against a same-device band of 0.99946-1.0, and
the confidence ranks the reach filter walks (8-36 and 45-99 on the 2026-09-14 X2
recordings) retain 27-29 of 29 and 53 of 55 either way.

What those two do **not** establish: pointwise numerical parity, which needs identical
noise rather than independent draws, and anything beyond this one frame, one object and
one point count. Note also that the coverage line
`verify_graspgenx_pose_parity.py --compare` prints is meaningless without a same-device
control beside it: at its 5 mm tolerance, against a ~16 mm median sample spacing, two
independent draws scored 0/10, 0/50, 1/200, 1/400 **on one device** as well.

`verify_graspgenx_loop_parity.py` fixes the loop's four input tensors and
compares the compiled graph against an eager CUDA reference using upstream's
own denoising head and DDPM scheduler. Both sides record post-step latents;
the reference also accumulates likelihood independently. Translation errors
are divided by the generator's `kappa` before conversion to millimetres.
This covers the denoising loop, excluding the PTV3 encoder and restoration
of the point cloud's world-frame origin.

The earlier 2026-10-18 table is withdrawn: its reference recorded pre-step
latents while the graph recorded post-step latents, and its reported
millimetres omitted the `kappa` scale. The corrected diagnostic needs a new
paired CUDA/310P run; no corrected board accuracy result is available yet.

`verify_graspgenx_pose_parity.py` serves complete requests, but CUDA and NPU
sample independent random streams even when seeded. Its comparison therefore
reports geometric coverage, using one-to-one pose matching within explicit
translation and rotation tolerances. Confidence selects each side's top-k
set; sorted row indices are never treated as corresponding grasps. This is
a sampling diagnostic, not an elementwise device parity test. The board run
still needs its `lxml`/`libxslt` environment repaired (`libxslt.so.1` requests
`LIBXML2_2.4.30`); only the CUDA request has been exercised here.

Exported constants and loop metadata now bind the graph to the SHA-256 of
both the generator checkpoint and configuration, plus inference-step and
grasp counts. Conversion checks the metadata digest; packaging and runtime
loading reject missing or stale bindings. Re-export constants and recompile
existing OM loops before packaging them. Updating shared generator assets in
a CUDA deployment also validates any retained NPU deployment first.

## Diagnostics

- `src/torch_models/test/test_graspgenx_ascend_ops_on_device.py` checks the
  built operators against independent computations: the reference serialization
  encoding, and a dense gather-and-accumulate.
- `src/perception_service/test/test_graspgenx_served_request.py` serves one
  request through the manifest, registry, adapter and session, and fails if any
  loaded module comes from outside the workspace. `DEPLOYMENT=torch_cuda` runs
  the same request through the upstream CUDA path, so the two can be compared
  without a second script to keep in step.
- `src/perception_service/test/graspgenx_npu_request_profile.py` attributes a
  request's time to the generator, the discriminator and the sampler.

Both skip themselves off-device. The one-off structural check that every file
a request opens lives inside the workspace was run once under
`strace -f -e trace=openat`: 92552 syscalls, none of them naming the
GraspGenX-ascend checkout, with the operator bridges, the offline model and
both checkpoints resolving to the operator cache and the bundle.
The module scan above is the cheap standing version of that check.

A test or tool that calls the operators directly no longer has to set
`torch.npu.set_compile_mode(jit_compile=False)` for their sake, and the reason
is worth writing down because this README got it wrong twice.

The flag does reach CANN. torch_npu maps it to
`aclSetCompileopt(ACL_OP_JIT_COMPILE)`, and a CANN debug log from the board
shows `ge.jit_compile` set to 0 a quarter of a second before the first
operator's compile request. An earlier version of this section said the switch
"never reached these operators"; that was wrong. What it cannot do is give GE a
binary it has not been told about: a custom package's *binary* store comes from
the GE option `op.binary.custom`, and under the external install that option is
never set ("GetCustomBinaryPath Not found custom binary key:op.binary.custom in
options"), so this package's own
`op_impl/ai_core/tbe/kernel/config/ascend310p/binary_info_config.json` is never
read and FE falls through to a full compile, reporting "Cur op[GridEncode1] not
support dynamic rank in binary info store". That is the 17.4 s per operator,
paid with the flag already False.

The earlier claim that *no* configuration could fix this was also wrong. FE
generates `op.binary.custom` by itself from `opp/vendors/config.ini`
(`Configuration::LoadOppConfigFile` / `ResolveBinaryPath`), so installing the
packages there -- which is what setup now does -- makes GE read both
`binary_info_config.json` files, load the three `.o` kernels and skip the
compile: `ParallelCompileOp` drops from 10.6 s to 1.5 ms. The flag is
irrelevant either way.

What remains on the GE path is `ParallelPreCompileOp`, ~2.94 s per operator,
which costs the same whichever way the packages are installed: it is FE's
prebuild step, not the kernel compile. That is why the first *call* of each
operator still costs 6.4 s, 3.4 s and 3.4 s rather than milliseconds, and why
the warmup below exists. Note that GridEncode logs "not support dynamic rank in
binary info store" even while reusing its binary, so that line alone does not
prove a compile happened.

`assemble` still sets the flag and asserts it afterwards, for the built-in
operators the model runs around these three.

### The operators are warmed when the model loads

The cost above is per process per operator and **independent of the input**: a
64-point call pays it in full, after which a real 3500-point request costs
1.2 ms, 1.5 ms and 0.9 ms. It is also paid on the first *call*, not at load, so
without a warmup it lands on the first grasp the robot is asked for -- 12.3 s
for a request that then takes 300 ms, once per model-service lifetime.

`perception_service.model_session_builders` therefore calls each operator once
with 64 points as soon as an npu Torch session has loaded its model. The first
served request drops from 12.3 s to 6.2 s, with 10.7 s moved into a load that
already takes tens of seconds. A failing warmup is logged and swallowed: it
would only mean a slow first grasp, which is what the code did before, and
refusing to serve would turn a latency problem into an outage.

The remaining 6.2 s is not these operators -- a warmed process still spends
~4 s on other first-call costs inside the model, and the same is true on CUDA.

