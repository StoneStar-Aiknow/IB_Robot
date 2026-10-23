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

**Two variables must be exported before the process starts.** `ASCEND_CUSTOM_OPP_PATH`
cannot be set from inside a running process, so a launch file has to carry it:

```bash
root=~/.cache/ibrobot/graspgenx_ascend_ops   # or wherever --root built them
export IBROBOT_GRASPGENX_ASCEND_OPS="${root}"
export ASCEND_CUSTOM_OPP_PATH="${root}/vendors/graspgenx_subm:${root}/vendors/graspgenx_grid"
```

`IBROBOT_GRASPGENX_ASCEND_OPS` is the only place the operator location is
written down: this package reads it, and a launch that has to carry
`ASCEND_CUSTOM_OPP_PATH` forwards it rather than recomputing a path. Export it
even when the operators sit in the cache default, or the launch has nothing to
forward and the operators fail to load.

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
cover costs about 315 ms against 196 ms for one it does. The adapter therefore
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

On Ascend310P1, CANN 8.1.RC1, 2048 points and 400 grasps, served through the
model session:

```text
load                 63 s      once per process, not per request
first request       ~9 s       kernel caches and internal-format conversions
steady state       ~196 ms     generator 120, discriminator 62, sampler 15
```

The same bundle and request on an RTX 3090 through `torch_cuda` take ~127 ms
with a 4.6 s load. Without the offline loop the 310P takes ~320 ms, so the
compiled graph is where the difference between 1.6x and 2.5x of CUDA lies.

Rotations were observed orthonormal to ~2e-3 on the board versus ~5e-7 on
CUDA. Attention, the FFN and the CPE run in FP16 on the board. These figures
alone do not establish pose translation accuracy or device parity.

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

A test or tool that calls the operators directly must set
`torch.npu.set_compile_mode(jit_compile=False)` itself. That flag is not a
performance knob here: left at its default the runtime recompiles a custom
operator instead of using its packaged binary kernels, and the recompile fails
on a host without an Ascend C toolchain with a message that names no caller.
`assemble` sets it and asserts it afterwards, so a deployment never has to.
