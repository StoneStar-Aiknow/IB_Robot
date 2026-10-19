"""Assemble GraspGenX for an Ascend 310P from a model bundle.

The upstream model is built for CUDA: its PTV3 encoder runs there, and its
sampler places tensors with `.cuda()`. This replaces the two encoders with the
Ascend implementations in this package, keeps the rest of the model on the NPU,
and opens the per-request scope the encoders share coordinate-derived work
through.

What stays on the host is deliberate, not unfinished: the serialization sort and
the per-stage pooling reduce are measurably faster there on this part, and the
diffusion loop runs as one offline model driven by `diffusion_loop_om`.

Checkpoint paths come from the bundle's adapter document rather than being
derived from an epoch number, so a repackaged bundle does not need a code
change.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from torch_models.graspgenx_ascend_310p.generator_ascend import GraspGenGeneratorAscend
from torch_models.graspgenx_ascend_310p.ptv3_ascend import (
    PointTransformerV3Ascend,
    shared_geometry,
)

__all__ = ["assemble", "verify_device", "EncoderDeviceBridge"]

# The encoder contract the checkpoints were trained under.
GRID_SIZE = 0.01
PTV3_OUTPUT_DIM = 512

# The only Ascend SKU this package has been measured on. The offline diffusion
# loop is compiled for one SoC and one CANN release, and the operator package
# for `ascend310p`, so a different device would load a graph it cannot honour.
SUPPORTED_DEVICE_SUBSTRING = "Ascend310P"


def verify_device(device_name: str) -> None:
    """Reject an Ascend SKU this package has not been validated on."""
    if SUPPORTED_DEVICE_SUBSTRING not in device_name:
        raise RuntimeError(
            f"GraspGenX on Ascend is validated on {SUPPORTED_DEVICE_SUBSTRING} only; "
            f"this device reports {device_name!r}. Its custom operators are built for "
            f"an ascend310p target and its offline diffusion loop for Ascend310P1."
        )


class EncoderDeviceBridge(torch.nn.Module):
    """Feed the host-bound PTV3 interface from an NPU pipeline.

    The encoder takes and returns host tensors because its serialization and
    pooling run there; the rest of the model is resident on the device. This
    adapts between the two without either side knowing about the other.
    """

    def __init__(self, encoder: torch.nn.Module, output_device: torch.device) -> None:
        super().__init__()
        self.encoder = encoder
        self.output_device = output_device

    def forward(self, data):
        host_data = {key: value.cpu() if torch.is_tensor(value) else value for key, value in data.items()}
        return self.encoder(host_data).to(self.output_device)


def _replacement_encoder(source: torch.nn.Module) -> torch.nn.Module:
    """An Ascend encoder carrying the CUDA encoder's weights."""
    encoder = PointTransformerV3Ascend(
        in_channels=3,
        output_dim=PTV3_OUTPUT_DIM,
        grid_size=GRID_SIZE,
        enable_flash=False,
        shuffle_orders=False,
    )
    encoder.load_state_dict(source.state_dict(), strict=True)
    for module in encoder.modules():
        if hasattr(module, "shuffle_orders"):
            # Shuffled serialization would draw a permutation per forward, which
            # the shared scope cannot reuse and inference does not want.
            module.shuffle_orders = False
    return encoder.eval()


def assemble(
    bundle_root: Path,
    assets: dict,
    *,
    device_name: str,
    num_grasps: int,
) -> torch.nn.Module:
    """Build the model this bundle describes, placed for the NPU.

    `assets` is the bundle's adapter document; `device_name` is what
    `torch.npu.get_device_name` reported, used only to refuse an unvalidated
    SKU before any weight is read.
    """
    verify_device(device_name)

    # Set before graspgenx is imported: its package hook otherwise clones
    # gripper descriptions and checkpoints from Hugging Face into site-packages.
    bundle_root = Path(bundle_root)
    os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(bundle_root / assets["gripper_assets_dir"])
    os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(bundle_root / assets["checkpoints_dir"])

    from graspgenx.models.grasp_gen import GraspGen
    from omegaconf import OmegaConf

    checkpoints = bundle_root / assets["checkpoints_dir"]
    generator_config = OmegaConf.load(checkpoints / "gen/config.yaml")
    config = OmegaConf.load(checkpoints / "dis/config.yaml")
    config.diffusion = generator_config.diffusion
    config.eval.gen_checkpoint = str(bundle_root / assets["generator_checkpoint"])
    config.eval.dis_checkpoint = str(bundle_root / assets["discriminator_checkpoint"])
    # Flash attention is a CUDA path; the Ascend attention is its own.
    config.diffusion.ptv3vanilla.enable_flash = False
    config.discriminator.ptv3vanilla.enable_flash = False

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    # The compile mode is process-global and decides whether an operator runs
    # from the binary kernels its package ships or is recompiled at run time.
    # A recompile needs an Ascend C toolchain that a deployment host need not
    # have, and fails with "inner_interface/inner_kernel_operator_intf.h file
    # not found". Constructing the encoder would set this as a side effect, but
    # stating it here makes the invariant independent of assembly order -- and
    # the diffusion head's own scope only flips the flag when it is already
    # False, so it has to be False before anything else runs.
    torch.npu.set_compile_mode(jit_compile=False)

    model = GraspGen.from_config(config.diffusion, config.discriminator, generator_class=GraspGenGeneratorAscend)
    model.load_state_dict(config.eval.gen_checkpoint, config.eval.dis_checkpoint)
    model.grasp_generator.num_grasps_per_object = num_grasps
    # The per-step grasp history is a diagnostic: it costs a device-to-host
    # copy per diffusion step, and a served request returns only the final
    # poses and their confidences, so nothing ever reads it here.
    model.grasp_generator.record_history = False

    # Build the replacements before moving the model: they load the CUDA
    # encoders' weights, and the originals are dropped straight after so the
    # device move does not carry two copies.
    generator_encoder = _replacement_encoder(model.grasp_generator.object_encoder)
    discriminator_encoder = _replacement_encoder(model.grasp_discriminator.object_encoder)
    model.grasp_generator.object_encoder = torch.nn.Identity()
    model.grasp_discriminator.object_encoder = torch.nn.Identity()
    model = model.to(device).eval()
    model.grasp_generator.object_encoder = EncoderDeviceBridge(generator_encoder, device).eval()
    model.grasp_discriminator.object_encoder = EncoderDeviceBridge(discriminator_encoder, device).eval()

    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False

    # One request encodes the same cloud twice, for the generator and the
    # discriminator. The scope lets the second reuse the first's coordinate
    # work; `GraspGen.forward` enters it per request.
    model.request_scope = shared_geometry

    # Assembly builds submodules that set the compile mode themselves; if a
    # future change leaves it on, the failure is a run-time recompile deep in
    # an operator rather than anything pointing back here.
    if not torch.npu.is_jit_compile_false():
        raise RuntimeError(
            "the NPU compile mode was left at jit_compile=True during assembly; "
            "the operators would be recompiled at run time instead of using "
            "their packaged binary kernels"
        )

    torch.npu.synchronize()
    return model
