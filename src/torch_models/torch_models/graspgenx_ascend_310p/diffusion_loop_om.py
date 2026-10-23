#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the unrolled diffusion loop from an ATC-built offline model.

``ascend/benchmark/build_diffusion_loop_om.py`` emits the graph and prints the
ATC command; this executes the result through pyACL inside the Torch process,
reading and writing Torch's own device memory, so no host round trip is needed.

One `aclmdlExecute` replaces the twenty eager steps -- about 1500 operator
launches -- and runs them in 23 ms against 190 ms. The loop is shape specific:
one model per grasp count and diffusion step count, checked against the
metadata the builder wrote.

Two things about mixing the two runtimes are not optional:

* Torch-NPU defers its launches to a dispatch thread, so a model execute issued
  straight to ACL can overtake the copies that fill the inputs. Both boundaries
  synchronize. Without that the first execute reads uninitialized memory and
  later ones read the previous request's inputs.
* `aclmdlExecute`, the synchronous entry point, took 25 ms for a single
  projection where `aclmdlExecuteAsync` on Torch's stream took 0.25 ms. Only the
  asynchronous form is used.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch

try:  # The module imports on hosts without the Ascend stack.
    import acl
except ImportError:  # pragma: no cover - exercised only off-device
    acl = None


class DiffusionLoopOM:
    """An offline-model diffusion loop bound to persistent device buffers."""

    def __init__(self, model_path, metadata_path=None, device="npu:0"):
        if acl is None:
            raise RuntimeError("pyACL is unavailable; cannot run an offline model")
        self.path = Path(model_path)
        metadata_path = Path(metadata_path or self.path.with_suffix(".json"))
        self.metadata = json.loads(metadata_path.read_text())
        self.device = torch.device(device)

        model_id, status = acl.mdl.load_from_file(str(self.path))
        if status != 0:
            raise RuntimeError(f"failed to load {self.path}: ACL status {status}")
        self.model_id = model_id
        self.description = acl.mdl.create_desc()
        status = acl.mdl.get_desc(self.description, model_id)
        if status != 0:
            raise RuntimeError(f"failed to describe {self.path}: ACL status {status}")

        grasps = self.metadata["grasps"]
        steps = self.metadata["steps"]
        sample = self.metadata["sample"]
        half = self.metadata["half"]
        self.inputs = {
            "observation": self._buffer((grasps, self.metadata["observation"])),
            "initial_noise": self._buffer((grasps, sample)),
            "position_noise": self._buffer((steps - 1, grasps, half)),
            "rotation_noise": self._buffer((steps - 1, grasps, half)),
        }
        self.outputs = {
            "grasps": self._buffer((grasps, sample)),
            "likelihood": self._buffer((grasps, 1)),
            "noise_predictions": self._buffer((steps, grasps, sample)),
            "latents": self._buffer((steps, grasps, sample)),
        }
        self._input_set = self._dataset(
            [
                (self.inputs[name], acl.mdl.get_input_size_by_index(self.description, index))
                for index, name in enumerate(self.metadata["inputs"])
            ]
        )
        self._output_set = self._dataset(
            [
                (self.outputs[name], acl.mdl.get_output_size_by_index(self.description, index))
                for index, name in enumerate(self.metadata["outputs"])
            ]
        )

    def _buffer(self, shape):
        return torch.empty(*shape, dtype=torch.float32, device=self.device)

    def _dataset(self, pairs):
        handle = acl.mdl.create_dataset()
        for tensor, size in pairs:
            expected = tensor.numel() * tensor.element_size()
            if size != expected:
                raise RuntimeError(f"{self.path.name} expects {size} bytes where the binding has {expected}")
            buffer = acl.create_data_buffer(tensor.data_ptr(), size)
            buffer = buffer[0] if isinstance(buffer, tuple) else buffer
            status = acl.mdl.add_dataset_buffer(handle, buffer)
            status = status[-1] if isinstance(status, tuple) else status
            if status != 0:
                raise RuntimeError(f"failed to bind a buffer: ACL status {status}")
        return handle

    def matches(self, grasps: int, steps: int) -> bool:
        return self.metadata["grasps"] == grasps and self.metadata["steps"] == steps

    def run(self, observation, initial_noise, position_noise, rotation_noise):
        """Execute the loop; returns grasps, likelihood and the per-step traces.

        The returned tensors are this runner's own buffers and are overwritten
        by the next call; clone what has to outlive it.
        """
        for name, value in (
            ("observation", observation),
            ("initial_noise", initial_noise),
            ("position_noise", position_noise),
            ("rotation_noise", rotation_noise),
        ):
            self.inputs[name].copy_(value)
        # Flush Torch's queue before handing the device to ACL, and wait for the
        # model before reading its outputs; see the module docstring.
        torch.npu.synchronize()
        status = acl.mdl.execute_async(
            self.model_id,
            self._input_set,
            self._output_set,
            torch.npu.current_stream().npu_stream,
        )
        if status != 0:
            raise RuntimeError(f"offline model execution failed: ACL status {status}")
        torch.npu.synchronize()
        return (
            self.outputs["grasps"],
            self.outputs["likelihood"],
            self.outputs["noise_predictions"],
            self.outputs["latents"],
        )

    def close(self):
        if getattr(self, "model_id", None) is None:
            return
        for handle in (self._input_set, self._output_set):
            for index in range(acl.mdl.get_dataset_num_buffers(handle)):
                acl.destroy_data_buffer(acl.mdl.get_dataset_buffer(handle, index))
            acl.mdl.destroy_dataset(handle)
        acl.mdl.destroy_desc(self.description)
        acl.mdl.unload(self.model_id)
        self.model_id = None

    def __del__(self):
        with contextlib.suppress(Exception):  # pragma: no cover - interpreter teardown
            self.close()
