# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGen is orchestrated by the host, so the orchestration is what these tests drive.

The eight roles are joined by PointNet++ grouping, a DDPM loop and an SO(3) pose
conversion, none of which is in any compiled graph. Those steps used to live in the policy
backend's ``_infer_graspgen``; they live in a ``ModelSession`` subclass now, so the generic
session keeps its straight-through loop and this one overrides only ``_execute``.

The ACL model is faked. What is real is the manifest: every test loads the bundle the
packager writes, so a binding the packager gets wrong fails here rather than on a device.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import (
    GRASPGEN_BATCH,
    GRASPGEN_DEPLOYMENT,
    FakeAclModel,
    FakeAclRuntimeManager,
    package_graspgen_export,
)

from inference_manifest import load_inference_manifest
from inference_service.backends.errors import BackendInferenceError, BackendLoadError
from inference_service.backends.types import RuntimeContext
from inference_service.unified_runtime import ExecutionContext, ModelRequest
from model_utils.graspgen_contract import GRASPGEN_EXECUTION
from perception_service.graspgen_adapter import GraspGenAdapter
from perception_service.graspgen_session import GraspGenAscendSession

_POINT_COUNT = 2048


def _session(bundle, fake_acl, **runtime_options) -> tuple[GraspGenAscendSession, FakeAclRuntimeManager]:
    manager = FakeAclRuntimeManager()
    session = GraspGenAscendSession(
        config=GraspGenAdapter.from_bundle(bundle).config,
        runtime_manager=manager,
        model_factory=fake_acl,
    )
    session.load(RuntimeContext(load_inference_manifest(bundle, GRASPGEN_DEPLOYMENT), runtime_options))
    return session, manager


def _points(count: int = _POINT_COUNT) -> np.ndarray:
    rng = np.random.default_rng(11)
    return np.ascontiguousarray(rng.normal(size=(count, 3)).astype(np.float32) * np.float32(0.03))


def _infer(session: GraspGenAscendSession):
    return session.execute(
        ModelRequest({"observation.object_points": _points()}),
        ExecutionContext("grasp-1"),
    )


def test_the_session_publishes_only_the_two_declared_service_outputs(graspgen_bundle, fake_acl):
    session, _ = _session(graspgen_bundle, fake_acl, random_seed=0)

    result = _infer(session)

    assert set(result) == {"grasp.poses", "grasp.confidence"}
    assert result["grasp.poses"].shape == (GRASPGEN_BATCH, 4, 4)
    assert result["grasp.confidence"].shape == (GRASPGEN_BATCH,)
    assert not [name for name in result if name.startswith(("host.", "internal."))]
    session.close()


def test_the_roles_run_in_the_shared_execution_order_with_the_denoiser_looping(graspgen_bundle, fake_acl):
    """Order is contract: the encoders feed the denoiser, which feeds the head."""
    session, _ = _session(graspgen_bundle, fake_acl, random_seed=0)

    _infer(session)

    steps = GraspGenAdapter.from_bundle(graspgen_bundle).config.diffusion_steps
    first_pass = [role for role in fake_acl.order if role != "denoiser"]
    assert first_pass == [role for role in GRASPGEN_EXECUTION if role != "denoiser"]
    assert fake_acl.order.count("denoiser") == steps
    assert fake_acl.order.index("generator_encoder_head") < fake_acl.order.index("denoiser")
    assert fake_acl.order.index("denoiser") < fake_acl.order.index("discriminator_head")
    session.close()


def test_the_host_computes_every_grouped_neighbourhood_the_compiled_slots_expect(graspgen_bundle, fake_acl):
    session, _ = _session(graspgen_bundle, fake_acl, random_seed=0)

    _infer(session)

    assert fake_acl.instances["generator_sa1"].calls[0][0].shape == (1, 3, 256, 64)
    assert fake_acl.instances["generator_sa2"].calls[0][0].shape == (1, 131, 64, 128)
    assert fake_acl.instances["generator_encoder_head"].calls[0][0].shape == (1, 259, 1, 64)
    session.close()


def test_the_device_linked_embeddings_are_never_copied_through_the_host(graspgen_bundle, fake_acl):
    """The two encoder heads hand their embedding straight to their consumer's input slot."""
    session, _ = _session(graspgen_bundle, fake_acl, random_seed=0)

    _infer(session)

    assert fake_acl.instances["generator_encoder_head"].read_outputs == set()
    assert fake_acl.instances["discriminator_encoder_head"].read_outputs == set()
    assert set(fake_acl.instances["denoiser"].input_overrides) == {0}
    assert set(fake_acl.instances["denoiser"].calls[0]) == {1, 2}
    assert set(fake_acl.instances["discriminator_head"].calls[0]) == {1}
    session.close()


def test_the_denoiser_walks_its_timesteps_down_to_zero(graspgen_bundle, fake_acl):
    session, _ = _session(graspgen_bundle, fake_acl, random_seed=0)

    _infer(session)

    timesteps = [float(call[2][0]) for call in fake_acl.instances["denoiser"].calls]
    assert timesteps == sorted(timesteps, reverse=True)
    assert timesteps[-1] == 0.0
    session.close()


def test_a_seeded_session_denoises_reproducibly(graspgen_bundle, fake_acl):
    first, _ = _session(graspgen_bundle, fake_acl, random_seed=7)
    poses = _infer(first)["grasp.poses"]
    first.close()

    FakeAclModel.instances = {}
    FakeAclModel.order = []
    second, _ = _session(graspgen_bundle, fake_acl, random_seed=7)
    repeat = _infer(second)["grasp.poses"]
    second.close()

    np.testing.assert_array_equal(poses, repeat)


def test_the_grasp_poses_are_valid_rigid_transforms(graspgen_bundle, fake_acl):
    """``rt_to_matrix`` is the only producer of ``grasp.poses``; nothing else validates it."""
    session, _ = _session(graspgen_bundle, fake_acl, random_seed=0)

    poses = _infer(session)["grasp.poses"]

    rotations = poses[:, :3, :3]
    identity = np.einsum("nij,nkj->nik", rotations, rotations)
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(3), identity.shape), atol=1e-4)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-4)
    np.testing.assert_array_equal(poses[:, 3, :], np.broadcast_to([0.0, 0.0, 0.0, 1.0], (len(poses), 4)))
    session.close()


def test_a_reordered_execution_plan_is_refused_at_load(graspgen_export, fake_acl):
    """The roles are not interchangeable, and a manifest is the only thing that says so."""
    package_graspgen_export(graspgen_export)
    manifest_path = graspgen_export.bundle / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    execution = manifest["deployments"][GRASPGEN_DEPLOYMENT]["execution"]
    execution[0], execution[1] = execution[1], execution[0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    session = GraspGenAscendSession(
        config=GraspGenAdapter.from_bundle(graspgen_export.bundle).config,
        runtime_manager=FakeAclRuntimeManager(),
        model_factory=fake_acl,
    )
    context = RuntimeContext(load_inference_manifest(graspgen_export.bundle, GRASPGEN_DEPLOYMENT))

    with pytest.raises(BackendLoadError) as error:
        session.load(context)

    assert error.value.code == "invalid_execution_plan"
    assert fake_acl.instances == {}


def test_a_non_integer_seed_is_refused_before_any_device_work(graspgen_bundle, fake_acl):
    session = GraspGenAscendSession(
        config=GraspGenAdapter.from_bundle(graspgen_bundle).config,
        runtime_manager=FakeAclRuntimeManager(),
        model_factory=fake_acl,
    )
    context = RuntimeContext(load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT), {"random_seed": "7"})

    with pytest.raises(BackendLoadError) as error:
        session.load(context)

    assert error.value.code == "invalid_runtime_options"
    assert fake_acl.instances == {}


def test_an_unknown_runtime_option_is_still_refused(graspgen_bundle, fake_acl):
    """Widening ``allowed_runtime_options`` for the seed must not open the set."""
    session = GraspGenAscendSession(
        config=GraspGenAdapter.from_bundle(graspgen_bundle).config,
        runtime_manager=FakeAclRuntimeManager(),
        model_factory=fake_acl,
    )
    context = RuntimeContext(load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT), {"curvature_log_path": "x"})

    with pytest.raises(BackendLoadError) as error:
        session.load(context)

    assert error.value.code == "invalid_runtime_options"


def test_a_role_that_returns_nothing_is_reported_against_its_own_name(graspgen_bundle, fake_acl):
    class _SilentSa1(fake_acl):
        def execute(self, inputs, *, read_outputs=None):
            outputs = super().execute(inputs, read_outputs=read_outputs)
            return {} if self.role == "generator_sa1" else outputs

    session, _ = _session(graspgen_bundle, _SilentSa1, random_seed=0)

    with pytest.raises(BackendInferenceError) as error:
        _infer(session)

    assert error.value.code == "missing_runtime_output"
    assert "generator_sa1" in str(error.value)
    session.close()


def test_closing_releases_every_role_and_the_acl_lease(graspgen_bundle, fake_acl):
    session, manager = _session(graspgen_bundle, fake_acl, random_seed=0)

    session.close()

    assert sorted(fake_acl.instances) == sorted(GRASPGEN_EXECUTION)
    assert all(model.close_calls == 1 for model in fake_acl.instances.values())
    assert manager.lease.close_calls == 1
