# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGen is reached the way every other perception model is: a typed ROS service.

It used to be driven through the policy inference service, which meant a grasp request
travelled as an untyped tensor bag and the caller had to know the model's execution plan.
Here it is a ``_SessionPlugin`` like ``SAM2GenerateMasksPlugin`` - one declared service
type, one adapter, one ``ModelSession`` - so these tests pin the three seams that join it
to the rest: the service contract, the session factory, and the ``PointCloud2`` ->
``GraspCandidateArray`` translation the plugin owns.

The ACL layer is faked (see ``conftest``); everything above it is the real plugin.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import FakeAclModel, FakeAclRuntimeManager
from sensor_msgs.msg import PointCloud2, PointField

from inference_manifest import TorchRuntimeProfile, load_inference_manifest
from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry, ResourceDomainAdmissions
from inference_service.backends.types import RuntimeContext
from inference_service.model_service_plugin import ModelServicePlugin
from inference_service.model_sessions import TorchModelSession
from inference_service.runtime_composition import RuntimeDependencies
from inference_service.unified_runtime import (
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeProviders,
    SessionBuilderRegistry,
)
from perception_service.graspgen_adapter import GraspGenAdapter
from perception_service.graspgen_session import GraspGenAscendSession
from perception_service.model_service_plugins import GraspGenGenerateGraspsPlugin
from perception_service.model_session_builders import build_graspgen_session

GRASPGEN_DEPLOYMENT = "ascend_310p"


@pytest.fixture
def runtime_dependencies(fake_acl):
    def build_fake_session(context, *, adapter, allowed_deployments, backend_registry, providers, **_kwargs):
        backend_registry.validate(context, allowed_deployments=allowed_deployments)
        return GraspGenAscendSession(
            device_id=context.device_id if context.device_id is not None else 0,
            config=adapter.config,
            runtime_manager=providers.acl_runtime_provider,
            model_factory=fake_acl,
        )

    backend_registry = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    session_registry = SessionBuilderRegistry()
    session_registry.register(
        "tensor_model",
        "graspgen",
        "generate_grasps",
        "ascend",
        build_fake_session,
    )
    registry_set = RegistrySet(backend_registry, session_registry, RuntimeAssemblerRegistry()).freeze()
    providers = RuntimeProviders.create(FakeAclRuntimeManager(), ResourceDomainAdmissions())
    yield RuntimeDependencies(registry_set, providers)
    providers.close()


def _plugin(validated, options, dependencies):
    return GraspGenGenerateGraspsPlugin(
        SimpleNamespace(bridge=None),
        validated,
        options,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )


@pytest.fixture
def plugin(graspgen_bundle, runtime_dependencies):
    """The real plugin over the real bundle, with only the ACL device replaced."""
    validated = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT)
    instance = _plugin(validated, {"random_seed": 0}, runtime_dependencies)
    yield instance
    instance.close()


def _cloud(points: np.ndarray, *, rgb: bool = False) -> PointCloud2:
    """An organised XYZ cloud, optionally with an RGB channel the decoder must step over."""
    point_step = 16 if rgb else 12
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if rgb:
        fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
    data = bytearray()
    for point in np.asarray(points, dtype=np.float32):
        data += struct.pack("<3f", *point)
        if rgb:
            data += struct.pack("<f", 0.0)
    return PointCloud2(
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * len(points),
        data=bytes(data),
        is_dense=True,
    )


def _points(count: int = 2048) -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.normal(size=(count, 3)).astype(np.float32) * np.float32(0.03) + np.float32([0.4, 0.0, 0.5])


def _request(points=None, *, max_grasps: int = 8, min_confidence: float = 0.0, rgb: bool = False):
    cloud = _cloud(_points() if points is None else points, rgb=rgb)
    cloud.header.frame_id = "camera_color_optical_frame"
    return SimpleNamespace(object_points=cloud, max_grasps=max_grasps, min_confidence=min_confidence)


def _response():
    return SimpleNamespace(grasps=None)


def test_the_plugin_declares_the_typed_grasp_service_and_its_model_type():
    assert GraspGenGenerateGraspsPlugin.service_type == "ibrobot_msgs/srv/GenerateGrasps"
    assert GraspGenGenerateGraspsPlugin.model_type == "graspgen"
    assert GraspGenGenerateGraspsPlugin.adapter_class is GraspGenAdapter
    assert issubclass(GraspGenGenerateGraspsPlugin, ModelServicePlugin)


def test_the_declared_service_response_carries_the_common_diagnostic_fields():
    """``model_service_node`` refuses any service type without them, ``PlanGrasp`` included."""
    from ibrobot_msgs.srv import GenerateGrasps

    fields = set(GenerateGrasps.Response.get_fields_and_field_types())

    assert {"model", "inference_time_ms", "success", "message"} <= fields
    assert "grasps" in fields
    assert set(GenerateGrasps.Request.get_fields_and_field_types()) == {
        "object_points",
        "max_grasps",
        "min_confidence",
    }


def test_the_plugin_hosts_a_graspgen_session_over_the_packaged_bundle(plugin):
    assert isinstance(plugin.session, GraspGenAscendSession)
    assert isinstance(plugin.adapter, GraspGenAdapter)
    assert plugin.runtime_status().ready


def test_a_point_cloud_request_comes_back_as_ranked_camera_frame_grasps(plugin):
    request, response = _request(max_grasps=5), _response()

    message = plugin.handle(request, response)

    grasps = response.grasps.grasps
    assert len(grasps) == 5
    assert message == "generated 5 grasps"
    assert response.grasps.header.frame_id == "camera_color_optical_frame"
    assert all(grasp.header.frame_id == "camera_color_optical_frame" for grasp in grasps)
    confidences = [grasp.confidence for grasp in grasps]
    assert confidences == sorted(confidences, reverse=True)


def test_each_candidate_is_a_flattened_row_major_pose_matrix(plugin):
    """``GraspCandidate.pose_matrix`` is 16 floats; the executor reshapes it back to 4x4."""
    response = _response()

    plugin.handle(_request(max_grasps=1), response)

    pose = np.asarray(response.grasps.grasps[0].pose_matrix, dtype=np.float64)
    assert pose.shape == (16,)
    matrix = pose.reshape(4, 4)
    np.testing.assert_allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-4)


def test_the_grasp_poses_land_back_in_the_frame_the_cloud_arrived_in(plugin):
    """The adapter centres the cloud before inference, so the centre has to be added back."""
    response = _response()

    plugin.handle(_request(max_grasps=4), response)

    translations = np.asarray([np.asarray(grasp.pose_matrix).reshape(4, 4)[:3, 3] for grasp in response.grasps.grasps])
    assert np.linalg.norm(translations.mean(axis=0) - _points().mean(axis=0)) < 0.5


def test_the_width_and_collision_fields_stay_at_their_defaults(plugin):
    """GraspGen scores a pose; it does not measure an aperture or clear the scene.

    ``manipulation_execution`` fills these in from its own geometry pass, so leaving them
    at zero is the honest answer - a fabricated aperture would be acted on.
    """
    response = _response()

    plugin.handle(_request(max_grasps=1), response)

    candidate = response.grasps.grasps[0]
    assert candidate.target_width_m == 0.0
    assert candidate.target_width_quality == 0.0
    assert list(candidate.width_axis_camera) == [0.0, 0.0, 0.0]
    assert candidate.collision_free is False


def test_an_rgb_carrying_cloud_is_decoded_through_its_field_offsets(plugin, fake_acl):
    """A RealSense cloud is not a packed XYZ buffer, and the plugin must not assume it is."""
    plugin.handle(_request(max_grasps=3, rgb=True), _response())
    coloured = fake_acl.instances["generator_sa1"].calls[-1][0]

    plugin.handle(_request(max_grasps=3), _response())
    plain = fake_acl.instances["generator_sa1"].calls[-1][0]

    np.testing.assert_array_equal(coloured, plain)


def test_the_request_confidence_floor_is_applied_before_the_count_limit(plugin):
    generous, strict = _response(), _response()

    plugin.handle(_request(max_grasps=10, min_confidence=0.0), generous)
    plugin.handle(_request(max_grasps=10, min_confidence=1.5), strict)

    assert len(generous.grasps.grasps) == 10
    assert strict.grasps.grasps == []


def test_a_cloud_that_is_not_a_point_cloud_is_refused_before_the_device(plugin):
    cloud = _cloud(_points(4))
    cloud.fields = [field for field in cloud.fields if field.name != "z"]

    with pytest.raises(ValueError, match="missing required fields"):
        plugin.handle(SimpleNamespace(object_points=cloud, max_grasps=1, min_confidence=0.0), _response())


def test_the_session_factory_rejects_a_non_cuda_torch_profile(graspgen_bundle, runtime_dependencies):
    validated = load_inference_manifest(graspgen_bundle, "torch_cuda")
    context = RuntimeContext(validated, runtime_profile=TorchRuntimeProfile(device="cpu"))

    with pytest.raises(RuntimeError, match="requires a typed cuda profile"):
        build_graspgen_session(
            context,
            adapter=GraspGenAdapter.from_bundle(graspgen_bundle),
            providers=runtime_dependencies.providers,
        )


def test_the_session_factory_accepts_a_seed_but_still_closes_the_option_set(graspgen_bundle, runtime_dependencies):
    """Widening the allowed options for ``random_seed`` must not open the set."""
    validated = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT)
    adapter = GraspGenAdapter.from_bundle(graspgen_bundle)

    session = build_graspgen_session(
        RuntimeContext(validated, runtime_options={"random_seed": 7, "device_id": 0}),
        adapter=adapter,
        providers=runtime_dependencies.providers,
    )
    assert isinstance(session, GraspGenAscendSession)

    with pytest.raises(ValueError, match=r"unknown graspgen runtime options: \['curvature_log_path'\]"):
        build_graspgen_session(
            RuntimeContext(validated, runtime_options={"curvature_log_path": "x"}),
            adapter=adapter,
            providers=runtime_dependencies.providers,
        )


def test_the_session_factory_selects_torch_cuda_without_runtime_options(graspgen_bundle, runtime_dependencies):
    validated = load_inference_manifest(graspgen_bundle, "torch_cuda")
    adapter = GraspGenAdapter.from_bundle(graspgen_bundle)

    session = build_graspgen_session(
        RuntimeContext(validated), adapter=adapter, providers=runtime_dependencies.providers
    )

    assert isinstance(session, TorchModelSession)
    with pytest.raises(ValueError, match="does not accept runtime options"):
        build_graspgen_session(
            RuntimeContext(validated, runtime_options={"device_id": 0}),
            adapter=adapter,
            providers=runtime_dependencies.providers,
        )


def test_the_plugin_refuses_a_raw_backend_selection(graspgen_bundle, runtime_dependencies):
    """A named deployment is the only way in; raw backend/device never reaches the session."""
    validated = load_inference_manifest(graspgen_bundle, GRASPGEN_DEPLOYMENT)

    with pytest.raises(ValueError, match="only a validated named deployment"):
        _plugin(validated, {"backend": "ascend"}, runtime_dependencies)


def test_closing_the_plugin_is_idempotent_and_releases_every_role(plugin, fake_acl):
    plugin.close()
    plugin.close()

    assert all(model.close_calls == 1 for model in FakeAclModel.instances.values())
    assert not plugin.runtime_status().ready
