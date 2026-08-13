from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from inference_manifest import ArtifactBindings, DeviceLink, TensorBinding
from inference_service.codecs import build_execution_plan
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.pi05_schedule import PI05DenoisingSchedule
from inference_service.pipeline import (
    ExecutionControl,
    ExecutionError,
    PI05TopologyError,
    StageFrame,
    create_pi05_executor,
    derive_pi05_topology,
)


def _tensor(semantic: str, index: int, shape: tuple[int, ...]) -> TensorBinding:
    return TensorBinding(semantic=semantic, runtime_name=semantic, index=index, dtype="float32", shape=shape)


class _ResultAdapter:
    @staticmethod
    def adapt(frame: StageFrame) -> object:
        return frame.values["action"]

    @staticmethod
    def adapt_error(error: ExecutionError) -> object:
        raise RuntimeError(error.message)


class _FakeExecution:
    def __init__(self, calls: list[tuple[str, frozenset[str]]]) -> None:
        self.calls = calls

    def invoke(self, role: str, values: dict[str, object]) -> dict[str, np.ndarray]:
        self.calls.append((role, frozenset(values)))
        if role in {"vlm", "vision"}:
            return {"internal.condition": np.ones((1, 2), dtype=np.float32)}
        if role == "embedding":
            return {"internal.embedding": np.ones((1, 2), dtype=np.float32)}
        if role == "prefill":
            return {"internal.cache": np.ones((1, 2), dtype=np.float32)}
        if role == "action_in_proj":
            return {"internal.projected": np.asarray(values["noise"], dtype=np.float32)}
        if role == "time_mlp":
            return {"internal.time": np.full((1,), values["time"], dtype=np.float32)}
        if role == "decode":
            assert "internal.cache" not in values
            return {"internal.decoded": np.asarray(values["internal.projected"], dtype=np.float32)}
        source = values.get("noise", values.get("internal.decoded"))
        return {"action": np.full_like(np.asarray(source), 2.0)}


class _FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, frozenset[str]]] = []

    @contextmanager
    def execution(self, request: NamedTensorRequest):
        yield _FakeExecution(self.calls)


def _two_role_plan(*, device_link: bool = False):
    shape = (1, 2, 3)
    bindings = {
        "vlm": ArtifactBindings(
            inputs=(_tensor("observation.state", 0, (1, 2)),), outputs=(_tensor("internal.condition", 0, (1, 2)),)
        ),
        "action_expert": ArtifactBindings(
            inputs=(
                _tensor("internal.condition", 0, (1, 2)),
                _tensor("noise", 1, shape),
                _tensor("time", 2, (1,)),
            ),
            outputs=(_tensor("action", 0, shape),),
        ),
    }
    links = (
        (
            DeviceLink(
                semantic="internal.condition",
                producer="vlm",
                consumer="action_expert",
                transport="device_pointer",
                owner="producer",
            ),
        )
        if device_link
        else ()
    )
    return build_execution_plan(("vlm", "action_expert"), bindings, links)


def _modular_plan():
    shape = (1, 2, 3)
    bindings = {
        "vision": ArtifactBindings(
            inputs=(_tensor("observation.state", 0, (1, 2)),), outputs=(_tensor("internal.condition", 0, (1, 2)),)
        ),
        "embedding": ArtifactBindings(
            inputs=(_tensor("internal.condition", 0, (1, 2)),), outputs=(_tensor("internal.embedding", 0, (1, 2)),)
        ),
        "prefill": ArtifactBindings(
            inputs=(_tensor("internal.embedding", 0, (1, 2)),), outputs=(_tensor("internal.cache", 0, (1, 2)),)
        ),
        "action_in_proj": ArtifactBindings(
            inputs=(_tensor("noise", 0, shape),), outputs=(_tensor("internal.projected", 0, shape),)
        ),
        "time_mlp": ArtifactBindings(inputs=(_tensor("time", 0, (1,)),), outputs=(_tensor("internal.time", 0, (1,)),)),
        "decode": ArtifactBindings(
            inputs=(
                _tensor("internal.cache", 0, (1, 2)),
                _tensor("internal.projected", 1, shape),
                _tensor("internal.time", 2, (1,)),
            ),
            outputs=(_tensor("internal.decoded", 0, shape),),
        ),
        "action_out_proj": ArtifactBindings(
            inputs=(_tensor("internal.decoded", 0, shape),), outputs=(_tensor("action", 0, shape),)
        ),
    }
    cache_link = DeviceLink(
        semantic="internal.cache",
        producer="prefill",
        consumer="decode",
        transport="device_pointer",
        owner="producer",
    )
    return build_execution_plan(tuple(bindings), bindings, (cache_link,))


@pytest.mark.parametrize("plan", [_two_role_plan(), _two_role_plan(device_link=True), _modular_plan()])
def test_pi05_executor_runs_validated_topologies_with_shared_euler_update(plan):
    session = _FakeSession()
    schedule = PI05DenoisingSchedule(name="test", timesteps=(1.0, 0.6, 0.0))
    executor = create_pi05_executor(plan, session, schedule, _ResultAdapter())
    request = NamedTensorRequest(
        "pi05",
        {"observation.state": np.ones((1, 2), dtype=np.float32), "noise": np.ones((1, 2, 3), dtype=np.float32)},
    )

    result = executor.execute(request, deadline=None, control=ExecutionControl("pi05"))

    np.testing.assert_allclose(result, -1.0)
    loop_roles = derive_pi05_topology(plan).loop_roles
    assert [role for role, _values in session.calls].count(loop_roles[0]) == 2
    for role, values in session.calls:
        if role in loop_roles:
            assert "internal.condition" not in values or not plan.device_links


def test_pi05_topology_rejects_unknown_role_layout():
    bindings = {
        "encoder": ArtifactBindings(
            inputs=(_tensor("observation.state", 0, (1, 2)),), outputs=(_tensor("action", 0, (1, 2, 3)),)
        ),
    }

    with pytest.raises(PI05TopologyError, match="topology must be"):
        derive_pi05_topology(build_execution_plan(("encoder",), bindings))


def test_pi05_executor_expands_modular_embedding_and_time_prep_stages():
    plan = _modular_plan()
    session = _FakeSession()
    schedule = PI05DenoisingSchedule(name="test", timesteps=(1.0, 0.6, 0.0))

    embedding_calls: list[frozenset[str]] = []
    time_prep_calls: list[frozenset[str]] = []

    def embedding_stage_op(values):
        embedding_calls.append(frozenset(values))
        return {"internal.embedding": np.ones((1, 2), dtype=np.float32)}

    def time_prep_op(values):
        time_prep_calls.append(frozenset(values))
        return {"time": np.ones((1,), dtype=np.float32)}

    from inference_service.pipeline.stages import HostComputeStage, HostRoleStage

    embedding_stage = HostRoleStage(role="embedding", operation=embedding_stage_op)
    time_prep_stage = HostComputeStage(operation=time_prep_op)

    executor = create_pi05_executor(
        plan,
        session,
        schedule,
        _ResultAdapter(),
        embedding_stage=embedding_stage,
        time_prep_stage=time_prep_stage,
    )

    topology = derive_pi05_topology(plan)
    assert topology.pre_loop_roles == ("vision", "embedding", "prefill")
    assert topology.loop_roles == ("action_in_proj", "time_mlp", "decode", "action_out_proj")

    iterative = next(stage for stage in executor._stages if hasattr(stage, "body"))
    body_role_names = []
    for stage in iterative.body:
        if hasattr(stage, "role"):
            body_role_names.append(stage.role)
        else:
            body_role_names.append(getattr(stage, "operation", None).__name__ or "host_compute")
    assert "time_mlp" in body_role_names
    assert body_role_names.index("time_mlp") > 0

    request = NamedTensorRequest(
        "pi05",
        {"observation.state": np.ones((1, 2), dtype=np.float32), "noise": np.ones((1, 2, 3), dtype=np.float32)},
    )
    executor.execute(request, deadline=None, control=ExecutionControl("pi05"))

    assert len(embedding_calls) == 1
    assert len(time_prep_calls) == 2


def test_pi05_executor_time_prep_stage_replaces_scalar_timestep():
    plan = _modular_plan()
    session = _FakeSession()
    schedule = PI05DenoisingSchedule(name="test", timesteps=(1.0, 0.6, 0.0))
    observed: list[object] = []

    def time_prep_op(values):
        observed.append(values["time"])
        return {"time": np.ones((1,), dtype=np.float32)}

    from inference_service.pipeline.stages import HostComputeStage

    time_prep_stage = HostComputeStage(operation=time_prep_op)
    executor = create_pi05_executor(
        plan,
        session,
        schedule,
        _ResultAdapter(),
        time_prep_stage=time_prep_stage,
    )

    request = NamedTensorRequest(
        "pi05",
        {"observation.state": np.ones((1, 2), dtype=np.float32), "noise": np.ones((1, 2, 3), dtype=np.float32)},
    )
    executor.execute(request, deadline=None, control=ExecutionControl("pi05"))

    assert len(observed) == 2
    for value in observed:
        assert isinstance(value, np.ndarray | np.floating)
