"""SDK owners keep failed cleanup reachable after partial construction."""

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("backend", ["hmm", "rknn"])
@pytest.mark.parametrize("construction_fails", [False, True])
@pytest.mark.parametrize("not_started", [False, True])
def test_sdk_release_failure_retains_owner_for_retry(backend, construction_fails, not_started):
    from inference_service.backends.hmm.backend import HMMModule
    from inference_service.backends.rknn.runtime import RKNNSession

    releases = []
    cleanup = []

    class SDK:
        def release(self):
            releases.append("release")
            if len(releases) == 1:
                error = RuntimeError("release pending")
                if not_started:
                    from inference_service.unified_runtime import OutcomeEvidence

                    error.evidence = OutcomeEvidence.not_started("release")
                raise error

        def load_rknn(self, _path):
            return 0

        def init_runtime(self, **_kwargs):
            if construction_fails:
                raise RuntimeError("initialization failed")
            return 0

        def get_num_inputs(self):
            if construction_fails:
                raise RuntimeError("initialization failed")
            return 0

        def get_num_outputs(self):
            return 0

    def construct():
        owner = SimpleNamespace(defer=cleanup.append)
        if backend == "hmm":
            return HMMModule(
                SimpleNamespace(load=lambda *_: SDK()),
                "model",
                Path("model.hmm"),
                SimpleNamespace(inputs=(), outputs=()),
                option=None,
                cleanup=owner,
            )
        return RKNNSession(SDK, "model", Path("model.rknn"), target=None, core_mask=0, data_format=None, cleanup=owner)

    if construction_fails:
        with pytest.raises(RuntimeError, match="initialization failed"):
            construct()
    else:
        construct()
    assert len(cleanup) == 1
    close = cleanup[0]
    with pytest.raises(RuntimeError, match="release pending"):
        close()
    if not_started:
        close()
        close()
        assert releases == ["release", "release"]
    else:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="process restart required") as caught:
                close()
            assert caught.value.close_pending
        assert releases == ["release"]


@pytest.mark.parametrize("backend", ["hmm", "rknn"])
def test_default_constructor_preserves_load_error_and_pending_cleanup_owner(backend):
    from inference_service.backends.hmm.backend import HMMModule
    from inference_service.backends.rknn.runtime import RKNNSession

    original = ValueError("initialization failed")
    releases = []

    class SDK:
        def release(self):
            releases.append("release")
            if len(releases) == 1:
                raise RuntimeError("release pending")

        def load_rknn(self, _path):
            return 0

        def init_runtime(self, **_kwargs):
            raise original

        def get_num_inputs(self):
            raise original

    with pytest.raises(ValueError) as caught:
        if backend == "hmm":
            HMMModule(
                SimpleNamespace(load=lambda *_: SDK()), "model", Path("model.hmm"), SimpleNamespace(), option=None
            )
        else:
            RKNNSession(SDK, "model", Path("model.rknn"), target=None, core_mask=0, data_format=None)
    assert caught.value is original and caught.value.close_pending
    for _ in range(2):
        with pytest.raises(RuntimeError, match="process restart required") as pending:
            caught.value.cleanup_owner.close()
        assert pending.value.close_pending
    assert releases == ["release"]


@pytest.mark.parametrize("failure_at", ["sync", "destroy"])
def test_default_acl_stream_pool_constructor_retains_failed_rollback_owner(failure_at):
    from inference_service.backends.ascend.acl_runtime import AclPriorityStreamPool

    original = ValueError("stream creation failed")
    blocked, released = [True], []

    def create(priority, _flags):
        if priority == 2:
            raise original
        return priority, 0

    def destroy(stream):
        if blocked[0] and failure_at == "destroy":
            return 1
        released.append(stream)
        return 0

    lease = SimpleNamespace(
        bind_current_thread=lambda: None,
        acl=SimpleNamespace(
            rt=SimpleNamespace(
                device_get_stream_priority_range=lambda: (7, 0, 0),
                create_stream_with_config=create,
                synchronize_stream=lambda _: int(blocked[0] and failure_at == "sync"),
                destroy_stream=destroy,
            )
        ),
    )
    with pytest.raises(ValueError) as caught:
        AclPriorityStreamPool.create(lease)
    assert caught.value is original and caught.value.close_pending
    assert caught.value.rollback_errors
    blocked[0] = False
    caught.value.cleanup_owner.close()
    caught.value.cleanup_owner.close()
    assert released == [1, 0]
