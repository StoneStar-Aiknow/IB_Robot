from pathlib import Path
from types import SimpleNamespace

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_legacy_supervised_script_does_not_exist():
    script_path = _REPOSITORY_ROOT / "scripts" / "test_banana_handeye_pick.py"

    assert not script_path.exists()


def test_supervised_pick_action_client_is_installed():
    package_dir = _REPOSITORY_ROOT / "src" / "manipulation_execution"

    assert (package_dir / "manipulation_execution" / "pick_action_client.py").exists()
    setup_source = (package_dir / "setup.py").read_text(encoding="utf-8")
    assert "pick_action_client = manipulation_execution.pick_action_client:main" in setup_source


def test_supervised_pick_action_client_uses_seconds_cli_names():
    from manipulation_execution.pick_action_client import build_parser

    args = build_parser().parse_args(
        [
            "--target-name",
            "banana",
            "--timeout-s",
            "230",
            "--ready-timeout-s",
            "30",
            "--goal-response-timeout-s",
            "10",
        ]
    )

    assert args.timeout_s == 230.0
    assert args.ready_timeout_s == 30.0
    assert args.goal_response_timeout_s == 10.0
    assert args.exact_task_id is False


def test_supervised_pick_task_id_is_unique_per_attempt():
    from manipulation_execution.pick_action_client import resolve_supervised_task_id

    assert resolve_supervised_task_id("pick-banana-pc-001", unique_suffix="attempt-a") == (
        "pick-banana-pc-001-attempt-a"
    )
    assert resolve_supervised_task_id("", unique_suffix="attempt-b") == "supervised-pick-attempt-b"


def test_supervised_pick_exact_task_id_is_explicit():
    from manipulation_execution.pick_action_client import resolve_supervised_task_id

    assert resolve_supervised_task_id(" pick-banana-pc-001 ", exact=True) == "pick-banana-pc-001"
    with pytest.raises(RuntimeError, match="requires a non-empty --task-id"):
        resolve_supervised_task_id("", exact=True)


def test_supervised_pick_timeout_leaves_root_budget_headroom():
    from manipulation_execution.pick_action_client import resolve_supervised_timeouts

    assert resolve_supervised_timeouts(230.0, 240.0) == (230.0, 240.0)


def test_supervised_pick_timeout_rejects_full_root_budget():
    from manipulation_execution.pick_action_client import resolve_supervised_timeouts

    with pytest.raises(RuntimeError, match="less than the Gateway task budget"):
        resolve_supervised_timeouts(240.0, 240.0)


def test_supervised_pick_retries_transient_readiness_query():
    from manipulation_execution.pick_action_client import PickActionClient

    response = object()
    wait_results = [RuntimeError("transient timeout"), response]
    requests = []
    warnings = []

    class FakeServiceClient:
        @staticmethod
        def wait_for_service(*, timeout_sec):
            assert timeout_sec > 0.0
            return True

        @staticmethod
        def call_async(request):
            requests.append(request)
            return object()

    def wait_future(_node, _future, timeout_sec, label):
        assert 0.0 < timeout_sec <= 5.0
        assert label == "/embodied/get_skill_gateway_status"
        result = wait_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    fake_node = SimpleNamespace(
        _wait_future=wait_future,
        get_logger=lambda: SimpleNamespace(warning=warnings.append),
    )
    actual = PickActionClient._call_readiness_service(
        fake_node,
        FakeServiceClient(),
        lambda: object(),
        30.0,
        "/embodied/get_skill_gateway_status",
    )

    assert actual is response
    assert len(requests) == 2
    assert len(warnings) == 1


def test_direct_place_action_client_does_not_exist():
    package_dir = _REPOSITORY_ROOT / "src" / "manipulation_execution"

    assert not (package_dir / "manipulation_execution" / "place_action_client.py").exists()
    setup_source = (package_dir / "setup.py").read_text(encoding="utf-8")
    assert "place_action_client" not in setup_source


def test_legacy_supervised_executor_does_not_exist():
    legacy_path = _REPOSITORY_ROOT / "scripts" / "test_banana_handeye_pick_legacy.py"

    assert not legacy_path.exists()


def test_production_executor_does_not_import_script_implementations():
    package_dir = _REPOSITORY_ROOT / "src" / "manipulation_execution" / "manipulation_execution"

    for module_path in package_dir.rglob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        assert "scripts.test_banana_handeye_pick" not in source, module_path
