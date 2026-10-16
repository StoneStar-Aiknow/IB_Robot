"""Per-suffix init-state path resolution and reshape tests.

Each LIBERO-plus perturbation suffix category gets an explicit test: _table_,
_tb_, _view_, _language_, _initstate_, _noise_, _light_ strip to the base
task's .pruned_init (preferred over .init); _add_ and _level route to the
libero_newobj/ subdirectory. hf-libero exact paths stay fail-closed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_libero.init_state_loader import (  # noqa: E402
    InitStateLoadError,
    load_task_init_states,
    resolve_init_states_path,
)


class _Task:
    def __init__(self, problem_folder: str, init_states_file: str):
        self.problem_folder = problem_folder
        self.init_states_file = init_states_file


def _make_files(root: Path, folder: str, names: list[str]) -> None:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        (d / name).write_bytes(b"\x00" * 4)


@pytest.fixture()
def init_tree(tmp_path: Path):
    """A miniature init_files tree with one base task and all suffix variants."""
    base = "pick_up_the_black_bowl_on_the_plate"
    _make_files(tmp_path, "libero_object", [f"{base}.pruned_init", f"{base}.init"])
    _make_files(
        tmp_path / "libero_newobj",
        "libero_object",
        [
            f"{base}_add_10.pruned_init",
            f"{base}_level1_sample1.pruned_init",
        ],
    )
    return tmp_path, base


def _get_libero_path(base_dir: Path):
    def fn(key: str) -> str:
        assert key == "init_states"
        return str(base_dir)

    return fn


class TestPerturbationSuffixResolution:
    @pytest.mark.parametrize("suffix", ["_table_11", "_tb_2", "_view_0_0_100_0_0"])
    def test_spatial_base_name_is_not_truncated_at_table_center(self, tmp_path, suffix) -> None:
        base = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
        _make_files(tmp_path, "libero_spatial", [f"{base}.pruned_init"])
        task = _Task("libero_spatial", f"{base}{suffix}.pruned_init")
        assert resolve_init_states_path(task, _get_libero_path(tmp_path), provider="libero-plus") == str(
            tmp_path / "libero_spatial" / f"{base}.pruned_init"
        )

    def test_table_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_table_1.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_tb_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_tb_2.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_view_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_view_0_0_100_0_0.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_language_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_language_48.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_initstate_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_initstate_3.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_noise_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_noise_8.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_light_suffix_strips_to_pruned_init(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_light_2.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_combined_suffix_strips_at_earliest_marker(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_view_0_0_100_0_0_initstate_0_noise_3.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_add_suffix_routes_to_libero_newobj(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_add_10.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_newobj" / "libero_object" / f"{base}_add_10.pruned_init")

    def test_level_suffix_routes_to_libero_newobj(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_level1_sample1.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="libero-plus")
        assert path == str(root / "libero_newobj" / "libero_object" / f"{base}_level1_sample1.pruned_init")


class TestPlanResolverTaskCount:
    """hf-libero must fail closed on unexpected task counts; libero-plus accepts expansion."""

    def test_hf_libero_rejects_more_tasks_than_expected(self) -> None:
        from benchmark_libero.plan_resolver import LiberoPlanRequest

        request = LiberoPlanRequest(
            suite="libero_object",
            task_order_index=0,
            tasks=None,
            episodes_per_task=2,
            seed=10000,
            max_steps=600,
            startup_timeout_sec=120.0,
            max_duration_sec=600.0,
            save_sim_states=True,
            video_enabled=True,
            write_native=True,
            write_canonical=True,
            provider="hf-libero",
        )

        class FakeProvider:
            def resolve_suite(self, suite: str, order: int):
                return tuple(f"task_{i}" for i in range(2518))  # way more than 10

        from benchmark_libero.plan_resolver import LiberoPlanResolver

        resolver = LiberoPlanResolver(FakeProvider())
        try:
            resolver.resolve(request)
            raise AssertionError("hf-libero with 2518 tasks should have raised")
        except ValueError as exc:
            assert "resolved 2518 tasks" in str(exc)
            assert "expected 10" in str(exc)

    def test_libero_plus_accepts_expanded_task_count(self) -> None:
        from benchmark_libero.plan_resolver import LiberoPlanRequest, LiberoPlanResolver

        request = LiberoPlanRequest(
            suite="libero_object",
            task_order_index=0,
            tasks=None,
            episodes_per_task=2,
            seed=10000,
            max_steps=600,
            startup_timeout_sec=120.0,
            max_duration_sec=600.0,
            save_sim_states=True,
            video_enabled=True,
            write_native=True,
            write_canonical=True,
            provider="libero-plus",
        )

        class FakeTask:
            def __init__(self, name: str):
                self.name = name
                self.prompt = "test"
                self.language = "test"
                self.problem_folder = "libero_object"
                self.bddl_file = f"{name}.bddl"
                self.init_states_file = f"{name}.pruned_init"
                self.init_state_count = 50

        class FakeProvider:
            def resolve_suite(self, suite: str, order: int):
                return [FakeTask(f"task_{i}") for i in range(2518)]

        resolver = LiberoPlanResolver(FakeProvider())
        plan = resolver.resolve(request)
        assert plan is not None
        assert any("LIBERO-plus expansion" in w for w in plan.warnings)

    def test_libero_plus_rejects_fewer_tasks_than_original(self) -> None:
        from benchmark_libero.plan_resolver import LiberoPlanRequest, LiberoPlanResolver

        request = LiberoPlanRequest(
            suite="libero_object",
            task_order_index=0,
            tasks=None,
            episodes_per_task=2,
            seed=10000,
            max_steps=600,
            startup_timeout_sec=120.0,
            max_duration_sec=600.0,
            save_sim_states=True,
            video_enabled=True,
            write_native=True,
            write_canonical=True,
            provider="libero-plus",
        )

        class FakeProvider:
            def resolve_suite(self, suite: str, order: int):
                return tuple(f"task_{i}" for i in range(5))  # fewer than 10

        resolver = LiberoPlanResolver(FakeProvider())
        try:
            resolver.resolve(request)
            raise AssertionError("libero-plus with 5 tasks should have raised")
        except ValueError as exc:
            assert "resolved 5 tasks" in str(exc)
            assert "expected at least 10" in str(exc)


class TestHfLiberoFailClosed:
    def test_exact_path_returned_directly(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="hf-libero")
        assert path == str(root / "libero_object" / f"{base}.pruned_init")

    def test_missing_file_returns_exact_path_for_downstream_error(self, init_tree) -> None:
        root, _ = init_tree
        task = _Task("libero_object", "nonexistent_task.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="hf-libero")
        assert path == str(root / "libero_object" / "nonexistent_task.pruned_init")
        assert not os.path.isfile(path)

    def test_no_cross_task_fallback(self, init_tree) -> None:
        root, _ = init_tree
        # A file that shares zero prefix with any existing task must not
        # resolve to a different task's init file.
        task = _Task("libero_object", "zzz_completely_different_task.pruned_init")
        path = resolve_init_states_path(task, _get_libero_path(root), provider="hf-libero")
        assert path == str(root / "libero_object" / "zzz_completely_different_task.pruned_init")
        assert not os.path.isfile(path)

    def test_missing_perturbation_does_not_use_existing_base_file(self, init_tree) -> None:
        root, base = init_tree
        task = _Task("libero_object", f"{base}_table_1.pruned_init")
        with pytest.raises(InitStateLoadError, match="does not exist"):
            load_task_init_states(task, _get_libero_path(root), provider="hf-libero")


def test_plus_missing_pruned_state_does_not_fall_back_to_unpruned(init_tree) -> None:
    root, base = init_tree
    (root / "libero_object" / f"{base}.pruned_init").unlink()
    task = _Task("libero_object", f"{base}_table_1.pruned_init")
    with pytest.raises(InitStateLoadError, match="does not exist"):
        load_task_init_states(task, _get_libero_path(root), provider="libero-plus")


def test_plus_missing_new_object_state_does_not_use_other_file(init_tree) -> None:
    root, base = init_tree
    name = f"{base}_add_10.pruned_init"
    (root / "libero_newobj" / "libero_object" / name).unlink()
    _make_files(root, "libero_object", [name])
    with pytest.raises(InitStateLoadError, match="libero_newobj"):
        load_task_init_states(_Task("libero_object", name), _get_libero_path(root), provider="libero-plus")


class TestAddLevelReshape:
    @pytest.mark.parametrize("provider", ["hf-libero", "libero-plus"])
    @pytest.mark.parametrize("suffix", ["_add_10", "_level1_sample1"])
    def test_normalization_is_provider_specific(self, init_tree, monkeypatch, provider, suffix) -> None:
        from benchmark_libero import init_state_loader

        root, base = init_tree
        flat = np.arange(110, dtype=np.float64)
        monkeypatch.setattr(init_state_loader, "load_trusted_init_states", lambda path: flat)
        task = _Task("libero_object", f"{base}{suffix}.pruned_init")
        states = load_task_init_states(task, _get_libero_path(root), provider=provider)
        assert states.shape == ((1, 110) if provider == "libero-plus" else (110,))

    @pytest.mark.parametrize("identity_mode", ["shared", "different", "missing"])
    @pytest.mark.parametrize("suffix", ["_add_10", "_level1_sample1"])
    def test_plan_reset_and_multi_lane_aggregation_agree(
        self, init_tree, tmp_path, monkeypatch, suffix, identity_mode
    ) -> None:
        import json

        from benchmark_libero import init_state_loader
        from benchmark_libero.adapter import LiberoAdapter
        from benchmark_libero.plan_provider import NativeLiberoPlanProvider
        from benchmark_libero.plan_resolver import LiberoPlanRequest, LiberoPlanResolver
        from benchmark_runtime.canonical_models import CanonicalEpisodeRecord
        from benchmark_runtime.canonical_report import CanonicalReportWriter, reduce_episode_records
        from benchmark_runtime.lane_aggregation import write_global_report
        from benchmark_runtime.models import ResetRequest
        from benchmark_runtime.plan import benchmark_plan_to_json, partition_plan

        root, base = init_tree
        task = SimpleNamespace(
            name=f"{base}{suffix}",
            language="pick up the bowl",
            problem_folder="libero_object",
            init_states_file=f"{base}{suffix}.pruned_init",
            bddl_file=f"{base}{suffix}.bddl",
        )
        suite = SimpleNamespace(
            get_num_tasks=lambda: 10,
            get_task=lambda index: SimpleNamespace(**{**vars(task), "name": f"{task.name}_{index}"}),
        )

        runtime = ModuleType("libero.libero")
        runtime.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {"libero_object": lambda **kwargs: suite})
        runtime.get_libero_path = lambda key: str(root)
        envs = ModuleType("libero.libero.envs")
        envs.OffScreenRenderEnv = lambda **kwargs: object()
        top = ModuleType("libero")
        top.libero = runtime
        for name, module in (("libero", top), ("libero.libero", runtime), ("libero.libero.envs", envs)):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(init_state_loader, "load_trusted_init_states", lambda path: np.arange(60))
        plan = LiberoPlanResolver(NativeLiberoPlanProvider(provider="libero-plus")).resolve(
            LiberoPlanRequest(
                suite="libero_object",
                task_order_index=0,
                tasks=(0,),
                episodes_per_task=2,
                seed=0,
                max_steps=10,
                startup_timeout_sec=10.0,
                max_duration_sec=10.0,
                save_sim_states=False,
                video_enabled=False,
                write_native=False,
                write_canonical=True,
                lane_count=2,
                provider="libero-plus",
            )
        )
        assert plan.tasks[0].init_state_count == 1
        adapter = LiberoAdapter()
        adapter._configured = True
        adapter._libero_provider = "libero-plus"
        adapter._plan = plan
        monkeypatch.setattr(adapter, "_resolve_native_task", lambda snapshot: (suite, task))
        monkeypatch.setattr(adapter, "_reset_active_environment", lambda seed, state: {"state": state})
        identity = {"packages": {"libero_provider": {"version": "libero-plus", "commit": "a" * 40}}}
        for lane in range(2):
            lane_plan = partition_plan(plan, lane, 2)
            manifest = {
                "run_id": "regression",
                "status": "running",
                "lane_index": lane,
                "lane_count": 2,
                "resolved_plan": json.loads(benchmark_plan_to_json(lane_plan)),
                "runtime_identity": identity,
            }
            if lane == 1 and identity_mode == "different":
                manifest["runtime_identity"] = {"packages": {"libero_provider": {"commit": "b" * 40}}}
            elif lane == 1 and identity_mode == "missing":
                del manifest["runtime_identity"]
            writer = CanonicalReportWriter(tmp_path / "run" / "lanes" / f"lane_{lane:03d}")
            writer.start_run(manifest)
            records = []
            for episode in range(2):
                reset = adapter.reset(ResetRequest(plan.suite, 0, 0, True, episode))
                assert reset.metadata["init_state_count"] == 1
                assert reset.metadata["actual_init_state_id"] == 0
                record = CanonicalEpisodeRecord(
                    run_id="regression",
                    suite=plan.suite,
                    task_id=0,
                    task_name=plan.tasks[0].name,
                    episode_index=episode,
                    seed=0,
                    native_result_available=True,
                    is_success=True,
                    requested_init_state_id=reset.metadata["requested_init_state_id"],
                    actual_init_state_id=reset.metadata["actual_init_state_id"],
                    metadata={"lane_index": lane},
                )
                writer.append_episode(record)
                records.append(record)
            summary = reduce_episode_records(
                records, run_id="regression", suite=plan.suite, planned_episodes=2, run_status="complete"
            )
            writer.finalize(summary, manifest={**manifest, "status": "complete"})
        summary = write_global_report(tmp_path / "run", run_id="regression", plan=plan, lane_count=2)
        assert summary.run_status == "complete"
        assert summary.counters.completed_native_episodes == 4
        assert summary.metadata["aggregation_errors"] == []
        manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
        if identity_mode == "shared":
            assert manifest["runtime_identity"] == identity
            assert manifest["lane_runtime_identities"] == {"lane_000": identity, "lane_001": identity}
        else:
            assert "runtime_identity" not in manifest
            assert manifest["lane_runtime_identities"]["lane_000"] == identity
            if identity_mode == "missing":
                assert "lane_001" not in manifest["lane_runtime_identities"]
            else:
                assert manifest["lane_runtime_identities"]["lane_001"] != identity
