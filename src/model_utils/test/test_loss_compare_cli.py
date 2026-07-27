from __future__ import annotations

import pytest

from model_utils import loss_compare_cli


def test_cli_selects_arbitrary_named_deployment(tmp_path):
    resolved = loss_compare_cli.resolve(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "--deployment",
            "my_compiled_target",
            "--policy_path",
            str(tmp_path / "bundle"),
            "--batch_path",
            str(tmp_path / "batches.json"),
            "--target_path",
            str(tmp_path / "target.json"),
        ]
    )

    assert resolved.args.deployment == "my_compiled_target"
    assert not hasattr(resolved.args, "device")
    assert not hasattr(resolved.args, "policy_type")


@pytest.mark.parametrize("removed_flag", ["--device", "--policy_type"])
def test_cli_rejects_removed_backend_and_policy_flags(tmp_path, removed_flag):
    with pytest.raises(SystemExit):
        loss_compare_cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                removed_flag,
                "cpu",
                "--policy_path",
                str(tmp_path / "bundle"),
                "--batch_path",
                str(tmp_path / "batches.json"),
                "--target_path",
                str(tmp_path / "target.json"),
            ]
        )


def test_diagnostic_paths_are_nonempty_cli_only_and_not_persisted(tmp_path):
    config_path = tmp_path / "loss.yaml"
    resolved = loss_compare_cli.resolve(
        [
            "--config",
            str(config_path),
            "--policy_path",
            str(tmp_path / "bundle"),
            "--batch_path",
            str(tmp_path / "batches.json"),
            "--target_path",
            str(tmp_path / "target.json"),
            "--metrics-json",
            str(tmp_path / "metrics.json"),
            "--schedule-override-path",
            str(tmp_path / "schedule.json"),
            "--curvature-log-path",
            str(tmp_path / "curvature.jsonl"),
        ]
    )

    loss_compare_cli.write_last(resolved)
    persisted = loss_compare_cli.load_config(str(config_path))["_last"]

    assert resolved.args.schedule_override_path.endswith("schedule.json")
    assert "metrics_json" not in persisted
    assert "schedule_override_path" not in persisted
    assert "curvature_log_path" not in persisted


@pytest.mark.parametrize("option", ["--metrics-json", "--schedule-override-path", "--curvature-log-path"])
def test_diagnostic_paths_reject_empty_strings(tmp_path, option):
    with pytest.raises(SystemExit, match="non-empty"):
        loss_compare_cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy_path",
                str(tmp_path / "bundle"),
                "--batch_path",
                str(tmp_path / "batches.json"),
                "--target_path",
                str(tmp_path / "target.json"),
                option,
                "   ",
            ]
        )


def test_tuning_diagnostics_cannot_generate_targets(tmp_path):
    with pytest.raises(SystemExit, match="compute-only"):
        loss_compare_cli.resolve(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--policy_path",
                str(tmp_path / "bundle"),
                "--batch_path",
                str(tmp_path / "batches.json"),
                "--target_path",
                str(tmp_path / "target.json"),
                "--schedule-override-path",
                str(tmp_path / "schedule.json"),
                "--generate-target",
            ]
        )
