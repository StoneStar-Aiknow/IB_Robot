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
