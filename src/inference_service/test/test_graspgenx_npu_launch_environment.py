"""The operator environment a GraspGen-X npu deployment needs at launch.

CANN reads ASCEND_CUSTOM_OPP_PATH once at initialisation, so a process cannot
set it for itself and the launch has to carry it. The location itself is owned
by torch_models.graspgenx_ascend_310p.custom_ops, which cannot be imported
here (its package pulls in torch_npu), so the builder forwards
IBROBOT_GRASPGENX_ASCEND_OPS instead of keeping a second copy of the path.
"""

import os
from types import SimpleNamespace

import pytest

from inference_service.launch_builders.perception_models import _graspgenx_npu_environment

VARIABLE = "IBROBOT_GRASPGENX_ASCEND_OPS"
VENDORS = ("graspgenx_subm", "graspgenx_grid")


def _service(model_type="graspgenx", device="npu"):
    return SimpleNamespace(
        validated_manifest=SimpleNamespace(
            manifest=SimpleNamespace(model=SimpleNamespace(model_type=model_type)),
            deployment=SimpleNamespace(device=device),
        )
    )


def _built_operators(root):
    for vendor in VENDORS:
        (root / "vendors" / vendor).mkdir(parents=True)
    return root


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)


def test_a_built_operator_root_is_carried_into_the_launch(tmp_path, monkeypatch):
    root = _built_operators(tmp_path / "ops")
    monkeypatch.setenv(VARIABLE, str(root))

    environment = _graspgenx_npu_environment(_service())

    assert environment[VARIABLE] == str(root)
    carried = environment["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)
    assert carried == [str(root / "vendors" / vendor) for vendor in VENDORS]


def test_an_unset_variable_carries_nothing(tmp_path, monkeypatch):
    """The builder keeps no path of its own, so there is nothing to fall back to.

    The operators are planted exactly where the superseded in-tree default
    would have looked, so this fails if a guessed path comes back: such a
    directory may hold operators built for another CANN release or Torch ABI.
    The loader in the served process reports the absence with the command that
    builds them.
    """
    _built_operators(tmp_path / "src" / "torch_models" / "ascend_ops" / "build")
    monkeypatch.setenv("WORKSPACE", str(tmp_path))

    assert _graspgenx_npu_environment(_service()) == {}


def test_a_root_without_built_operators_carries_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, str(tmp_path / "never-built"))

    assert _graspgenx_npu_environment(_service()) == {}


def test_a_partly_built_root_carries_only_what_is_there(tmp_path, monkeypatch):
    root = tmp_path / "ops"
    (root / "vendors" / "graspgenx_subm").mkdir(parents=True)
    monkeypatch.setenv(VARIABLE, str(root))

    environment = _graspgenx_npu_environment(_service())

    assert environment["ASCEND_CUSTOM_OPP_PATH"] == str(root / "vendors" / "graspgenx_subm")


def test_an_existing_operator_path_is_kept_after_ours(tmp_path, monkeypatch):
    root = _built_operators(tmp_path / "ops")
    monkeypatch.setenv(VARIABLE, str(root))
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/opt/vendor/custom")

    carried = _graspgenx_npu_environment(_service())["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep)

    assert carried[-1] == "/opt/vendor/custom"
    assert len(carried) == len(VENDORS) + 1


@pytest.mark.parametrize(
    ("model_type", "device"),
    [("graspgenx", "cuda"), ("graspgen", "npu"), ("pi05", "npu")],
)
def test_deployments_without_the_ascend_operators_get_nothing_extra(model_type, device, tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, str(_built_operators(tmp_path / "ops")))

    assert _graspgenx_npu_environment(_service(model_type, device)) == {}
