"""The operator environment a GraspGen-X npu deployment needs at launch.

Only the Torch bridges are located this way: the CANN vendor packages live in
the CANN installation's own ``opp/vendors/`` and are found through its
``config.ini``. The bridges' location is owned by
torch_models.graspgenx_ascend_310p.custom_ops, which cannot be imported here
(its package pulls in torch_npu), so the builder forwards
IBROBOT_GRASPGENX_ASCEND_OPS instead of keeping a second copy of the path.

The other half of what these tests pin down is a *negative*:
ASCEND_CUSTOM_OPP_PATH must not be set for these operators. Setting it selects
CANN's external-install branch, which registers their operator-info, proto and
tiling stores but not the binary store, so GE recompiles each operator once per
process -- seconds per operator on a 310P, paid on the first grasp.
"""

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
        bridge = root / vendor / "torch_bridge.so"
        bridge.parent.mkdir(parents=True)
        bridge.write_bytes(b"")
    return root


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)


def test_a_built_operator_root_is_carried_into_the_launch(tmp_path, monkeypatch):
    root = _built_operators(tmp_path / "ops")
    monkeypatch.setenv(VARIABLE, str(root))

    environment = _graspgenx_npu_environment(_service())

    assert environment == {VARIABLE: str(root)}


def test_the_external_install_path_is_never_set(tmp_path, monkeypatch):
    """Setting it would cost seconds per operator on the first grasp.

    CANN only reuses the packaged binary kernels when it finds the packages
    through ``opp/vendors``; naming them in ASCEND_CUSTOM_OPP_PATH instead
    takes the branch that leaves the binary store unregistered and recompiles.
    """
    root = _built_operators(tmp_path / "ops")
    monkeypatch.setenv(VARIABLE, str(root))

    assert "ASCEND_CUSTOM_OPP_PATH" not in _graspgenx_npu_environment(_service())


def test_an_inherited_external_install_path_is_left_alone(tmp_path, monkeypatch):
    """It may belong to another package, so the builder neither extends nor clears it."""
    root = _built_operators(tmp_path / "ops")
    monkeypatch.setenv(VARIABLE, str(root))
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", "/opt/vendor/custom")

    assert _graspgenx_npu_environment(_service()) == {VARIABLE: str(root)}


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


def test_a_partly_built_root_carries_nothing(tmp_path, monkeypatch):
    """One bridge is not a usable deployment, and half an answer would hide that.

    Both packages are needed: the encoder calls GridEncode from one and
    BuildSubmMap and SubmConv3d from the other.
    """
    root = tmp_path / "ops"
    bridge = root / "graspgenx_subm" / "torch_bridge.so"
    bridge.parent.mkdir(parents=True)
    bridge.write_bytes(b"")
    monkeypatch.setenv(VARIABLE, str(root))

    assert _graspgenx_npu_environment(_service()) == {}


@pytest.mark.parametrize(
    ("model_type", "device"),
    [("graspgenx", "cuda"), ("graspgen", "npu"), ("pi05", "npu")],
)
def test_deployments_without_the_ascend_operators_get_nothing_extra(model_type, device, tmp_path, monkeypatch):
    monkeypatch.setenv(VARIABLE, str(_built_operators(tmp_path / "ops")))

    assert _graspgenx_npu_environment(_service(model_type, device)) == {}
