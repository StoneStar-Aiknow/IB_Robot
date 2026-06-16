"""Shared helpers for loading LeRobot policy config from on-disk policies.

The stock ``PreTrainedConfig.from_pretrained`` uses draccus to strictly decode
``config.json`` against the policy dataclass.  IB-Robot extends some policies'
``config.json`` with hardware-backend hints (Ascend OM / RKNN paths) that are
**not** part of the upstream dataclass and therefore make draccus raise
``DecodingError``.  Runtime deployment may also need a tensor device different
from the training-time ``device`` recorded in the model config.

This module materializes a temporary runtime-safe policy directory when needed:
it removes IB-Robot-only config keys and overrides ``device`` with the current
runtime tensor device, without modifying the source model directory.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_lerobot_policy_registry(policy_type: str | None = None) -> None:
    """Import policy config modules so draccus choice registry is populated.

    Board-side runtime may load ``PreTrainedConfig`` without having imported the
    concrete policy config modules first. In that case, ``type: act`` in
    ``config.json`` fails to decode because the registry entry was never
    registered. Keep this lazy and config-only to avoid importing heavy modeling
    dependencies during startup.
    """

    import importlib

    modules_by_type = {
        "act": "lerobot.policies.act.configuration_act",
        "diffusion": "lerobot.policies.diffusion.configuration_diffusion",
        "groot": "lerobot.policies.groot.configuration_groot",
        "multi_task_dit": "lerobot.policies.multi_task_dit.configuration_multi_task_dit",
        "pi0": "lerobot.policies.pi0.configuration_pi0",
        "pi05": "lerobot.policies.pi05.configuration_pi05",
        "sac": "lerobot.policies.sac.configuration_sac",
        "sarm": "lerobot.policies.sarm.configuration_sarm",
        "smolvla": "lerobot.policies.smolvla.configuration_smolvla",
        "tdmpc": "lerobot.policies.tdmpc.configuration_tdmpc",
        "vqbet": "lerobot.policies.vqbet.configuration_vqbet",
        "wall_x": "lerobot.policies.wall_x.configuration_wall_x",
        "xvla": "lerobot.policies.xvla.configuration_xvla",
    }
    if policy_type is not None and policy_type in modules_by_type:
        modules = (modules_by_type[policy_type],)
    else:
        modules = tuple(modules_by_type.values())
    for module in modules:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            # Some policy families are optional in trimmed deployments.
            continue


# Keys that IB-Robot writes into ``config.json`` but are not part of the
# upstream lerobot policy dataclasses.  Keep this list conservative; anything
# upstream may eventually adopt should be removed from here.
_IBROBOT_ONLY_KEYS: frozenset[str] = frozenset(
    {
        # RKNN backend hints
        "is_rknn_enabled",
        "rknn_model_path",
        # HMM (Houmo LQ50 / M50 xh2) backend hints
        "is_hmm_enabled",
        "hmm_model_path",
        # Ascend OM backend hints (written into config.json by the OM export
        # tooling; not part of the upstream lerobot policy dataclasses).
        "is_ascend_om_enabled",
        "om_vlm_model_path",
        "om_action_expert_model_path",
        "om_model_path",
    }
)


@dataclass
class RuntimePolicyPath:
    """Path to a runtime-safe policy directory with optional cleanup."""

    path: str
    tmpdir: Path | None = None

    def __enter__(self) -> str:
        return self.path

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.cleanup()

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    def cleanup(self) -> None:
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None


def override_runtime_policy_device(
    config: dict[str, Any],
    runtime_device: Any | None = None,
) -> dict[str, Any]:
    """Return ``config`` with runtime ``device`` set when requested."""
    if runtime_device is None:
        return config

    device_text = str(runtime_device)
    if config.get("device") == device_text:
        return config

    updated = dict(config)
    updated["device"] = device_text
    return updated


def _sanitize_lerobot_policy_config(
    raw: dict[str, Any],
    runtime_device: Any | None = None,
) -> dict[str, Any]:
    stripped = {k: v for k, v in raw.items() if k not in _IBROBOT_ONLY_KEYS}
    return override_runtime_policy_device(stripped, runtime_device)


def _read_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Policy config must be a JSON object: {path}")
    return raw


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def materialize_runtime_policy_path(
    policy_path: str,
    runtime_device: Any | None = None,
) -> RuntimePolicyPath:
    """Return a local policy path whose ``config.json`` matches runtime device.

    The source model directory is never modified.  When no sanitization is
    needed, the original path is returned.  When a sanitized config is needed,
    a tempdir mirrors the model directory via symlinks and replaces only
    ``config.json``.  Call ``cleanup`` or use this as a context manager to
    remove that tempdir after the upstream loader has consumed it.
    """
    src_dir = Path(policy_path)
    src_cfg = src_dir / "config.json"
    if not src_cfg.is_file():
        return RuntimePolicyPath(policy_path)

    raw = _read_json_object(src_cfg)
    sanitized = _sanitize_lerobot_policy_config(raw, runtime_device)
    if sanitized == raw:
        return RuntimePolicyPath(policy_path)

    tmpdir = Path(tempfile.mkdtemp(prefix="ibrobot_policy_"))
    try:
        for child in src_dir.iterdir():
            if child.name == "config.json":
                continue
            _link_or_copy(child, tmpdir / child.name)

        with open(tmpdir / "config.json", "w", encoding="utf-8") as f:
            json.dump(sanitized, f)
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    return RuntimePolicyPath(str(tmpdir), tmpdir)


def read_local_policy_config_device(policy_path: str | None) -> str | None:
    """Read the source policy ``config.json`` device for startup diagnostics."""
    if not policy_path:
        return None

    src_cfg = Path(policy_path) / "config.json"
    if not src_cfg.is_file():
        return None

    try:
        raw = _read_json_object(src_cfg)
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    device = raw.get("device")
    return str(device) if device is not None else None


def load_pretrained_policy_config(
    policy_path: str,
    runtime_device: Any | None = None,
) -> Any:
    """Load a ``PreTrainedConfig`` instance, tolerating IB-Robot custom keys.

    Falls back to plain ``PreTrainedConfig.from_pretrained`` when ``policy_path``
    is not a local directory (e.g. an HF hub repo id) or when local
    ``config.json`` already matches the runtime contract.
    """
    from lerobot.configs.policies import PreTrainedConfig

    policy_type = None
    src_cfg = Path(policy_path) / "config.json"
    if src_cfg.is_file():
        try:
            raw = _read_json_object(src_cfg)
            raw_type = raw.get("type")
            if raw_type is not None:
                policy_type = str(raw_type)
        except (OSError, ValueError, json.JSONDecodeError):
            policy_type = None

    _ensure_lerobot_policy_registry(policy_type)

    with materialize_runtime_policy_path(policy_path, runtime_device) as runtime_path:
        return PreTrainedConfig.from_pretrained(runtime_path)
