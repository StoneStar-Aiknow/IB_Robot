"""FullSubNet stateful executor 与公共 Host 增强器装配。"""

from __future__ import annotations

from contextlib import suppress

from .fullsubnet_stateful import StatefulFullSubNetEnhancer


def build_stateful_fullsubnet(
    *,
    backend: str,
    checkpoint_path: str = "",
    manifest_path: str = "",
    fb_om_path: str = "",
    sb_om_path: str = "",
    device: str = "cuda",
    device_id: int = 0,
    timing_enabled: bool = False,
    initialize_backend: bool = True,
    executor=None,
) -> StatefulFullSubNetEnhancer:
    """仅在此选择模型执行器；两个平台随后共用同一 Host 算法实现。"""
    owned_executor = executor is None
    try:
        if executor is not None:
            pass
        elif backend == "ascend":
            from .fullsubnet_stateful_acl import StatefulAclFullSubNetRunner

            executor = StatefulAclFullSubNetRunner(
                fb_om_path,
                sb_om_path,
                device_id=device_id,
                timing_enabled=timing_enabled,
            )
        elif backend in {"stateful_torch_cuda", "stateful_torch_cpu"}:
            from .fullsubnet_stateful_torch import StatefulTorchFullSubNetExecutor

            requested_device = "cpu" if backend.endswith("_cpu") else device
            if backend == "stateful_torch_cuda" and requested_device != "cuda":
                raise ValueError("stateful_torch_cuda 必须配置 fullsubnet_device=cuda")
            executor = StatefulTorchFullSubNetExecutor(
                checkpoint_path,
                manifest_path,
                device=requested_device,
                timing_enabled=timing_enabled,
            )
        else:
            raise ValueError(f"不支持的 stateful FullSubNet backend: {backend}")
        return StatefulFullSubNetEnhancer(
            executor,
            manifest_path=manifest_path,
            timing_enabled=timing_enabled,
            initialize_backend=initialize_backend,
        )
    except Exception:
        if owned_executor and executor is not None:
            with suppress(Exception):
                executor.close()
        raise


__all__ = ["build_stateful_fullsubnet"]
