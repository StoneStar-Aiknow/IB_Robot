#!/usr/bin/env python3
"""校验 speech_direction 的 310P 模型资产（纯校验，不下载）。

走两条校验：
1. 标准 inference_manifest.json —— 用 load_inference_manifest_metadata 校验
   bundle 结构、deployment bindings/execution 与 semantic_identity，不校验
   文件存在。
2. manifest deployment artifacts —— 逐资产校验文件存在性和 SHA-256，确保从 NAS
   手动获取的 OM/FB/SB/manifest 没有损坏或被篡改。

资产不在本仓库管理，需从 NAS 手动获取后放入 <models-root> 对应路径；
缺失的资产会打印提示并跳过（不终止），已存在的资产校验不通过则报错。

Ubuntu 环境依赖（Silero ONNX、FullSubNet 源码仓与 cumulative checkpoint）
请使用 scripts/download_speech_direction_models.sh 下载，不在本脚本范围。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# 脚本从源码树运行时，把两个包源码根加入路径：inference_manifest（标准加载入口）
# 与 voice_asr_service（契约 SSOT STATEFUL_FULLSUBNET_CONTRACT）。
_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_MANIFEST_ROOT = _WORKSPACE_ROOT / "src" / "inference_manifest"
_PKG_ROOT = _WORKSPACE_ROOT / "src" / "voice_asr_service"
for _root in (_INFERENCE_MANIFEST_ROOT, _PKG_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from inference_manifest import load_inference_manifest_metadata  # noqa: E402

# 标准 bundle 的 manifest 与资产清单：models/voice_asr/ 下的
# inference_manifest.json（bindings、runtime profile 与 artifact SHA-256）。
_DEFAULT_MANIFEST_DIR = _WORKSPACE_ROOT / "models" / "voice_asr"

# 支持的 deployment：与 inference_manifest.json 的 deployments 字典 key 对齐。
_DEPLOYMENTS = ("ascend_310p_silero", "ascend_310p_fullsubnet")

# 资产来源提示：310P OM/FB/SB/manifest 不在本仓库管理，需从 NAS 手动获取。
_NAS_HINT = "请从 NAS 手动获取该资产后放入上述路径"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    """校验通过静默返回，不通过抛异常（用于已存在的资产）。"""
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    actual_sha = _sha256(path)
    if actual_sha != expected:
        raise ValueError(f"{path}: sha256={actual_sha},期望={expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=_DEFAULT_MANIFEST_DIR,
        help="含 inference_manifest.json 的 bundle 目录",
    )
    parser.add_argument(
        "--deployment", choices=_DEPLOYMENTS, default=None, help="只校验该 deployment 的资产（默认校验全部 deployment）"
    )
    args = parser.parse_args()

    # 1. 标准入口校验 inference_manifest.json 的结构、bindings、semantic_identity。
    #    用 metadata 模式：只校验语义与身份，不要求 artifacts 文件存在
    #    （NAS 资产未入库，文件存在性由下方 artifact 校验）。
    manifest_dir = args.manifest_dir.resolve()
    loaded = []
    for dep_name in _DEPLOYMENTS:
        if args.deployment and dep_name != args.deployment:
            continue
        vm = load_inference_manifest_metadata(manifest_dir, dep_name)
        loaded.append((dep_name, vm))
        print(f"[manifest] {dep_name} 结构校验 OK (fingerprint={vm.fingerprint[:16]}...)")

    # 2. 逐 deployment artifact 校验文件存在性和 SHA-256。
    assets = {}
    for _, vm in loaded:
        for role, artifact in vm.deployment.artifacts.items():
            if artifact.sha256 is None:
                raise ValueError(f"{vm.deployment_name}/{role} is missing artifact sha256")
            assets[artifact.path] = artifact.sha256
    missing = False
    for relative_path, expected_sha in sorted(assets.items()):
        target = manifest_dir / relative_path
        desc = str(relative_path)
        if not target.is_file():
            # 缺失资产不报错终止，只提示来源，让其余已存在的资产继续校验。
            print(f"[missing] {desc}: {target}")
            print(f"           {_NAS_HINT}")
            missing = True
            continue
        _verify(target, expected_sha)
        print(f"[ok] {desc} SHA-256 OK: {target}")

    if missing:
        print("\n存在缺失资产，请从 NAS 手动获取后重新校验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
