#!/usr/bin/env python3
"""校验 speech_direction 的 310P 模型资产（纯校验，不下载）。

走两条校验：
1. 标准 inference_manifest.json —— 用 load_inference_manifest_metadata 校验
   bundle 结构、deployment bindings/execution 与 semantic_identity，不校验
   文件存在（assets/adapter.json 承载 sha256/size，标准 schema 不收留这些字段）。
2. assets/adapter.json —— 逐资产校验文件存在性、大小和 SHA-256，确保从 NAS
   手动获取的 OM/FB/SB/manifest 没有损坏或被篡改。

资产不在本仓库管理，需从 NAS 手动获取后放入 <models-root> 对应路径；
缺失的资产会打印提示并跳过（不终止），已存在的资产校验不通过则报错。

Ubuntu 环境依赖（Silero ONNX、FullSubNet 源码仓与 cumulative checkpoint）
请使用 scripts/download_speech_direction_models.sh 下载，不在本脚本范围。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
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
from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful_executor import (  # noqa: E402
    STATEFUL_FULLSUBNET_CONTRACT,
)

# 标准 manifest 与资产清单的位置：config/ 下 inference_manifest.json（结构）+
# assets/adapter.json（sha256/size + 算法契约）。
_CONFIG_DIR = _WORKSPACE_ROOT / "src" / "voice_asr_service" / "config"
_DEFAULT_MANIFEST_DIR = _CONFIG_DIR
_DEFAULT_ADAPTER = _CONFIG_DIR / "assets" / "adapter.json"

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


def _verify(path: Path, asset: dict) -> None:
    """校验通过静默返回，不通过抛异常（用于已存在的资产）。"""
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    expected_size = int(asset["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path}: size={actual_size},期望={expected_size}")
    actual_sha = _sha256(path)
    if actual_sha != asset["sha256"]:
        raise ValueError(f"{path}: sha256={actual_sha},期望={asset['sha256']}")


def _verify_adapter_contract(adapter: dict, adapter_path: Path) -> None:
    """adapter.json 的 algorithm_contract 必须与 Python SSOT 逐项一致。

    标准 inference_manifest.json 不收留 algorithm_contract（schema forbid extra），
    故算法契约存放在 assets/adapter.json，仍由 STATEFUL_FULLSUBNET_CONTRACT 做 SSOT
    校验，避免 Python/JSON 两处各写一份。
    """
    contract = adapter.get("algorithm_contract", {})
    expected = asdict(STATEFUL_FULLSUBNET_CONTRACT)
    for key, want in expected.items():
        # Python 多出的实现细节字段（batch/num_freqs/sb_features）不强制 adapter 声明。
        if key in contract and contract.get(key) != want:
            raise ValueError(f"{adapter_path}: algorithm_contract.{key}={contract.get(key)!r},期望={want!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=_WORKSPACE_ROOT / "models")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=_DEFAULT_MANIFEST_DIR,
        help="含 inference_manifest.json + assets/adapter.json 的目录",
    )
    parser.add_argument(
        "--adapter", type=Path, default=None, help="adapter.json 路径（默认 <manifest-dir>/assets/adapter.json）"
    )
    parser.add_argument(
        "--deployment", choices=_DEPLOYMENTS, default=None, help="只校验该 deployment 的资产（默认校验全部 deployment）"
    )
    args = parser.parse_args()

    # 1. 标准入口校验 inference_manifest.json 的结构、bindings、semantic_identity。
    #    用 metadata 模式：只校验语义与身份，不要求 artifacts 文件存在
    #    （NAS 资产未入库，文件存在性由下方 adapter.json 逐资产校验）。
    manifest_dir = args.manifest_dir.resolve()
    loaded = []
    for dep_name in _DEPLOYMENTS:
        if args.deployment and dep_name != args.deployment:
            continue
        vm = load_inference_manifest_metadata(manifest_dir, dep_name)
        loaded.append((dep_name, vm))
        print(f"[manifest] {dep_name} 结构校验 OK (fingerprint={vm.fingerprint[:16]}...)")

    # 2. adapter.json 承载 sha256/size + 算法契约（标准 schema 不收留这些字段）。
    adapter_path = args.adapter or (manifest_dir / "assets" / "adapter.json")
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    _verify_adapter_contract(adapter, adapter_path)
    print(f"[adapter] {adapter_path} 算法契约 SSOT 校验 OK")

    # 3. 逐资产校验文件存在性、大小、SHA-256（NAS 获取的资产完整性）。
    assets = adapter.get("assets", [])
    missing = False
    for asset in assets:
        # 兼容两种 install_path：旧 speech_direction_models.json 的 models-root 相对路径，
        # 与 inference_manifest.json 的 manifest_dir 相对路径。优先按 install_path 找。
        target = args.models_root / asset["install_path"]
        desc = asset["logical_name"]
        if not target.is_file():
            # 缺失资产不报错终止，只提示来源，让其余已存在的资产继续校验。
            print(f"[missing] {desc}: {target}")
            print(f"           {_NAS_HINT}")
            missing = True
            continue
        _verify(target, asset)
        print(f"[ok] {desc} size+SHA-256 OK: {target}")

    if missing:
        print("\n存在缺失资产，请从 NAS 手动获取后重新校验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
