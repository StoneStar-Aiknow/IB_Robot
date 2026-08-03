#!/usr/bin/env python3
"""speech_direction 模型校验。

对齐 perception_service/grounded_sam2_wrapper.py 的风格:节点启动时校验模型,
缺失则抛 FileNotFoundError 并提示运行下载脚本,节点直接退出(不降级保持运行)。

需要的模型(均不入库,放 models/ 下,通过 scripts/download_speech_direction_models.sh 下载):
  1. Silero VAD v5 ONNX    — models/voice_asr/silero-vad/silero_vad_v5.onnx(与 voice_asr 共用)
  2. FullSubNet 源码仓      — models/fullsubnet_repo/(git clone,需 model.py)
  3. FullSubNet ckpt        — models/fullsubnet/fullsubnet_best_model_58epochs.tar(~67MB)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# 工作区根目录(voice_asr_service 包向上 3 级到 IB_Robot/)
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_MODELS_ROOT = _WORKSPACE_ROOT / "models"

# Silero VAD v5(与 voice_asr_service 共用)
SILERO_VAD_REL = "voice_asr/silero-vad/silero_vad_v5.onnx"

# FullSubNet 源码仓(git clone 产物,只需 recipes/dns_interspeech_2020/fullsubnet/model.py)
FULLSUBNET_REPO_REL = "fullsubnet_repo"
FULLSUBNET_MODEL_REL = "recipes/dns_interspeech_2020/fullsubnet/model.py"
FULLSUBNET_REPO_URL = "https://github.com/Audio-WestlakeU/FullSubNet.git"

# FullSubNet ckpt(67MB,不入库)
# 官方下载源:Audio-WestlakeU/FullSubNet v0.2 release(文件名匹配 58epochs 预训练版本)
FULLSUBNET_CKPT_REL = "fullsubnet/fullsubnet_best_model_58epochs.tar"
FULLSUBNET_CKPT_URL = (
    "https://github.com/Audio-WestlakeU/FullSubNet/releases/download/v0.2/fullsubnet_best_model_58epochs.tar"
)


@dataclass(frozen=True)
class ModelAsset:
    """单个模型资产描述。"""

    name: str  # 资产名(展示用)
    path: Path  # 绝对路径
    required_for: str  # 用途说明

    def exists(self) -> bool:
        return self.path.is_file()


@dataclass(frozen=True)
class ModelStatus:
    """模型校验结果。"""

    silero_vad: ModelAsset
    fullsubnet_repo: ModelAsset
    fullsubnet_ckpt: ModelAsset

    def missing(self) -> list[ModelAsset]:
        """返回缺失的资产列表。"""
        return [a for a in (self.silero_vad, self.fullsubnet_repo, self.fullsubnet_ckpt) if not a.exists()]

    def all_present(self) -> bool:
        return not self.missing()


def resolve_assets(models_root: Path | None = None) -> ModelStatus:
    """解析所有模型资产路径。

    Args:
        models_root: 模型根目录(默认 <workspace>/models)

    Returns:
        ModelStatus: 各资产的存在性可通过 .exists() 检查
    """
    root = models_root or _DEFAULT_MODELS_ROOT
    return ModelStatus(
        silero_vad=ModelAsset(
            name="Silero VAD v5",
            path=root / SILERO_VAD_REL,
            required_for="人声门控(Silero VAD 推理)",
        ),
        fullsubnet_repo=ModelAsset(
            name="FullSubNet 源码模型入口",
            path=root / FULLSUBNET_REPO_REL / FULLSUBNET_MODEL_REL,
            required_for="语音增强(FullSubNet Model 定义)",
        ),
        fullsubnet_ckpt=ModelAsset(
            name="FullSubNet ckpt",
            path=root / FULLSUBNET_CKPT_REL,
            required_for="语音增强(模型权重,~67MB)",
        ),
    )


def resolve_configured_assets(
    silero_vad_path: str | Path,
    fullsubnet_repo_dir: str | Path,
    fullsubnet_ckpt_path: str | Path,
) -> ModelStatus:
    """按 ROS 配置给出的最终路径构造模型资产状态。"""
    return ModelStatus(
        silero_vad=ModelAsset(
            name="Silero VAD v5",
            path=Path(silero_vad_path),
            required_for="人声门控(Silero VAD 推理)",
        ),
        fullsubnet_repo=ModelAsset(
            name="FullSubNet 源码模型入口",
            path=Path(fullsubnet_repo_dir) / FULLSUBNET_MODEL_REL,
            required_for="语音增强(FullSubNet Model 定义)",
        ),
        fullsubnet_ckpt=ModelAsset(
            name="FullSubNet ckpt",
            path=Path(fullsubnet_ckpt_path),
            required_for="语音增强(模型权重,~67MB)",
        ),
    )


def _raise_missing(status: ModelStatus, location_summary: str) -> None:
    """统一报告缺失资产，确保默认布局与显式路径使用相同错误语义。"""
    missing = status.missing()
    if not missing:
        return

    # 逐项列出缺失(对齐 perception 的 "X not found: <path>" 风格)。
    detail = "\n".join(f"{asset.name} not found: {asset.path}" for asset in missing)
    raise FileNotFoundError(
        f"speech_direction 模型缺失(共 {len(missing)} 项):\n{detail}\n"
        "Run: ./scripts/download_speech_direction_models.sh\n"
        f"模型位置: {location_summary}"
    )


def require_configured_models(
    silero_vad_path: str | Path,
    fullsubnet_repo_dir: str | Path,
    fullsubnet_ckpt_path: str | Path,
) -> None:
    """校验 ROS 参数指定的三个最终模型资产路径。"""
    status = resolve_configured_assets(
        silero_vad_path,
        fullsubnet_repo_dir,
        fullsubnet_ckpt_path,
    )
    _raise_missing(status, "来自 speech_direction ROS 参数")


def require_models(models_root: Path | None = None) -> None:
    """校验默认下载布局,保留 CLI 与既有调用兼容性。

    与 perception_service/grounded_sam2_wrapper.py 一致:模型缺失时抛
    FileNotFoundError 并提示运行下载脚本,节点直接退出,不降级保持运行。

    Args:
        models_root: 模型根目录(默认 <workspace>/models)

    Raises:
        FileNotFoundError: 任一模型缺失,消息含缺失项路径与下载脚本提示
    """
    root = models_root or _DEFAULT_MODELS_ROOT
    _raise_missing(resolve_assets(models_root), str(root))


def main() -> int:
    """命令行入口:检查模型状态并打印指引。"""
    import argparse

    parser = argparse.ArgumentParser(description="Check speech_direction model assets.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=_DEFAULT_MODELS_ROOT,
        help=f"Models root directory (default: {_DEFAULT_MODELS_ROOT})",
    )
    args = parser.parse_args()

    status = resolve_assets(args.models_root)
    print(f"Models root: {args.models_root}")
    print()
    for a in (status.silero_vad, status.fullsubnet_repo, status.fullsubnet_ckpt):
        mark = "✓" if a.exists() else "✗"
        print(f"  [{mark}] {a.name}")
        print(f"       path: {a.path}")
        print(f"       use: {a.required_for}")
        print()

    if status.all_present():
        print("All models present.")
        return 0

    print("Missing models detected. Run:")
    print("  ./scripts/download_speech_direction_models.sh")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
