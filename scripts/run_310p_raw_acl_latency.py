#!/usr/bin/env python3
"""310P raw_acl 端到端时延测试：用生产链路的 StatefulAclFullSubNetRunner + Host 编排。

走真实生产路径：inference_service.backends.ascend.AclModel.execute_bank
（acl.mdl.execute C 扩展）+ dataset bank dual hidden/cell 回环 + Host STFT/norm/OLA。

用法（310P 上）：
    source /opt/ros/humble/setup.bash
    source <repo>/install/setup.bash
    cd <repo>
    python3 scripts/run_310p_raw_acl_latency.py --wav <raw6ch.wav> [--fb-om ... --sb-om ... --manifest ...]

--wav 必传（无默认）：raw6ch 麦克风阵列录音，可用节点跑诊断生成（见 diagnostics）。
模型默认相对仓库根 models/fullsubnet/（mixed16 OM + 218epochs manifest）。
预算 32ms/block（512 samples = 2 tick）。
对照：test_offline_regression.py 内置时延断言（Ubuntu CUDA 侧 Host 编排 baseline）。

指标：每个 512-sample block（=2 tick=32ms 预算）的端到端处理时延 p50/p95/p99，
分 OM 推理（FB+SB）与 Host 编排（stft/fb_prepare/sb_prepare/postprocess）两段。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful import (
    StatefulFullSubNetEnhancer,
)
from voice_asr_service.speech_direction.enhancement.fullsubnet_stateful_acl import (
    StatefulAclFullSubNetRunner,
)

INPUT_SAMPLES = 512

# 仓库根（scripts/ 上两级），模型默认路径相对此根，不依赖具体部署位置。
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _REPO_ROOT / "models" / "fullsubnet"


def _stat(values: list[float]) -> dict:
    if not values:
        return {}
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # wav 必传：raw6ch 录音是运行时产物，不应硬编码默认路径
    parser.add_argument("--wav", required=True, help="raw6ch 麦克风阵列录音 wav 路径")
    # 模型默认相对仓库根 models/fullsubnet/，mixed16（310P 实测与 fp16 差异很小）
    parser.add_argument(
        "--fb-om",
        default=str(_MODEL_DIR / "fullsubnet_cum_stateful_fb_b4_t2_mixed16.om"),
    )
    parser.add_argument(
        "--sb-om",
        default=str(_MODEL_DIR / "fullsubnet_cum_stateful_sb_b4_t2_mixed16.om"),
    )
    parser.add_argument(
        "--manifest",
        default=str(_MODEL_DIR / "cum_fullsubnet_best_model_218epochs.manifest.json"),
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5, help="预热 block 数")
    parser.add_argument("--channel-indices", default="1,2,3,4")
    args = parser.parse_args()

    import soundfile as sf

    audio, sr = sf.read(args.wav, dtype="float32")
    ch_idx = [int(x) for x in args.channel_indices.split(",")]
    audio4 = audio[:, ch_idx]
    print(f"WAV: {args.wav} {audio.shape} -> ch {ch_idx} = {audio4.shape}, sr={sr}")

    runner = StatefulAclFullSubNetRunner(
        fb_om_path=args.fb_om,
        sb_om_path=args.sb_om,
        device_id=args.device_id,
        acl_config_path="",
        timing_enabled=True,
    )
    enhancer = StatefulFullSubNetEnhancer(
        runner,
        manifest_path=args.manifest,
        timing_enabled=True,
    )
    print(f"FB OM: {args.fb_om}")
    print(f"SB OM: {args.sb_om}")
    print(f"manifest: {args.manifest}")
    print("实时预算: 32ms/block (512 samples = 2x16ms tick @ 16kHz)")
    print(f"预热: {args.warmup} blocks")
    print()

    n_blocks = audio4.shape[0] // INPUT_SAMPLES
    blocks = [audio4[i * INPUT_SAMPLES : (i + 1) * INPUT_SAMPLES] for i in range(n_blocks)]
    print(f"总 block 数: {n_blocks}")

    # enhancer.last_timing_ms 含完整分段
    keys = (
        "stft_ms",
        "fb_prepare_ms",
        "fb_infer_ms",
        "sb_prepare_ms",
        "sb_infer_ms",
        "postprocess_ms",
        "fullsubnet_total_ms",
    )
    timings: dict[str, list[float]] = {k: [] for k in keys}
    output_count = 0
    for block in blocks:
        result = enhancer.process_4ch(block)
        # last_timing_ms 是 enhancer 的属性（dict），仅 result 非 None 时（推理完成）才填
        if result is not None:
            output_count += 1
            if output_count > args.warmup:
                t = enhancer.last_timing_ms
                if isinstance(t, dict):
                    for k in keys:
                        timings[k].append(float(t.get(k, 0.0)))

    print()
    print("=== raw_acl 端到端分段时延（预热后，单位 ms）===")
    for k in keys:
        s = _stat(timings[k])
        if s:
            print(
                f"  {k:22s}: mean={s['mean']:7.3f}  p50={s['p50']:7.3f}  "
                f"p95={s['p95']:7.3f}  p99={s['p99']:7.3f}  max={s['max']:7.3f}"
            )

    # 汇总：OM 推理 vs Host 编排 vs 端到端
    om = [a + b for a, b in zip(timings["fb_infer_ms"], timings["sb_infer_ms"], strict=False)]
    host = [
        a + b + c + d
        for a, b, c, d in zip(
            timings["stft_ms"],
            timings["fb_prepare_ms"],
            timings["sb_prepare_ms"],
            timings["postprocess_ms"],
            strict=False,
        )
    ]
    print()
    print("=== 汇总 ===")
    for label, vals in (("OM(fb+sb)", om), ("Host 编排", host), ("端到端 total", timings["fullsubnet_total_ms"])):
        s = _stat(vals)
        if s:
            print(
                f"  {label:16s}: mean={s['mean']:7.3f}  p50={s['p50']:7.3f}  "
                f"p95={s['p95']:7.3f}  p99={s['p99']:7.3f}  max={s['max']:7.3f}"
            )

    total = _stat(timings["fullsubnet_total_ms"])
    if total:
        print()
        print("=== raw_acl 端到端实时性 ===")
        print(f"  p99={total['p99']:.2f}ms  max={total['max']:.2f}ms  预算=32ms")
        overrun = sum(1 for v in timings["fullsubnet_total_ms"] if v > 32.0)
        print(
            f"  超 32ms 的 block: {overrun}/{len(timings['fullsubnet_total_ms'])} ({overrun / len(timings['fullsubnet_total_ms']) * 100:.1f}%)"
        )
        if total["p99"] < 32.0:
            print(f"  结论: raw_acl p99={total['p99']:.2f}ms < 32ms，生产链路 OM 推理+Host 编排可实时")
    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
