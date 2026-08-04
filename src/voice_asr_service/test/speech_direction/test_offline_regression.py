"""离线 WAV 回归测试:6 个真实录音 → pipeline 段级 DOA → 比对 GT,误差 ≤15°。

回归基线:
  - 6 个 6ch/16kHz 真实录音,文件名含 GT 角度
  - 通过 runtime.feed_audio 灌入(离线模式,无采集设备)
  - 段级 DOA 输出与 GT 比对,圆周误差 ≤15°(5° 步长量化内)

GT:
  sound_16s_180.flac        → 180° 单段
  sound_10s_90.flac         → 90°  单段
  sound_18s_0_0.flac        → 0°, 0° 两段
  sound_10s_90_0.flac       → 90°, 0° 两段
  sound_9s_0_90.flac        → 0°, 90° 两段(第3段碎段无GT,不统计)
  sound_19s_90_90_90_90.flac → 90°×4 四段

音频文件以 FLAC 无损压缩随仓库提供(见 audio/ 目录,合计约 6MB),
soundfile 原生支持 FLAC 读取,DOA 结果与原始 PCM 逐位一致。
环境变量 SPEECH_DIRECTION_AUDIO_DIR 可覆盖默认音频目录(供外部夹具调试)。

基线证据见 audio/BASELINE.md(6 文件 12 段,每段误差均 ≤ 15°)。

运行:
  # 默认读 audio/ 目录;模型缺失或音频缺失时整体 skip
  python -m pytest test_offline_regression.py -v -s
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from voice_asr_service.speech_direction.config import SpeechDirectionConfig  # noqa: E402
from voice_asr_service.speech_direction.doa.srp_phat import StftSrpPhat  # noqa: E402
from voice_asr_service.speech_direction.enhancement.fullsubnet import (  # noqa: E402
    FullSubNetEnhancer,
)
from voice_asr_service.speech_direction.pipeline import (  # noqa: E402
    DoaState,
    PipelineParams,
    SpeechDirectionPipeline,
    VadState,
)
from voice_asr_service.speech_direction.runtime import (  # noqa: E402
    SpeechDirectionRuntime,
)
from voice_asr_service.speech_direction.speech_gate import SpeechGate  # noqa: E402

# ============================ 测试音频与 GT ============================

# 文件 → GT 角度序列(按时间顺序,None=碎段不统计)
SOUND_FILES = [
    ("sound_16s_180.flac", [180]),
    ("sound_10s_90.flac", [90]),
    ("sound_18s_0_0.flac", [0, 0]),
    ("sound_10s_90_0.flac", [90, 0]),
    ("sound_9s_0_90.flac", [0, 90, None]),
    ("sound_19s_90_90_90_90.flac", [90, 90, 90, 90]),
]

# 误差门限:max err 15°(5° 步长量化内)
ERR_THRESHOLD = 15.0

# 音频目录:默认为同目录 audio/(FLAC 夹具随仓库提供);
# 环境变量 SPEECH_DIRECTION_AUDIO_DIR 可覆盖,供外部夹具调试。
_audio_dir_env = os.environ.get("SPEECH_DIRECTION_AUDIO_DIR", "")
AUDIO_DIR = Path(_audio_dir_env) if _audio_dir_env else Path(__file__).with_name("audio")


def _circular_diff(a: float, b: float) -> float:
    """圆周绝对误差(度),结果 [0,180]。"""
    return float(abs(np.mod(a - b + 180, 360) - 180))


def _audio_available() -> bool:
    """检查测试音频是否可用(环境变量已指定且文件齐全)。"""
    if AUDIO_DIR is None:
        return False
    if not AUDIO_DIR.exists():
        return False
    return all((AUDIO_DIR / fname).exists() for fname, _ in SOUND_FILES)


# 真实录音类在资产不可用时整体 skip；资产无关契约仍应固定执行。
_REAL_AUDIO_REQUIRED = pytest.mark.skipif(
    not _audio_available(),
    reason="测试音频不可用:默认目录 audio/ 缺文件,或 SPEECH_DIRECTION_AUDIO_DIR 指向的目录缺文件",
)


def _build_pipeline():
    """构造与 node.py 一致的算法链(FullSubNet + SpeechGate + SRP + Pipeline)。

    需要 FullSubNet 模型就绪(已通过 scripts/download_speech_direction_models.sh 下载),
    否则跳过整个测试。
    """
    cfg = SpeechDirectionConfig()

    # 模型缺失则跳过(require_models 会 raise,这里提前检查)
    from voice_asr_service.speech_direction.model_downloader import require_models

    try:
        require_models()
    except FileNotFoundError as e:
        pytest.skip(f"模型未就绪,跳过回归测试: {e}")

    fullnet = FullSubNetEnhancer(
        repo_dir=cfg.fullnet.repo_dir,
        ckpt=cfg.fullnet.ckpt,
        device=cfg.fullnet.device,
    )
    speech_gate = SpeechGate(
        model_path=cfg.vad.model_path,
        sample_rate=cfg.vad.sample_rate,
        vad_threshold=cfg.gray_region.vad_threshold,
        rms_threshold=cfg.gray_region.rms_threshold,
    )
    angles = np.arange(0, 360, cfg.doa.angle_step_degree, dtype=np.float32)
    srp = StftSrpPhat(
        frame_size=cfg.doa.frame_size,
        hop_size=cfg.doa.hop_size,
        angles_deg=angles,
        sample_rate=cfg.doa.sample_rate,
        mic_positions=np.array(cfg.doa.mic_positions, dtype=np.float64),
        sound_speed=cfg.doa.sound_speed,
        freq_lo=cfg.doa.freq_band_hz[0],
        freq_hi=cfg.doa.freq_band_hz[1],
        diag_freq_hi=cfg.doa.diag_pair_freq_max_hz,
    )
    params = PipelineParams(
        sample_rate=cfg.pipeline.sample_rate,
        frame_size=cfg.pipeline.frame_size,
        hop_size=cfg.pipeline.hop_size,
        input_channels=cfg.pipeline.input_channels,
        seg_end_gap_s=cfg.gray_region.seg_end_gap_s,
        min_seg_dur_s=cfg.gray_region.min_seg_dur_s,
        min_accum_frames=cfg.gray_region.min_accum_frames,
        max_accum_dur_s=cfg.gray_region.max_accum_dur_s,
        seg_max_rms_threshold=cfg.gray_region.seg_max_rms_threshold,
    )
    vad_state = VadState()
    doa_state = DoaState()
    pipeline = SpeechDirectionPipeline(fullnet, speech_gate, srp, params, vad_state, doa_state)

    # 离线模式:enable_capture=False,无采集设备,靠 feed_audio 灌数据
    runtime = SpeechDirectionRuntime(cfg, pipeline, enable_capture=False)
    return runtime, pipeline


def _feed_wav(runtime: SpeechDirectionRuntime, wav_path: Path, chunk: int = 2048) -> None:
    """把 6ch wav 灌入 runtime(按 chunk=hop 节奏,让 worker 实时处理)。

    每次写一个 hop 的 6ch int16 bytes,按真实 hop 节奏(chunk/sr=128ms)sleep,
    让 worker 实时跟随不堆积。
    不用加速系数——会让 hop 间隔 > seg_end_gap(150ms)误切段。
    """
    audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
    if sr != 16000:
        raise ValueError(f"采样率 {sr} != 16000")
    if audio.shape[1] < 6:
        raise ValueError(f"通道数 {audio.shape[1]} < 6")
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    n = int16.shape[0]
    for s in range(0, n, chunk):
        block = int16[s : s + chunk]
        if block.shape[0] < chunk:
            block = np.pad(block, ((0, chunk - block.shape[0]), (0, 0)))
        runtime.feed_audio(block.tobytes())
        time.sleep(chunk / sr)


@_REAL_AUDIO_REQUIRED
class TestOfflineRegression:
    """离线 WAV 回归测试:6 个真实录音,段级 DOA 误差 ≤15°。"""

    @pytest.fixture(scope="class")
    def runtime(self):
        """构造算法链(所有测试共享,模型只加载一次)。"""
        rt, pipeline = _build_pipeline()
        rt.start()
        yield rt
        rt.stop()

    @pytest.mark.parametrize(
        "fname,gt_list",
        SOUND_FILES,
        ids=[f[0].replace(".flac", "") for f in SOUND_FILES],
    )
    def test_segment_doa_within_15deg(self, runtime, fname, gt_list):
        """单个音频:段级 DOA 输出与 GT 圆周误差 ≤15°。"""
        wav_path = AUDIO_DIR / fname
        if not wav_path.exists():
            pytest.skip(f"音频不存在: {wav_path}")

        # 重置 pipeline 状态(每个文件独立测,丢弃上个文件残留)
        runtime.pipeline.reset()
        runtime.doa_state._angle = None
        runtime.doa_state._wall_clock_ts = 0.0
        runtime.doa_state._seq_id = 0
        # 重新注册 reader 从最新位置读
        runtime._reader_id = runtime.ring_buffer.register(start_latest=True)
        time.sleep(0.3)  # 给 worker 时间消化残留
        runtime._reader_id = runtime.ring_buffer.register(start_latest=True)

        # 灌入音频
        _feed_wav(runtime, wav_path)
        # 等 worker 处理完尾部(seg_end_gap=0.15s,多等一会)
        time.sleep(0.8)

        # 收集所有段级输出(中间方向 + 段末方向,按时间顺序)
        history = runtime.pipeline.get_segment_history()

        # 期望段数 = GT 中非 None 的个数
        expected_segs = [g for g in gt_list if g is not None]

        # 段数不足(漏检)直接失败
        if len(history) < len(expected_segs):
            pytest.fail(f"{fname}: 段数不足,期望 {len(expected_segs)} 实际 {len(history)}(漏检)")

        # 按时间顺序逐段与 GT 比对
        # history[i] 对应 gt_list[i],GT 为 None 的段(碎段)跳过不统计
        errors = []
        for i, (_output_seq, angle, _hop_t, seg_type) in enumerate(history):
            gt = gt_list[i] if i < len(gt_list) else None
            if gt is None:
                # GT 无此段(碎段或超出预期),跳过不统计
                continue
            err = _circular_diff(float(angle), float(gt))
            errors.append((fname, i, angle, gt, err, seg_type))

        # 所有段误差 ≤15°
        failed = [e for e in errors if e[4] > ERR_THRESHOLD]
        if failed:
            detail = "; ".join(f"段{e[1]}({e[5]}): DOA={e[2]}° GT={e[3]}° err={e[4]:.0f}°" for e in failed)
            pytest.fail(f"{fname}: {len(failed)}/{len(errors)} 段误差 >{ERR_THRESHOLD}° — {detail}")

        # 打印通过信息(调试用,-s 时可见)
        for fname_, seg_i, angle, gt, err, seg_type in errors:
            print(f"    {fname_} 段{seg_i}({seg_type}): DOA={angle}° GT={gt}° err={err:.0f}° [PASS]")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
