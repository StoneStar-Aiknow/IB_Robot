"""speech_direction 执行契约，由打包脚本、校验脚本与 session 共享。

speech_direction 在 310P 上由三段 OM 组成：Silero VAD 与 FullSubNet 的 FB/SB
两个子网。三段各有自己的 LSTM recurrent state，且每段 state 的 producer 与
consumer 是同一个 role（state_out 喂回 state_in），属于 v3 ``Deployment``
禁止的 device link 自循环，因此 state 一律走 session 内 ``allocate_device_buffer``
+ ``prepare_dataset_banks`` 双 bank，在 Device 内 ping-pong，不声明任何
``DeviceLink``。FB 的输出不直接喂给 SB，而是回到 Host 经 STFT 归一化、sub-band
滑窗组织后再进 SB，所以 FB→SB 也没有 device 直传，全部用 ``host.`` 命名空间。

这套语义以前散落在 ``silero_acl.py`` 的 ``_LooseBindings`` 与
``fullsubnet_stateful_executor.py`` 的 shape 常量里，三处各写一份。打包脚本、
session、校验脚本统一读这里，避免一端改了 shape 另一端仍按旧值比对。
"""

from __future__ import annotations

from inference_manifest.models import HOST_SEMANTIC_PREFIX, ArtifactBindings, TensorBinding

# 契约版本：当 role 集合、role 顺序、state shape 或 binding 语义发生变化、导致
# 已导出的 OM 不可再用时递增。打包脚本把它写进 manifest，session 加载时校验，
# 避免用旧契约导出的 OM 被新契约的 session 驱动出错误的 shape。
#
# 1: speech_direction 接入 inference_manifest，三个 role 统一为 host.* 语义，
#    LSTM state 改由 session 双 bank 常驻 Device，不再用 _LooseBindings 自循环。
SPEECH_DIRECTION_CONTRACT_VERSION = 1

# Silero VAD 与 FullSubNet stateful 是两个独立模型族，拆成两个 deployment，
# 各自独立 acquire lease（与现状一致）。这里分两组列出 role 执行顺序：
# - silero 组只有 silero_vad 一个 role，每音频帧推进一次 LSTM state；
# - fullsubnet 组固定 FB 在前、SB 在后，FB 输出经 Host 组织后喂给 SB。
SPEECH_DIRECTION_SILERO_EXECUTION: tuple[str, ...] = ("silero_vad",)
SPEECH_DIRECTION_FULLSUBNET_EXECUTION: tuple[str, ...] = ("fullsubnet_fb", "fullsubnet_sb")

# Silero VAD OM（openvino_16k 变体）已把采样率折叠为常量，仅保留音频与 LSTM state
# 两个输入、state 与概率两个输出。state 自循环（state_out→state_in），走双 bank。
_SILERO = f"{HOST_SEMANTIC_PREFIX}silero."
SILERO_AUDIO_SEMANTIC = f"{_SILERO}audio"
SILERO_STATE_IN_SEMANTIC = f"{_SILERO}state_in"
SILERO_STATE_OUT_SEMANTIC = f"{_SILERO}state_out"
SILERO_PROB_SEMANTIC = f"{_SILERO}prob"

# Silero VAD OM 的静态 ABI shape（openvino_16k 变体：音频 576 采样、state 折叠为
# 单个 [2,1,128] tensor、概率标量）。runner 与校验脚本统一从这里读，不再各处硬编码。
SILERO_AUDIO_SHAPE = (1, 576)
SILERO_STATE_SHAPE = (2, 1, 128)
SILERO_PROB_SHAPE = (1, 1)
SILERO_DTYPE = "float32"

# FullSubNet FB/SB 子网的固定 ABI。FB 输入频谱帧、输出增强特征；SB 输入 sub-band
# 特征、输出复数掩码分量。两者的 LSTM hidden/cell 是两个独立 tensor，各自自循环
# （hidden_out 喂回 hidden_in、cell_out 喂回 cell_in），走双 bank；FB 输出不直接进
# SB，而是回 Host 经 _build_and_normalize_sb 组织，故 FB 输出语义也是 host.*
# （无 in-graph 的下游 producer）。注意 OM 的 state_in/out 是 hidden、cell 两个
# 独立 tensor，与 Silero 把 h/c 折叠成单个 state tensor 不同，故这里拆成四个语义。
_FULLSUBNET = f"{HOST_SEMANTIC_PREFIX}fullsubnet."
FULLSUBNET_FB_SPECTRUM_SEMANTIC = f"{_FULLSUBNET}fb_spectrum"
FULLSUBNET_FB_HIDDEN_IN_SEMANTIC = f"{_FULLSUBNET}fb_hidden_in"
FULLSUBNET_FB_CELL_IN_SEMANTIC = f"{_FULLSUBNET}fb_cell_in"
FULLSUBNET_FB_HIDDEN_OUT_SEMANTIC = f"{_FULLSUBNET}fb_hidden_out"
FULLSUBNET_FB_CELL_OUT_SEMANTIC = f"{_FULLSUBNET}fb_cell_out"
FULLSUBNET_FB_FEATURES_SEMANTIC = f"{_FULLSUBNET}fb_features"
FULLSUBNET_SB_FEATURES_SEMANTIC = f"{_FULLSUBNET}sb_features"
FULLSUBNET_SB_HIDDEN_IN_SEMANTIC = f"{_FULLSUBNET}sb_hidden_in"
FULLSUBNET_SB_CELL_IN_SEMANTIC = f"{_FULLSUBNET}sb_cell_in"
FULLSUBNET_SB_HIDDEN_OUT_SEMANTIC = f"{_FULLSUBNET}sb_hidden_out"
FULLSUBNET_SB_CELL_OUT_SEMANTIC = f"{_FULLSUBNET}sb_cell_out"
FULLSUBNET_SB_MASK_SEMANTIC = f"{_FULLSUBNET}sb_mask"

# FullSubNet FB/SB 静态 ABI shape（与 OM 导出 ABI 逐字对齐）。FB 输入频谱帧
# [B=4, T=2, F=257]、SB 输入 sub-band 特征 [B*F=1028, T=2, features=32]；两者
# 的 LSTM state 是 [layers=2, batch, hidden]，FB hidden=512、SB hidden=384。
# runner 与校验脚本统一从这里读，不再在 silero_acl/fullsubnet_stateful_acl 各写一份。
_FULLSUBNET_DTYPE = "float32"
FULLSUBNET_FB_FRAME_SHAPE = (4, 2, 257)
FULLSUBNET_FB_STATE_SHAPE = (2, 4, 512)
FULLSUBNET_FB_OUTPUT_SHAPE = FULLSUBNET_FB_FRAME_SHAPE
FULLSUBNET_SB_FRAME_SHAPE = (1028, 2, 32)
FULLSUBNET_SB_STATE_SHAPE = (2, 1028, 384)
FULLSUBNET_SB_OUTPUT_SHAPE = (1028, 2, 2)

# 节点对外暴露的服务语义：进 4ch 音频、出 VAD 概率与 4ch 增强音频。
# 这是 ModelDescriptor 声明的唯一对外契约。
SPEECH_DIRECTION_AUDIO_INPUT_SEMANTIC = "observation.audio_4ch"
SPEECH_DIRECTION_VAD_PROB_SEMANTIC = "voice.vad_prob"
SPEECH_DIRECTION_AUDIO_OUTPUT_SEMANTIC = "voice.audio_enhanced_4ch"
SPEECH_DIRECTION_OUTPUTS = frozenset({SPEECH_DIRECTION_VAD_PROB_SEMANTIC, SPEECH_DIRECTION_AUDIO_OUTPUT_SEMANTIC})

# speech_direction 全部用 host.* 语义（无 device 间直传），便于 session 统一识别。
SPEECH_DIRECTION_HOST_SEMANTICS = frozenset(
    {
        SILERO_AUDIO_SEMANTIC,
        SILERO_STATE_IN_SEMANTIC,
        SILERO_STATE_OUT_SEMANTIC,
        SILERO_PROB_SEMANTIC,
        FULLSUBNET_FB_SPECTRUM_SEMANTIC,
        FULLSUBNET_FB_HIDDEN_IN_SEMANTIC,
        FULLSUBNET_FB_CELL_IN_SEMANTIC,
        FULLSUBNET_FB_HIDDEN_OUT_SEMANTIC,
        FULLSUBNET_FB_CELL_OUT_SEMANTIC,
        FULLSUBNET_FB_FEATURES_SEMANTIC,
        FULLSUBNET_SB_FEATURES_SEMANTIC,
        FULLSUBNET_SB_HIDDEN_IN_SEMANTIC,
        FULLSUBNET_SB_CELL_IN_SEMANTIC,
        FULLSUBNET_SB_HIDDEN_OUT_SEMANTIC,
        FULLSUBNET_SB_CELL_OUT_SEMANTIC,
        FULLSUBNET_SB_MASK_SEMANTIC,
    }
)


def speech_direction_input_semantics(role: str) -> dict[str, str]:
    """按 role 返回输入槽位到 manifest 语义的映射（ABI 索引顺序）。

    Silero 的 LSTM 把 h/c 折叠成单个 state tensor，故只有 audio + state_in；
    FB/SB 的 hidden、cell 是两个独立 tensor，紧跟主输入之后，与 OM 输入 ABI
    顺序对齐。session 据此把双 bank 的 Device buffer 注入到对应 dataset 槽位。
    """
    if role == "silero_vad":
        return {"audio": SILERO_AUDIO_SEMANTIC, "state_in": SILERO_STATE_IN_SEMANTIC}
    if role == "fullsubnet_fb":
        return {
            "spectrum": FULLSUBNET_FB_SPECTRUM_SEMANTIC,
            "hidden_in": FULLSUBNET_FB_HIDDEN_IN_SEMANTIC,
            "cell_in": FULLSUBNET_FB_CELL_IN_SEMANTIC,
        }
    if role == "fullsubnet_sb":
        return {
            "features": FULLSUBNET_SB_FEATURES_SEMANTIC,
            "hidden_in": FULLSUBNET_SB_HIDDEN_IN_SEMANTIC,
            "cell_in": FULLSUBNET_SB_CELL_IN_SEMANTIC,
        }
    raise ValueError(f"unknown speech_direction role {role!r}")


def speech_direction_output_semantics(role: str) -> dict[str, str]:
    """按 role 返回输出槽位到 manifest 语义的映射（ABI 索引顺序）。

    hidden_out/cell_out 与主输出并列，session 据此把双 bank 切换后的 Device
    buffer 读回或就地续用。hidden_out/cell_out 与 hidden_in/cell_in 是同一个
    Device buffer 的两个逻辑视角（ping-pong 下互为镜像），不构成 device link。
    """
    if role == "silero_vad":
        return {"prob": SILERO_PROB_SEMANTIC, "state_out": SILERO_STATE_OUT_SEMANTIC}
    if role == "fullsubnet_fb":
        return {
            "features": FULLSUBNET_FB_FEATURES_SEMANTIC,
            "hidden_out": FULLSUBNET_FB_HIDDEN_OUT_SEMANTIC,
            "cell_out": FULLSUBNET_FB_CELL_OUT_SEMANTIC,
        }
    if role == "fullsubnet_sb":
        return {
            "mask": FULLSUBNET_SB_MASK_SEMANTIC,
            "hidden_out": FULLSUBNET_SB_HIDDEN_OUT_SEMANTIC,
            "cell_out": FULLSUBNET_SB_CELL_OUT_SEMANTIC,
        }
    raise ValueError(f"unknown speech_direction role {role!r}")


def speech_direction_bindings(role: str) -> ArtifactBindings:
    """构造 role 的标准 ``ArtifactBindings``（含 semantic/index/dtype/shape）。

    替代原 ``silero_acl._bindings`` / ``fullsubnet_stateful_acl._bindings`` 两处
    重复的私有 ``_LooseBindings``。runner 直接把返回值传给 ``AclModel(lease,
    role, path, bindings)``，ABI 契约（semantic + shape）由本模块 SSOT，消费方
    不再各自定义 binding 类。

    index 按 ABI 索引顺序（0,1,2）赋值，与 OM 输入/输出 ABI 顺序对齐。
    """
    if role == "silero_vad":
        return ArtifactBindings(
            inputs=(
                TensorBinding(semantic=SILERO_AUDIO_SEMANTIC, index=0, dtype=SILERO_DTYPE, shape=SILERO_AUDIO_SHAPE),
                TensorBinding(semantic=SILERO_STATE_IN_SEMANTIC, index=1, dtype=SILERO_DTYPE, shape=SILERO_STATE_SHAPE),
            ),
            outputs=(
                TensorBinding(semantic=SILERO_PROB_SEMANTIC, index=0, dtype=SILERO_DTYPE, shape=SILERO_PROB_SHAPE),
                TensorBinding(
                    semantic=SILERO_STATE_OUT_SEMANTIC, index=1, dtype=SILERO_DTYPE, shape=SILERO_STATE_SHAPE
                ),
            ),
        )
    if role == "fullsubnet_fb":
        return ArtifactBindings(
            inputs=(
                TensorBinding(
                    semantic=FULLSUBNET_FB_SPECTRUM_SEMANTIC,
                    index=0,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_FB_FRAME_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_FB_HIDDEN_IN_SEMANTIC,
                    index=1,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_FB_STATE_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_FB_CELL_IN_SEMANTIC,
                    index=2,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_FB_STATE_SHAPE,
                ),
            ),
            outputs=(
                TensorBinding(
                    semantic=FULLSUBNET_FB_FEATURES_SEMANTIC,
                    index=0,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_FB_OUTPUT_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_FB_HIDDEN_OUT_SEMANTIC,
                    index=1,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_FB_STATE_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_FB_CELL_OUT_SEMANTIC,
                    index=2,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_FB_STATE_SHAPE,
                ),
            ),
        )
    if role == "fullsubnet_sb":
        return ArtifactBindings(
            inputs=(
                TensorBinding(
                    semantic=FULLSUBNET_SB_FEATURES_SEMANTIC,
                    index=0,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_SB_FRAME_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_SB_HIDDEN_IN_SEMANTIC,
                    index=1,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_SB_STATE_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_SB_CELL_IN_SEMANTIC,
                    index=2,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_SB_STATE_SHAPE,
                ),
            ),
            outputs=(
                TensorBinding(
                    semantic=FULLSUBNET_SB_MASK_SEMANTIC,
                    index=0,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_SB_OUTPUT_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_SB_HIDDEN_OUT_SEMANTIC,
                    index=1,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_SB_STATE_SHAPE,
                ),
                TensorBinding(
                    semantic=FULLSUBNET_SB_CELL_OUT_SEMANTIC,
                    index=2,
                    dtype=_FULLSUBNET_DTYPE,
                    shape=FULLSUBNET_SB_STATE_SHAPE,
                ),
            ),
        )
    raise ValueError(f"unknown speech_direction role {role!r}")
