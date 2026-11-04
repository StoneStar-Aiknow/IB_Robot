"""VAD 判停帧数回归测试。

从最后一个语音帧到节点发布最终结果的延迟，完全由 SPEAKING→ENDING→SILENCE
所需的静音帧数决定（`voice_asr_node._process_audio` 只在 VAD 回到 SILENCE
时才调用 `end_streaming`）。该延迟是与说话内容无关的固定开销，此前没有任何
测试覆盖，参数一改就会静默退化，因此在这里把帧数钉死。

帧时长 = frame_size / sample_rate = 512 / 16000 = 32 ms。
"""

import math

from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
from voice_asr_service.vad_module import VADConfig, VADModule, VADState

_THRESHOLD = 0.5
_SPEECH_CONFIDENCE = 0.9
_SILENCE_CONFIDENCE = 0.1
_ENERGY = 0.05


def _drive_to_speaking(vad: VADModule) -> None:
    for _ in range(math.ceil(vad.config.min_speech_duration / _frame_seconds(vad)) + 1):
        result = vad._update_state(_SPEECH_CONFIDENCE, _ENERGY, _THRESHOLD)
    assert result.state is VADState.SPEAKING


def _frame_seconds(vad: VADModule) -> float:
    return vad.config.frame_size / vad.config.sample_rate


def _silence_frames_until_endpoint(vad: VADModule) -> int:
    """Number of silent frames the module needs to return to SILENCE."""
    for index in range(1, 500):
        result = vad._update_state(_SILENCE_CONFIDENCE, _ENERGY, _THRESHOLD)
        if result.state is VADState.SILENCE:
            return index
    raise AssertionError("VAD never returned to SILENCE")


def test_endpoint_frame_count_matches_configured_timings():
    """判停帧数 = ceil(min_silence/帧) + int(post_roll/帧)，与配置一致。"""
    vad = VADModule(VADConfig())
    _drive_to_speaking(vad)

    frame_seconds = _frame_seconds(vad)
    expected_silence = math.ceil(vad.config.min_silence_duration / frame_seconds)
    expected_post_roll = int(vad.config.sample_rate * vad.config.post_roll_ms / 1000 / vad.config.frame_size)

    assert _silence_frames_until_endpoint(vad) == expected_silence + expected_post_roll


def test_tuned_endpoint_latency_stays_under_half_a_second():
    """调优后的判停时延上限（本地 VAD 部署按 profile 的 voice_asr.vad_* 开启）。

    放宽判停参数会直接拖慢整条语音链路的响应，而且在上层表现为"机器人反应慢"
    而不是测试失败。这里对调优值设一个明确上限：0.3 s 静音 + 0.15 s post_roll
    = 14 帧 = 448 ms。默认值仍是 0.5 s + 500 ms（31 帧），板端整句复核后再改。

    post_roll 曾因为"ENDING 的帧同时是解码器尾部上下文"而不能压到 250 ms 以下
    （实测丢尾字："挥个手" -> "挥个"）。解码器需要的是尾部**样本**而不是尾部
    墙钟时间，这已由 `final_tail_padding_seconds` 在最终解码前补静音承担，
    所以判停可以独立压到 VAD 本身允许的下限。
    """
    vad = VADModule(VADConfig(min_silence_duration=0.3, post_roll_ms=150.0))
    _drive_to_speaking(vad)

    frames = _silence_frames_until_endpoint(vad)

    assert frames == 14
    assert frames * _frame_seconds(vad) < 0.5


def test_runtime_defaults_drive_the_same_endpoint_latency():
    """节点默认参数与 VADConfig 默认值必须给出同一个判停时延。

    节点从 `VOICE_ASR_DEFAULTS` 声明 ROS 参数，再传给 `VADConfig`。两边一旦
    漂移，YAML 不配这些字段时的实际行为就和 dataclass 默认值不一致。
    """
    vad = VADModule(
        VADConfig(
            min_silence_duration=VOICE_ASR_DEFAULTS["vad_min_silence_duration"],
            post_roll_ms=VOICE_ASR_DEFAULTS["vad_post_roll_ms"],
            pre_roll_ms=VOICE_ASR_DEFAULTS["vad_pre_roll_ms"],
            min_speech_duration=VOICE_ASR_DEFAULTS["vad_min_speech_duration"],
        )
    )
    _drive_to_speaking(vad)
    reference = VADModule(VADConfig())
    _drive_to_speaking(reference)

    # Untuned default: 0.5 s + 500 ms, 31 frames (992 ms).
    assert _silence_frames_until_endpoint(vad) == _silence_frames_until_endpoint(reference) == 31


def test_speech_during_post_roll_restarts_the_whole_silence_wait():
    """ENDING 期间的一帧语音会退回 SPEAKING，静音等待从头开始。

    这是真实对话里"短停顿后继续说"的路径；判停参数调小后它更容易被触发，
    必须保证不会提前截断一句话。
    """
    vad = VADModule(VADConfig())
    _drive_to_speaking(vad)

    frame_seconds = _frame_seconds(vad)
    to_ending = math.ceil(vad.config.min_silence_duration / frame_seconds)
    for _ in range(to_ending):
        vad._update_state(_SILENCE_CONFIDENCE, _ENERGY, _THRESHOLD)
    assert vad.state is VADState.ENDING

    assert vad._update_state(_SPEECH_CONFIDENCE, _ENERGY, _THRESHOLD).state is VADState.SPEAKING
    assert _silence_frames_until_endpoint(vad) == to_ending + int(
        vad.config.sample_rate * vad.config.post_roll_ms / 1000 / vad.config.frame_size
    )
