"""speech_direction 子包:人声方向感知能力。

基于 ReSpeaker 4-Mic Array,实现 FullSubNet 增强 + Silero VAD + 保相位 STFT-SRP-PHAT
的实时人声方向估计。算法与回归基线逐字对齐。

目录职责:
- types.py:         公共数据类型
- config.py:        配置加载与参数
- runtime.py:       ROS 音频缓冲 + worker 线程生命周期
- streaming_runtime.py: unified-runtime stream ownership adapter
- pipeline.py:      增强 + 人声门控 + DOA 处理链
- speech_gate.py:   Silero + RMS 灰区段判定
- node.py:          ROS 适配层
- enhancement/:     FullSubNet 增强
- doa/:             SRP-PHAT 算法
- diagnostics/:     维测旁路
"""
