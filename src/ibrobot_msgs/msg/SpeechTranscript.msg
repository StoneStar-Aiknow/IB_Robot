# 一段人声的最终转写,并说明它属于哪一段
#
# 发布契约(每个产出文本的段恰好一条):
#   - 只在外部分段输入(voice_asr.input_mode=external_vad)下发布,与 output_topic 上的纯文本
#     std_msgs/String 同一时刻、同一文本;本地 VAD 输入没有段标识,不发布
#   - (source_id, segment_sequence) 与 SpeakerEmbedding、SpeechWakeEvent 同一标识:开启该段的
#     BEGIN 块的 SpeechAudioChunk.source_id 与 sequence。消费者据此把转写、声纹与唤醒事件按段配对,
#     而不是按到达时间:到达顺序在两个节点各自收尾时不可靠,按时间配对会把一段的声纹判给另一段
#   - 被丢弃的段(序号跳变、换源、段未结束又来 BEGIN、样本不足或无文本)不发布

std_msgs/Header header    # stamp=发布时刻;frame_id 同音源
string source_id          # 对应 SpeechAudioChunk.source_id,音源重启后变化
uint64 segment_sequence   # 开启本段的 BEGIN 块的 SpeechAudioChunk.sequence
string text               # 与 output_topic 上发布的文本相同
