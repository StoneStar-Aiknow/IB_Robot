# 一段人声的最终转写,并说明它属于哪一段
#
# 发布契约(每个产出文本的段恰好一条):
#   - 在外部分段输入(external_vad)或共享音频的本地 VAD 输入(shared_audio)下发布,
#     与 output_topic 上的纯文本 std_msgs/String 使用相同文本;两种路径都提供段标识
#   - (source_id, segment_sequence) 与 SpeakerEmbedding、SpeechWakeEvent 同一标识:开启该段的
#     BEGIN 块的 SpeechAudioChunk.source_id 与 sequence。消费者据此把转写、声纹与唤醒事件按段配对,
#     而不是按到达时间:到达顺序在两个节点各自收尾时不可靠,按时间配对会把一段的声纹判给另一段
#   - 被丢弃的段(序号跳变、换源、段未结束又来 BEGIN、样本不足或无文本)不发布

std_msgs/Header header    # stamp=发布时刻;frame_id 同音源
string source_id          # 对应 SpeechAudioChunk.source_id,音源重启后变化
uint64 segment_sequence   # 开启本段的 BEGIN 块的 SpeechAudioChunk.sequence
string text               # 与 output_topic 上发布的文本相同
# 逐 token 对齐(可选,用于按时间截取一段转写中的一部分):
#   - token_starts[i] 是第 i 个 token 的起始时间,相对本段第一个音频样本的秒数
#   - token_ends[i] 是第 i 个 token 在 text 中结束位置(不含)的字符偏移,单调不减,最后一个等于 len(text)
#   - 两者等长;识别器不提供时间戳或对齐无法与 text 对上时都为空,消费者此时只能整段使用 text
float32[] token_starts
uint32[] token_ends
