# 音源报告自己的唤醒词:唤醒词所在的那一段人声
#
# 发布契约(每次唤醒恰好一条):
#   - 音源在释放人声前就吃掉唤醒词(X2 实测如此),文本里看不到它;这条消息说明
#     (source_id, segment_sequence) 那一段人声是由唤醒词开启的,而不是同一会话的后续话
#   - (source_id, segment_sequence) 与 SpeakerEmbedding 同一标识:开启该段的 BEGIN 块的
#     SpeechAudioChunk.source_id 与 sequence
#   - 判定在 BEGIN 前后才成立(X2 的确认最晚在 BEGIN 后 1.0 s 到达),所以消息可能晚于该段的 BEGIN 块
#
# 消费者按段使用它:同一音源、从该段起的第一条 SpeechTranscript 是唤醒后的首句(音源有时先为
# 唤醒词残留单开一段再丢弃);序号更早的段即使转写晚于本消息到达,也不是唤醒后的首句。
# 本消息与转写走不同话题,谁先到都可能:消费者让无会话时的转写等它一个宽限期;
# 该段(或同源更晚的段)的转写已经处理过才到的本消息作废,不转给之后的段。

std_msgs/Header header    # stamp=判定成立的时刻;frame_id 同音源
string source_id          # 对应 SpeechAudioChunk.source_id,音源重启后变化
uint64 segment_sequence   # 唤醒词所在那一段人声的 BEGIN 块的 SpeechAudioChunk.sequence
