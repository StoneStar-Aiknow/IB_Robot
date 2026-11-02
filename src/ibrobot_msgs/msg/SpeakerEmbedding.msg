# 一段人声的声纹嵌入(说话人向量)
#
# 发布契约(每段人声至多一条,无人声不发布):
#   - SpeechAudioChunk 的 BEGIN..END 构成一段人声,段末发布一条;没有 END、靠空闲超时结束的段
#     同样发布一条;被下一个 BEGIN 打断、换源或序号跳变的段与 ASR 一致丢弃,不发布
#     (规则见 voice_asr_service/speech_segments.py)
#   - 语音过短(低于节点 min_speech_seconds)或模型返回空时 embedding 为空:表示"这段没有
#     可用声纹",消费者据此立即判定而不必等待;空向量不是任何人的声纹,不能参与比较
#   - 推理出错时这一段不发布:消费者等满宽限期后按缺失处理(fail-open),节点自行尝试恢复
#   - 不周期性空转发布
#
# 消费者按 (source_id, segment_sequence) 与同一段的 SpeechTranscript 配对,而不是按到达时间:
# 转写要等 VAD 静音加尾部补静音,向量通常先到;没有转写的段的向量不判定任何一句。
#
# 向量是生物特征数据:只在会话内存中比较,不落盘、不跨会话保留。

std_msgs/Header header    # stamp=本段人声最后一个音频样本的时间;frame_id 同音源
string source_id          # 对应 SpeechAudioChunk.source_id,音源重启后变化
uint64 segment_sequence   # 开启本段的 BEGIN 块的 SpeechAudioChunk.sequence
float32[] embedding       # 已 L2 归一化的说话人向量,因此余弦相似度等于点积
string model_id           # 产出该向量的模型标识;不同模型产出的向量不可互相比较
float32 speech_seconds    # 本段人声时长,消费者据此判断向量可信度
