# Mono 16000 Hz signed 16-bit little-endian PCM, interleaved with its VAD state.
# BEGIN owns all onset/backfill PCM; END owns its final PCM. Never split a
# boundary from its audio. Exactly one source publishes on a configured topic.
# source_id changes on producer restart; sequence increments for every chunk.
std_msgs/Header header
string source_id
uint64 sequence
uint8 state
uint8 STATE_UNDEFINED = 0
uint8 STATE_BEGIN = 1
uint8 STATE_PROCESSING = 2
uint8 STATE_END = 3
uint8[] audio_data
