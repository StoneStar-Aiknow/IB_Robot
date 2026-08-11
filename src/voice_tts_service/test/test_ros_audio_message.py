from array import array

from ibrobot_msgs.msg import SynthesizedAudio


def test_synthesized_audio_keeps_uint8_payload_compact():
    message = SynthesizedAudio(audio_data=array("B", b"RIFF"))

    assert isinstance(message.audio_data, array)
    assert message.audio_data.typecode == "B"
    assert message.audio_data.tobytes() == b"RIFF"
