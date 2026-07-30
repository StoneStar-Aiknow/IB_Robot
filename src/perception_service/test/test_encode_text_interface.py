from ibrobot_msgs.msg import TextEmbedding
from ibrobot_msgs.srv import EncodeText


def test_encode_text_interface_preserves_aligned_vector_metadata():
    request = EncodeText.Request(texts=["cup", "red bottle"])
    response = EncodeText.Response(
        results=[
            TextEmbedding(text_index=0, embedding=[0.6, 0.8], embedding_dim=2, success=True),
            TextEmbedding(text_index=1, embedding=[1.0, 0.0], embedding_dim=2, success=True),
        ],
        success=True,
    )

    assert request.texts == ["cup", "red bottle"]
    assert [item.text_index for item in response.results] == [0, 1]
    assert all(item.embedding_dim == len(item.embedding) for item in response.results)
