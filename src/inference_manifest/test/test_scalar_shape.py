from inference_manifest import SemanticTensor, TensorBinding, manifest_schema


def test_rank_zero_scalar_shape_is_valid_in_typed_and_json_schema_contracts():
    semantic = SemanticTensor(semantic="temperature", dtype="float32", shape=())
    binding = TensorBinding(
        semantic="temperature",
        runtime_name="temperature",
        index=0,
        dtype="float32",
        shape=(),
    )

    assert semantic.shape == ()
    assert binding.shape == ()
    assert "minItems" not in manifest_schema()["$defs"]["semantic_tensor"]["properties"]["shape"]
    assert "minItems" not in manifest_schema()["$defs"]["binding"]["properties"]["shape"]
