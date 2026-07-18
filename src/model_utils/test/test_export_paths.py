from model_utils.export_paths import ensure_output_parent, export_work_dir


def test_export_work_dir_defaults_inside_bundle(tmp_path):
    bundle = tmp_path / "policy"
    bundle.mkdir()

    work_dir = export_work_dir(bundle, "ascend")

    assert work_dir == bundle / "model_utils_work" / "ascend"
    assert work_dir.is_dir()


def test_export_work_dir_respects_explicit_override(tmp_path):
    bundle = tmp_path / "policy"
    bundle.mkdir()
    override = tmp_path / "external-work"

    assert export_work_dir(bundle, "rknn", override) == override
    assert override.is_dir()


def test_ensure_output_parent_creates_directory(tmp_path):
    output = tmp_path / "nested" / "model.onnx"

    assert ensure_output_parent(output) == output
    assert output.parent.is_dir()
