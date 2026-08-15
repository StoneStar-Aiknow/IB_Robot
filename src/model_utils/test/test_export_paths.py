from model_utils.export_paths import ensure_output_parent, export_work_dir


def test_export_work_dir_defaults_outside_bundle(tmp_path):
    models_root = tmp_path / "models"
    bundle = models_root / "policy"
    bundle.mkdir(parents=True)

    work_dir = export_work_dir(bundle, "ascend")

    assert work_dir == models_root / "_work" / "policy" / "ascend"
    assert work_dir.is_dir()
    assert not (bundle / "model_utils_work").exists()


def test_export_work_dir_supports_nested_bundle(tmp_path):
    models_root = tmp_path / "models"
    bundle = models_root / "perception" / "policy"
    bundle.mkdir(parents=True)

    work_dir = export_work_dir(bundle, "rknn")

    assert work_dir == models_root / "_work" / "perception" / "policy" / "rknn"


def test_export_work_dir_falls_back_without_models_root(tmp_path):
    bundle = tmp_path / "policy"
    bundle.mkdir()

    work_dir = export_work_dir(bundle, "ascend")

    assert work_dir == tmp_path / "_work" / "policy" / "ascend"
    assert not (bundle / "model_utils_work").exists()


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
