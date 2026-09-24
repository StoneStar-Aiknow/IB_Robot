"""Contract tests for the PI0.5 Ascend910B offline checkpoint tools."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_checkpoint_tool():
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "npu"
        / "pi05_910b"
        / "make_selective_positive_quant_pi05_checkpoint.py"
    )
    spec = spec_from_file_location("pi05_selective_checkpoint_tool", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_model_sidecars_preserves_nested_tokenizer_assets(tmp_path) -> None:
    tool = _load_checkpoint_tool()
    source = tmp_path / "source"
    output = tmp_path / "output"
    tokenizer = source / "tokenizer"
    cache = source / ".cache"
    tokenizer.mkdir(parents=True)
    cache.mkdir()
    output.mkdir()
    (source / "README.md").write_text("model card\n", encoding="utf-8")
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "model.safetensors").touch()
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (cache / "download.lock").touch()

    tool.copy_model_sidecars(source, output)

    assert (output / "README.md").read_text(encoding="utf-8") == "model card\n"
    assert (output / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8") == "{}\n"
    assert not (output / "config.json").exists()
    assert not (output / "model.safetensors").exists()
    assert not (output / ".cache").exists()
