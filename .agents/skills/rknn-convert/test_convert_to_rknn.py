import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("convert_to_rknn.py")
SPEC = importlib.util.spec_from_file_location("convert_to_rknn", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_abi_only_keeps_layout_for_image_inputs():
    attrs = {
        "pixel_values": {
            "idx": 0,
            "dtype": "float32",
            "shape": [1, 3, 512, 512],
            "layout": "nchw",
            "is_output": False,
        },
        "past_key_0": {
            "idx": 1,
            "dtype": "float32",
            "shape": [1, 177, 5, 64],
            "layout": "NCHW",
            "is_output": False,
        },
        "past_value_0": {
            "idx": 0,
            "dtype": "float32",
            "shape": [1, 177, 5, 64],
            "layout": "NCHW",
            "is_output": True,
        },
    }

    abi = MODULE._runtime_abi_from_attrs(attrs)

    assert abi["inputs"][0]["layout"] == "NCHW"
    assert "layout" not in abi["inputs"][1]
    assert "layout" not in abi["outputs"][0]
