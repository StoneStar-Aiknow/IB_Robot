"""
Convert ONNX models to RKNN format for RK3588 NPU.

Uses dedicated .venv-rknn environment (torch<=2.4.0, numpy<=1.26.4).

Usage:
  .venv-rknn/bin/python convert_to_rknn.py --onnx model.onnx --output model.rknn --mode float16
  .venv-rknn/bin/python convert_to_rknn.py --onnx model.onnx --output model.rknn --mode int8
  .venv-rknn/bin/python convert_to_rknn.py --onnx model.onnx --output model.rknn --mode hybrid
  .venv-rknn/bin/python convert_to_rknn.py --onnx model.onnx --output model.rknn --mode float16 --verify
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx

if not hasattr(onnx, "mapping"):
    import types

    _t2np = {k: v.np_dtype for k, v in onnx._mapping.TENSOR_TYPE_MAP.items()}
    _np2t = {v: k for k, v in _t2np.items()}
    _m = types.ModuleType("onnx.mapping")
    _m.TENSOR_TYPE_TO_NP_TYPE = _t2np
    _m.NP_TYPE_TO_TENSOR_TYPE = _np2t
    onnx.mapping = _m
    sys.modules["onnx.mapping"] = _m


def inspect_onnx(path):
    model = onnx.load(path)
    print(f"Model: {path}")
    print("Inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    print("Outputs:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    print(f"Opset: {model.opset_import[0].version}")
    return model


def generate_random_calibration(dataset_path, onnx_path, num_samples=20):
    model = onnx.load(onnx_path)
    cal_dir = os.path.join(os.path.dirname(dataset_path), "calibration_data")
    os.makedirs(cal_dir, exist_ok=True)

    input_infos = []
    for inp in model.graph.input:
        name = inp.name
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        elem_type = inp.type.tensor_type.elem_type
        input_infos.append((name, shape, elem_type))

    lines = []
    for i in range(num_samples):
        paths = []
        for j, (_name, shape, elem_type) in enumerate(input_infos):
            if elem_type == 1:
                data = np.random.rand(*shape).astype(np.float32)
            elif elem_type == 7:
                data = np.random.randint(0, 10, size=shape).astype(np.int64)
            else:
                data = np.random.rand(*shape).astype(np.float32)
            p = os.path.join(cal_dir, f"input_{i}_{j}.npy")
            np.save(p, data)
            paths.append(p)
        lines.append(" ".join(paths))

    with open(dataset_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated {num_samples} calibration samples -> {dataset_path}")
    return dataset_path


def _is_image_tensor(name, shape):
    # Project exporters use image/pixel in controlled ONNX image input names.
    normalized_name = name.lower()
    return len(shape) == 4 and ("image" in normalized_name or "pixel" in normalized_name)


def _runtime_abi_from_attrs(attrs):
    if not isinstance(attrs, dict) or not attrs:
        raise RuntimeError("RKNN Toolkit did not expose compiler-resolved tensor metadata")

    def collect(is_output):
        tensors = []
        for name, value in attrs.items():
            if not isinstance(value, dict) or bool(value.get("is_output", False)) != is_output:
                continue
            index = value.get("idx")
            dtype = value.get("dtype")
            shape = value.get("shape")
            layout = value.get("layout")
            if type(index) is not int or not isinstance(dtype, str) or not isinstance(shape, list) or not shape:
                raise RuntimeError(f"RKNN tensor metadata is incomplete for {name!r}: {value!r}")
            item = {"name": name, "index": index, "dtype": dtype, "shape": shape}
            if not is_output and _is_image_tensor(name, shape) and isinstance(layout, str):
                item["layout"] = layout.upper()
            tensors.append(item)
        tensors.sort(key=lambda item: item["index"])
        indices = [item["index"] for item in tensors]
        if len(indices) != len(set(indices)):
            raise RuntimeError("RKNN tensor metadata contains duplicate indices")
        if not is_output and indices != list(range(len(tensors))):
            raise RuntimeError("RKNN input tensor metadata indices are not contiguous from zero")
        return tensors

    abi = {"inputs": collect(False), "outputs": collect(True)}
    if not abi["inputs"] or not abi["outputs"]:
        raise RuntimeError("RKNN compiler metadata contains no runtime inputs or outputs")
    return abi


def _extract_compiled_runtime_abi(rknn):
    metadata = getattr(rknn.rknn_base, "inputs_meta", None)
    attrs = metadata.get("attrs") if isinstance(metadata, dict) else None
    return _runtime_abi_from_attrs(attrs)


def convert(onnx_path, output_path, mode, dataset=None, verify=False, abi_output=None):
    from rknn.api import RKNN

    inspect_onnx(onnx_path)

    rknn = RKNN(verbose=True)

    if mode == "float16":
        print("=" * 60)
        print("Mode: float16 (no quantization)")
        print("=" * 60)
        ret = rknn.config(
            target_platform="rk3588",
            float_dtype="float16",
            optimization_level=3,
            single_core_mode=False,
        )
    elif mode == "int8":
        print("=" * 60)
        print("Mode: int8 quantization")
        print("=" * 60)
        ret = rknn.config(
            target_platform="rk3588",
            quantized_dtype="w8a8",
            quantized_algorithm="normal",
            quantized_method="channel",
            optimization_level=3,
        )
    elif mode == "hybrid":
        print("=" * 60)
        print("Mode: hybrid quantization")
        print("=" * 60)
        ret = rknn.config(
            target_platform="rk3588",
            quantized_dtype="w8a8",
            quantized_algorithm="normal",
            quantized_method="channel",
            optimization_level=3,
        )

    if ret != 0:
        print("Config failed!")
        return -1

    print("--> Loading ONNX model")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print("Load ONNX failed!")
        return -1

    if mode == "float16":
        print("--> Building RKNN model (float16)")
        ret = rknn.build(do_quantization=False)
    elif mode == "int8":
        if dataset is None:
            dataset = os.path.join(os.path.dirname(output_path), "calibration_dataset.txt")
            generate_random_calibration(dataset, onnx_path)
        print(f"--> Building RKNN model with calibration: {dataset}")
        ret = rknn.build(do_quantization=True, dataset=dataset)
    elif mode == "hybrid":
        if dataset is None:
            dataset = os.path.join(os.path.dirname(output_path), "calibration_dataset.txt")
            generate_random_calibration(dataset, onnx_path)
        print("--> Building RKNN model with hybrid quantization")
        ret = rknn.build(do_quantization=True, dataset=dataset, auto_hybrid=True)

    if ret != 0:
        print("Build failed!")
        return -1

    try:
        abi = _extract_compiled_runtime_abi(rknn)
    except RuntimeError as exc:
        print(f"Runtime ABI introspection failed: {exc}")
        rknn.release()
        return -1

    print(f"--> Exporting RKNN model to {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export failed!")
        return -1

    abi_path = abi_output or f"{output_path}.abi.json"
    os.makedirs(os.path.dirname(abi_path) or ".", exist_ok=True)
    with open(abi_path, "w", encoding="utf-8") as f:
        json.dump(abi, f, indent=2)
        f.write("\n")
    print(f"Runtime ABI: {abi_path}")

    rknn.release()

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    rknn_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX: {onnx_size:.1f}MB -> RKNN: {rknn_size:.1f}MB ({mode})")
    print("Done!")

    if verify:
        verify_accuracy(onnx_path, output_path)

    return 0


def verify_accuracy(onnx_path, rknn_path):
    print("=" * 60)
    print("Verifying RKNN vs ONNX accuracy")
    print("=" * 60)

    import onnxruntime as ort

    model = onnx.load(onnx_path)
    input_infos = []
    for inp in model.graph.input:
        name = inp.name
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
        input_infos.append((name, shape))

    inputs = []
    for _name, shape in input_infos:
        inputs.append(np.random.randn(*shape).astype(np.float32))

    onnx_session = ort.InferenceSession(onnx_path)
    input_dict = {info[0]: inp for info, inp in zip(input_infos, inputs, strict=False)}
    onnx_out = onnx_session.run(None, input_dict)[0]

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(target=None)
    rknn_out = rknn.inference(inputs=inputs, data_format="nchw")
    rknn.release()

    if rknn_out is not None:
        rknn_arr = np.array(rknn_out[0])
        diff = np.abs(onnx_out.flatten() - rknn_arr.flatten())
        print(f"ONNX output shape: {onnx_out.shape}")
        print(f"RKNN output shape: {rknn_arr.shape}")
        print(f"Max abs diff: {diff.max():.6f}")
        print(f"Mean abs diff: {diff.mean():.6f}")
    print("Verification done!")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for RK3588")
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--output", required=True, help="Output RKNN model path")
    parser.add_argument(
        "--mode", choices=["float16", "int8", "hybrid"], default="float16", help="Conversion mode (default: float16)"
    )
    parser.add_argument("--dataset", default=None, help="Calibration dataset file (int8/hybrid)")
    parser.add_argument("--verify", action="store_true", help="Verify accuracy after conversion")
    parser.add_argument(
        "--abi-output",
        default=None,
        help="Output compiler-resolved ABI JSON (default: <output>.abi.json)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX model not found: {args.onnx}")
        sys.exit(1)

    ret = convert(args.onnx, args.output, args.mode, args.dataset, args.verify, args.abi_output)
    sys.exit(ret if ret else 0)


if __name__ == "__main__":
    main()
