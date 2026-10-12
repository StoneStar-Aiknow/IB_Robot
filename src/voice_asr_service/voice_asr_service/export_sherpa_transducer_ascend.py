"""Prepare a sherpa-onnx streaming transducer encoder for Ascend ATC.

ATC has no ``MatMulInteger`` and needs static shapes, so this tool:

1. rebuilds float ``MatMul`` nodes from an onnxruntime dynamic-quantized (int8) encoder,
   using the dequantized weights (activations stay float, accuracy matches the int8 graph);
2. pins the batch dimension to 1 and, when ``onnxsim`` is installed, constant-folds the graph.

The static ONNX is both the ATC input and the source of the OM tensor bindings
(``package_sherpa_asr_bundle --ascend-encoder-onnx``). Keep it under ``models/_work``.

    python3 -m voice_asr_service.export_sherpa_transducer_ascend \\
        --encoder <bundle_src>/encoder.int8.onnx --output models/_work/<bundle>/ascend/encoder.static.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path


def dequantize_dynamic(model):
    """Replace DynamicQuantizeLinear+MatMulInteger+Cast+Mul chains with MatMul on float weights."""
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    producer = {out: node for node in graph.node for out in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def const(name: str):
        if name in initializers:
            return numpy_helper.to_array(initializers[name])
        node = producer[name]
        if node.op_type != "Constant":
            raise ValueError(f"{name} is not a constant ({node.op_type})")
        return numpy_helper.to_array(node.attribute[0].t)

    replacements = {}
    for node in graph.node:
        if node.op_type != "MatMulInteger":
            continue
        quantize = producer[node.input[0]]
        (cast,) = consumers[node.output[0]]
        (scale_out,) = consumers[cast.output[0]]
        if quantize.op_type != "DynamicQuantizeLinear" or cast.op_type != "Cast" or scale_out.op_type != "Mul":
            raise ValueError(f"unexpected dynamic-quantization pattern around {node.name}")
        scale_mul = producer[next(name for name in scale_out.input if name != cast.output[0])]
        weight_scale = next(name for name in scale_mul.input if name != quantize.output[1])
        zero_point = const(node.input[3]).astype(np.float32) if len(node.input) > 3 and node.input[3] else 0.0
        weight = ((const(node.input[1]).astype(np.float32) - zero_point) * const(weight_scale)).astype(np.float32)
        weight_name = f"{node.input[1]}__dequantized"
        graph.initializer.append(numpy_helper.from_array(weight, weight_name))
        replacements[id(scale_out)] = helper.make_node(
            "MatMul", [quantize.input[0], weight_name], [scale_out.output[0]], name=f"{node.name}__float"
        )
    nodes = [replacements.get(id(node), node) for node in graph.node]

    live = {output.name for output in graph.output}
    kept = []
    for node in reversed(nodes):
        if any(name in live for name in node.output):
            kept.append(node)
            live.update(name for name in node.input if name)
    del graph.node[:]
    graph.node.extend(reversed(kept))
    used = [init for init in graph.initializer if init.name in live]
    del graph.initializer[:]
    graph.initializer.extend(used)
    for prop in list(model.metadata_props):
        if prop.key == "onnx.infer":
            model.metadata_props.remove(prop)
    if any(node.op_type in {"MatMulInteger", "DynamicQuantizeLinear"} for node in graph.node):
        raise ValueError("quantized operators remain after dequantization")
    return model, len(replacements)


def make_static(model):
    """Pin every dynamic dimension to 1 (batch) and constant-fold when onnxsim is available."""
    shapes = {}
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        shapes[value.name] = [dim.dim_value if dim.dim_value else 1 for dim in dims]
        for dim in dims:
            if not dim.dim_value:
                dim.dim_value = 1
    try:
        from onnxsim import simplify
    except ImportError:
        return model, False
    simplified, ok = simplify(model, overwrite_input_shapes=shapes)
    if not ok:
        raise ValueError("onnxsim could not validate the simplified encoder")
    return simplified, True


def main() -> int:
    import onnx

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", type=Path, required=True, help="sherpa encoder ONNX (int8 or float)")
    parser.add_argument("--output", type=Path, required=True, help="static float encoder ONNX to write")
    parser.add_argument("--soc", default="Ascend310P1")
    args = parser.parse_args()

    model, restored = dequantize_dynamic(onnx.load(str(args.encoder)))
    model, simplified = make_static(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"{args.output}: restored {restored} float MatMuls, {'simplified' if simplified else 'not simplified'}")
    om_prefix = args.output.with_name(args.output.stem.replace(".static", "") + "_fp16")
    print(
        "atc --framework=5 "
        f"--model={args.output} --output={om_prefix} --soc_version={args.soc} --precision_mode_v2=fp16"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
