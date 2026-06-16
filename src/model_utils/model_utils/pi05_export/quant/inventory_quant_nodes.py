#!/usr/bin/env python
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Inventory the int8 vs fp16 Linears in a (quantized or not) PI05 ONNX.

Answers three concrete questions about a W8A8 export *without running it*:

* **What actually got quantized?** For every ``MatMul`` / ``Gemm`` / ``Conv``
  node it reports int8 (weight initializer is ``INT8`` and/or fed by an
  ``AscendQuant``) vs fp16, and — crucially — splits fp16 into ``fp16w`` (the
  node HAS a weight initializer but stayed fp16 → a *missed* quant target) and
  ``fp16a`` (no weight: an activation×activation BMM like Q@Kᵀ / attn@V, which is
  *correctly* fp16). Without that split the QK/AV BMMs inflate the "fp16" count
  and make a fully-quantized graph look half-done.
* **Where did the bytes go?** It sums weight-initializer bytes per bucket.
* **Is the ``.data`` bloated?** It sums **every** initializer (weights + scales +
  biases + embeddings) — the true in-graph data size — and compares it to the
  ``.data`` file size on disk. A large positive gap means dead/appended bytes
  (e.g. ONNX's append-mode external-data writer doubling a re-saved sidecar),
  not real extra weights.

Nodes are bucketed by name namespace: ``vision`` (SigLIP / ``vision_tower`` /
``patch_embed``), ``projector`` (``multi_modal_projector``), ``llm`` (the Gemma
trunk) and ``other``.

Topology only is loaded (``load_external_data=False``), so it is fast and never
needs the multi-GB ``.data`` sidecar in memory.

Usage::

    python -m model_utils.pi05_export.quant.inventory_quant_nodes \\
        /path/to/pi05-vlm-w8a8-all.onnx

    # Compare two graphs side by side (e.g. all vs linear-only, or donor vs NPU):
    python -m model_utils.pi05_export.quant.inventory_quant_nodes \\
        /path/to/a.onnx /path/to/b.onnx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import onnx
from onnx import TensorProto

_COMPUTE_OPS = {"MatMul", "Gemm", "Conv", "MatMulInteger", "QLinearMatMul"}

# Bytes per element for the dtypes that appear on weight initializers.
_DTYPE_BYTES: dict[int, int] = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.INT64: 8,
}
_DTYPE_NAME: dict[int, str] = {
    TensorProto.FLOAT: "fp32",
    TensorProto.FLOAT16: "fp16",
    TensorProto.BFLOAT16: "bf16",
    TensorProto.INT8: "int8",
    TensorProto.UINT8: "uint8",
    TensorProto.INT32: "int32",
    TensorProto.UINT64: "uint64",
    TensorProto.INT64: "int64",
}

_VISION_RE = re.compile(r"vision|siglip|patch_embed", re.IGNORECASE)
_PROJ_RE = re.compile(r"multi_modal_projector|/projector", re.IGNORECASE)


def _bucket(name: str) -> str:
    if _PROJ_RE.search(name):
        return "projector"
    if _VISION_RE.search(name):
        return "vision"
    return "llm"


def _init_bytes(init: TensorProto) -> int:
    n = 1
    for d in init.dims:
        n *= d
    return n * _DTYPE_BYTES.get(init.data_type, 0)


def _trace_weight_init(node, producer, initializers):
    """Return the weight initializer feeding a compute node, or ``None``.

    For MatMul/Gemm/Conv the weight is input[1]; it may be a direct initializer
    or reached through a shallow Transpose/Cast/Reshape — follow a few hops.
    """
    if len(node.input) < 2:
        return None
    name = node.input[1]
    for _ in range(4):
        if name in initializers:
            return initializers[name]
        prod = producer.get(name)
        if prod is None or not prod.input:
            return None
        name = prod.input[0]
    return None


def inspect(path: Path, top: int = 15) -> None:
    model = onnx.load(str(path), load_external_data=False)
    g = model.graph
    initializers = {i.name: i for i in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    # buckets[bucket]["int8"|"fp16w"|"fp16a"]["count"|"bytes"]
    #   int8  = quantized weight compute
    #   fp16w = fp16 compute that HAS a weight initializer (a Linear NOT quantized)
    #   fp16a = fp16 compute with NO weight (activation×activation BMM, correctly fp16)
    buckets: dict[str, dict[str, dict[str, int]]] = {}
    n_quant = sum(1 for n in g.node if n.op_type == "AscendQuant")
    n_dequant = sum(1 for n in g.node if n.op_type == "AscendDequant")

    def _b(bucket: str) -> dict[str, dict[str, int]]:
        return buckets.setdefault(
            bucket,
            {k: {"count": 0, "bytes": 0} for k in ("int8", "fp16w", "fp16a")},
        )

    for node in g.node:
        if node.op_type not in _COMPUTE_OPS:
            continue
        name = node.name or "<unnamed>"
        bucket = _bucket(name)

        fed_by_quant = any(
            producer.get(inp) is not None and producer[inp].op_type == "AscendQuant" for inp in node.input
        )
        winit = _trace_weight_init(node, producer, initializers)
        wtype = winit.data_type if winit is not None else None
        wbytes = _init_bytes(winit) if winit is not None else 0

        is_int8 = (
            node.op_type in ("MatMulInteger", "QLinearMatMul")
            or wtype in (TensorProto.INT8, TensorProto.UINT8)
            or fed_by_quant
        )
        if is_int8:
            kind = "int8"
        elif winit is not None:
            kind = "fp16w"  # has a weight but stayed fp16 -> a MISSED quant target
        else:
            kind = "fp16a"  # no weight (QK^T / AV) -> correctly fp16

        b = _b(bucket)
        b[kind]["count"] += 1
        b[kind]["bytes"] += wbytes

    # True in-graph data size = sum of EVERY initializer (weights + scales +
    # biases + embeddings). Comparing this to the .data file size reveals dead
    # or appended bytes that no TensorProto references.
    total_init_bytes = sum(_init_bytes(i) for i in g.initializer)

    # Backward reachability from graph outputs. A node is LIVE only if a path of
    # consumers leads from it to a graph output. The plain orphan check below is
    # too weak: it marks an initializer "live" if ANY node lists it as input —
    # even a node that is itself unreachable (e.g. an Identity left dangling
    # after the fp16 MatMul it fed was removed). Reachability finds those weights
    # that are referenced only by dead-but-still-present nodes.
    out_names = {o.name for o in g.output}
    live_tensors: set[str] = set(out_names)
    # Fixed-point: a node is live if any of its outputs is needed; then its
    # inputs become needed too.
    changed = True
    live_node_ids: set[int] = set()
    while changed:
        changed = False
        for n in g.node:
            if id(n) in live_node_ids:
                continue
            if any(o in live_tensors for o in n.output):
                live_node_ids.add(id(n))
                changed = True
                for i in n.input:
                    if i and i not in live_tensors:
                        live_tensors.add(i)
    unreachable_init_bytes = 0
    unreachable_by_consumer: dict[str, list[int]] = {}
    for init in g.initializer:
        if init.name in live_tensors:
            continue
        # Referenced by some node (so not a plain orphan) but unreachable.
        refs = [n.op_type for n in g.node if init.name in n.input]
        if not refs:
            continue  # plain orphan, already counted below
        unreachable_init_bytes += _init_bytes(init)
        key = refs[0]
        d = unreachable_by_consumer.setdefault(key, [0, 0])
        d[0] += 1
        d[1] += _init_bytes(init)
    dead_node_bytes = unreachable_init_bytes  # alias for clarity in print

    # Orphaned initializers = declared but referenced by NO node input and not a
    # graph output. These are dead weights the DCE failed to prune (e.g. the
    # fp16 originals of Linears that were replaced by int8). They still occupy
    # .data bytes and inflate the file even though nothing reads them.
    consumed: set[str] = {o.name for o in g.output}
    for n in g.node:
        consumed.update(n.input)
    dead_bytes = 0
    dead_by_bucket: dict[str, dict[str, int]] = {}
    for init in g.initializer:
        if init.name in consumed:
            continue
        b = _init_bytes(init)
        dead_bytes += b
        bk = _bucket(init.name)
        d = dead_by_bucket.setdefault(bk, {"count": 0, "bytes": 0})
        d["count"] += 1
        d["bytes"] += b

    data_file = path.with_name(path.name + ".data")
    data_size = data_file.stat().st_size if data_file.exists() else None

    print(f"\n=== {path.name} ===")
    print(f"AscendQuant nodes: {n_quant}   AscendDequant nodes: {n_dequant}")
    print(f"{'bucket':<10} {'int8 #':>7} {'int8 GB':>9} {'fp16w #':>8} {'fp16w GB':>9} {'fp16a #':>8}")
    print("-" * 56)
    tot = {"i8c": 0, "i8b": 0, "fwc": 0, "fwb": 0, "fac": 0}
    for name in ("vision", "projector", "llm", "other"):
        if name not in buckets:
            continue
        b = buckets[name]
        print(
            f"{name:<10} {b['int8']['count']:>7} {b['int8']['bytes'] / 1e9:>9.3f} "
            f"{b['fp16w']['count']:>8} {b['fp16w']['bytes'] / 1e9:>9.3f} "
            f"{b['fp16a']['count']:>8}"
        )
        tot["i8c"] += b["int8"]["count"]
        tot["i8b"] += b["int8"]["bytes"]
        tot["fwc"] += b["fp16w"]["count"]
        tot["fwb"] += b["fp16w"]["bytes"]
        tot["fac"] += b["fp16a"]["count"]
    print("-" * 56)
    print(
        f"{'TOTAL':<10} {tot['i8c']:>7} {tot['i8b'] / 1e9:>9.3f} "
        f"{tot['fwc']:>8} {tot['fwb'] / 1e9:>9.3f} {tot['fac']:>8}"
    )
    print(
        "  int8 = quantized | fp16w = HAS weight but still fp16 (MISSED) | fp16a = no weight (QK^T/AV, correctly fp16)"
    )
    print(f"\nall-initializer bytes (truth): {total_init_bytes / 1e9:.3f} GB")
    if data_size is not None:
        gap = data_size - total_init_bytes
        print(f".data file on disk:            {data_size / 1e9:.3f} GB")
        flag = "   ⚠ APPEND/DEAD-BYTE BLOAT" if gap > 0.05e9 else "   (ok)"
        print(f"dead/append bytes (disk-truth): {gap / 1e9:+.3f} GB{flag}")

    # Unreachable initializers — referenced ONLY by nodes that are themselves
    # unreachable from any graph output. These bytes are truly dead even though
    # the plain orphan check (below) considers them "in use".
    if dead_node_bytes > 0:
        print(
            f"\n⚠ UNREACHABLE-from-output initializers = {dead_node_bytes / 1e9:.3f} GB "
            f"(referenced only by dead nodes; the real DCE leak):"
        )
        for op, (cnt, byt) in sorted(unreachable_by_consumer.items(), key=lambda kv: -kv[1][1]):
            print(f"    consumed-by {op:<14} {cnt:>5} tensor(s)  {byt / 1e9:>7.3f} GB")
    else:
        print("\nall initializers are reachable from graph outputs (no dead-node leak).")

    # Orphaned (zero-reference) initializers — dead weights DCE missed.
    if dead_bytes > 0:
        print(
            f"\n⚠ ORPHANED initializers (referenced by NO node) = "
            f"{dead_bytes / 1e9:.3f} GB across {sum(d['count'] for d in dead_by_bucket.values())} tensor(s):"
        )
        for bk in ("vision", "projector", "llm", "other"):
            if bk in dead_by_bucket:
                d = dead_by_bucket[bk]
                print(
                    f"    {bk:<10} {d['count']:>5} tensor(s)  {d['bytes'] / 1e9:>7.3f} GB  (dead fp16/weights not pruned)"
                )
    else:
        print("\nno orphaned initializers (DCE clean).")

    # Top-N largest initializers with dtype + consumer op-types. The decisive
    # view: if fp16 weights of *quantized* Linears still appear here (consumed by
    # AscendQuant or a dangling Transpose) they are duplicated dead weight that
    # DCE could not see because a live node still references them.
    consumers: dict[str, list[str]] = {}
    for n in g.node:
        for inp in n.input:
            if inp in initializers:
                consumers.setdefault(inp, []).append(n.op_type)
    ranked = sorted(g.initializer, key=_init_bytes, reverse=True)[:top]
    print(f"\ntop {len(ranked)} largest initializer(s):")
    print(f"  {'GB':>7}  {'dtype':>6}  {'consumed-by(op:count)':<34}  name")
    for init in ranked:
        cons = consumers.get(init.name, [])
        if cons:
            from collections import Counter

            csum = ", ".join(f"{op}:{c}" for op, c in Counter(cons).items())
        else:
            csum = "<NONE — orphan>"
        dt = _DTYPE_NAME.get(init.data_type, str(init.data_type))
        print(f"  {_init_bytes(init) / 1e9:>7.3f}  {dt:>6}  {csum:<34}  {init.name[:70]}")

    # Decisive view: total initializer bytes grouped by (first consumer op-type,
    # dtype). Shows exactly where the bytes live — e.g. fp16 under "Gather"
    # (embedding) vs under "MatMul" (Linear weight) vs under "AscendQuant"
    # (runtime-quant weight duplicate).
    from collections import defaultdict

    agg: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])  # (op,dtype)->[count,bytes]
    for init in g.initializer:
        cons = consumers.get(init.name, [])
        op = cons[0] if cons else "<orphan>"
        dt = _DTYPE_NAME.get(init.data_type, str(init.data_type))
        agg[(op, dt)][0] += 1
        agg[(op, dt)][1] += _init_bytes(init)
    print("\ninitializer bytes by (first-consumer-op, dtype):")
    print(f"  {'op':<16} {'dtype':>6} {'count':>7} {'GB':>9}")
    for (op, dt), (cnt, byt) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {op:<16} {dt:>6} {cnt:>7} {byt / 1e9:>9.3f}")

    # Decisive fate-tracing: for every fp16 *weight* initializer in the vision
    # tower, walk downstream through pass-through ops (Identity/Cast/Transpose/
    # Reshape) until a compute node, and classify what consumes it:
    #   to_int8 = feeds ONLY quantized compute  -> dead FakeQuant source (prune!)
    #   to_fp16 = feeds ONLY fp16 compute        -> a MISSED quant target
    #   to_both = feeds BOTH                     -> shared weight, partial quant
    # This distinguishes "fp16 source not pruned" from "vision only half quantized".
    node_consumers: dict[str, list] = {}
    for n in g.node:
        for inp in n.input:
            node_consumers.setdefault(inp, []).append(n)

    def _is_int8_node(n) -> bool:
        if n.op_type in ("MatMulInteger", "QLinearMatMul"):
            return True
        if any(producer.get(i) is not None and producer[i].op_type == "AscendQuant" for i in n.input):
            return True
        for i in n.input[1:]:
            ini = initializers.get(i)
            if ini is not None and ini.data_type in (TensorProto.INT8, TensorProto.UINT8):
                return True
        return False

    pass_through = {"Identity", "Cast", "Transpose", "Reshape", "Squeeze", "Unsqueeze"}
    fate = {k: [0, 0] for k in ("to_int8", "to_fp16", "to_both", "to_none")}
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT16:
            continue
        wb = _init_bytes(init)
        if wb < 1_000_000:  # skip bias/norm; keep real weight tensors
            continue
        # Skip the token embedding (consumed by Gather, not a Linear weight).
        direct = node_consumers.get(init.name, [])
        if direct and all(n.op_type == "Gather" for n in direct):
            continue
        frontier = [init.name]
        seen: set[str] = set()
        hit_int8 = hit_fp16 = False
        while frontier:
            t = frontier.pop()
            if t in seen:
                continue
            seen.add(t)
            for n in node_consumers.get(t, []):
                if n.op_type in _COMPUTE_OPS:
                    if _is_int8_node(n):
                        hit_int8 = True
                    else:
                        hit_fp16 = True
                elif n.op_type in pass_through:
                    frontier.extend(o for o in n.output if o)
        key = "to_both" if hit_int8 and hit_fp16 else "to_int8" if hit_int8 else "to_fp16" if hit_fp16 else "to_none"
        fate[key][0] += 1
        fate[key][1] += wb
    print("\nfp16-weight fate (>1MB fp16 weights, excl. embedding; what consumes each):")
    for k, label in (
        ("to_int8", "feeds ONLY int8  -> DEAD FakeQuant source (prunable)"),
        ("to_fp16", "feeds ONLY fp16  -> NOT quantized (fp16 Linear)"),
        ("to_both", "feeds BOTH       -> shared weight, partial quant"),
        ("to_none", "feeds no compute -> dangling pass-through chain"),
    ):
        c, by = fate[k]
        if c:
            print(f"    {label:<52} {c:>4} tensor(s)  {by / 1e9:>6.3f} GB")

    # Per-role split: classify every vision compute node by its Linear role
    # (q/k/v/o_proj, fc1, fc2, patch_embed, attn-BMM) and whether it is int8 or
    # fp16. This reveals exactly WHICH SigLIP Linears msModelSlim quantized vs
    # skipped — the pattern (e.g. all fc1/fc2 int8 but all *_proj fp16, or
    # biased vs bias-less) tells us why it only did half.
    role_pat = [
        ("q_proj", re.compile(r"q_proj", re.I)),
        ("k_proj", re.compile(r"k_proj", re.I)),
        ("v_proj", re.compile(r"v_proj", re.I)),
        ("out_proj", re.compile(r"out_proj|o_proj", re.I)),
        ("fc1", re.compile(r"fc1", re.I)),
        ("fc2", re.compile(r"fc2", re.I)),
        ("patch_embed", re.compile(r"patch_embed", re.I)),
        ("attn_bmm", re.compile(r"self_attn(_\d+)?/MatMul", re.I)),
    ]

    def _role(name: str) -> str:
        for r, pat in role_pat:
            if pat.search(name):
                return r
        return "other"

    role_split: dict[str, list[int]] = {r: [0, 0] for r, _ in role_pat}
    role_split["other"] = [0, 0]
    for node in g.node:
        if node.op_type not in _COMPUTE_OPS or _bucket(node.name) != "vision":
            continue
        r = _role(node.name)
        if _is_int8_node(node):
            role_split[r][0] += 1
        else:
            role_split[r][1] += 1
    print("\nvision compute nodes by role (int8 / fp16):")
    print(f"  {'role':<14} {'int8':>5} {'fp16':>5}")
    for r in [rp[0] for rp in role_pat] + ["other"]:
        i8, f16 = role_split[r]
        if i8 or f16:
            mark = "  <- SKIPPED by quantizer" if f16 and not i8 else ""
            print(f"  {r:<14} {i8:>5} {f16:>5}{mark}")

    # Shared-weight probe: does each int8 weight (named "<stem>_quantized") have
    # a sibling fp16 MatMul still consuming the ORIGINAL "<stem>" weight? If yes,
    # the vision tower was invoked twice (e.g. one pass per camera) sharing
    # weights, and msModelSlim quantized only ONE of the two MatMuls — leaving
    # the other fp16 and pinning the original weight alive. That is the exact
    # cause of int8+fp16 coexistence and the file growth.
    int8_weight_stems: set[str] = set()
    for init in g.initializer:
        if init.data_type == TensorProto.INT8 and init.name.endswith("_quantized"):
            int8_weight_stems.add(init.name[: -len("_quantized")])
    fp16_init_names = {i.name for i in g.initializer if i.data_type == TensorProto.FLOAT16}
    shared = sorted(int8_weight_stems & fp16_init_names)
    if shared:
        # How many fp16 MatMuls consume each shared original weight?
        still_fp16 = 0
        for stem in shared:
            for n in node_consumers.get(stem, []):
                if n.op_type in _COMPUTE_OPS and not _is_int8_node(n):
                    still_fp16 += 1
        print(
            f"\n⚠ SHARED-WEIGHT duplication: {len(shared)} weight(s) exist as BOTH "
            f"'<stem>' (fp16) and '<stem>_quantized' (int8),\n"
            f"  with {still_fp16} fp16 MatMul(s) still using the original. "
            "=> vision tower invoked >1x (doublecam?), only one copy quantized."
        )
    else:
        print("\nno shared-weight int8/fp16 duplication (int8 weights have no fp16 same-stem sibling).")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx_paths", nargs="+", help="One or more ONNX files to inventory.")
    p.add_argument("--top", type=int, default=15, help="How many largest initializers to list per file.")
    args = p.parse_args()
    for raw in args.onnx_paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            print(f"!! not found: {path}")
            continue
        inspect(path, top=args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
