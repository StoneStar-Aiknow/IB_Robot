"""Inference-only PTV3: hybrid CPE and a continuous NPU FP16 transformer tail.

Load CANN's set_env.sh before use. Checkpoint keys and the point/output contract
match ptv3_vanilla. Construct, load_state_dict(strict=True), eval(), then forward;
do not move/cast the whole model, which intentionally has mixed placement.
A deployment that encodes one point cloud with several of these models per
request -- GraspGenX runs a generator and a discriminator encoder -- should set
``model.request_scope = ptv3_ascend.shared_geometry`` so the coordinate-derived
work happens once; see shared_geometry below.
No flash-attn package, RPE, training or FP32 feature/upcast path is provided. LayerNorm's
unused mean/rstd statistics can be FP32. PointTransformerV3AttentionOnly is the
previous CPU-FFN control; fusion switches below affect only new resident models.
"""

import math
import threading
from contextlib import contextmanager

import torch
import torch_npu  # noqa: F401
from graspgenx.models.ptv3.ptv3_vanilla import (
    HashSparseConv3d,
    PointTransformerV3Vanilla,
    VanillaPoint,  # noqa: F401 - re-export for the stage profiler
    VanillaPointModule,
    VanillaSerializedAttention,
    VanillaSerializedPooling,
    offset2bincount,
    segment_csr_vanilla,  # noqa: F401 - re-export for the stage profiler
)

NPU_DEVICE = "npu:0"
NPU_JIT_COMPILE = False
FUSE_ADD_LAYER_NORM = True
FUSE_FFN = True

# One GraspGenX request encodes the same object point cloud twice, once for the
# generator and once for the discriminator, with different weights. Everything
# derived from the coordinates alone -- serialization, the CPE neighbour maps,
# the pooling clusters and the stem's neighbour gather -- is therefore computed
# twice, and on Ascend310P1 that work is CPU bound.
#
# Sharing it requires the caller to open a request scope, which keeps the reuse
# where it belongs: nothing is shared between unrelated forwards, and the scope
# clears on exit, so replaying one point cloud cannot make a later request look
# faster than a deployment that sees a new cloud every time. Entries are also
# verified by content, so a mismatch recomputes instead of returning the wrong
# geometry. Concurrent requests in one process may share across scopes; the
# content check keeps that correct, it only blurs attribution.
# The scope is per thread rather than per process. Sharing is only ever wanted
# between the two encoders of one request, which run on one thread, while a
# module-level depth counter would be read-modify-written by any concurrent
# request: losing one decrement leaves the depth above zero for good, so the
# cache never clears and grows for the life of the process. Entries from a
# concurrent request were already harmless -- `_geometry_take` compares the
# identity tensors by content -- so what this changes is the leak, not
# correctness.
_SCOPE = threading.local()


def _scope_state() -> tuple[dict, int]:
    cache = getattr(_SCOPE, "cache", None)
    if cache is None:
        cache = _SCOPE.cache = {}
        _SCOPE.depth = 0
    return cache, _SCOPE.depth


@contextmanager
def shared_geometry():
    """Compute coordinate-derived work once for the encoders inside the scope."""
    cache, depth = _scope_state()
    _SCOPE.depth = depth + 1
    try:
        yield
    finally:
        _SCOPE.depth -= 1
        if _SCOPE.depth == 0:
            cache.clear()


def _geometry_enabled(shuffle: bool) -> bool:
    # Shuffled serialization draws a fresh permutation per forward, so sharing
    # it would both repeat one draw and skip the other's RNG consumption.
    _, depth = _scope_state()
    return depth > 0 and not shuffle and not torch.is_grad_enabled()


def _geometry_take(signature, identity, shuffle: bool = False):
    """Return geometry already computed for identical inputs in this scope."""
    if not _geometry_enabled(shuffle):
        return None
    cache, _ = _scope_state()
    entry = cache.get(signature)
    if entry is None:
        return None
    stored, value = entry
    if len(stored) != len(identity) or not all(
        current.shape == previous.shape and current.dtype == previous.dtype and torch.equal(current, previous)
        for current, previous in zip(identity, stored, strict=True)
    ):
        return None
    return value


def _geometry_put(signature, identity, value, shuffle: bool = False):
    """Offer geometry to the other encoders of this request."""
    if _geometry_enabled(shuffle):
        cache, _ = _scope_state()
        # Clone the identity so later in-place work cannot forge a match.
        cache[signature] = (
            tuple(tensor.clone() for tensor in identity),
            value,
        )
    return value


class AscendSerializedAttention(VanillaSerializedAttention):
    """Reuse CPU padding/index construction, but compute attention only in half."""

    def __init__(self, source):
        if source.enable_rpe or source.enable_flash:
            raise ValueError("Ascend attention does not support RPE or Flash")
        super().__init__(
            channels=source.channels,
            num_heads=source.num_heads,
            patch_size=source.patch_size_max,
            qkv_bias=source.qkv.bias is not None,
            qk_scale=source.scale,
            order_index=source.order_index,
            upcast_attention=False,
            upcast_softmax=False,
        )
        self.load_state_dict(source.state_dict(), strict=True)
        self.to(device=NPU_DEVICE, dtype=torch.float16)
        self.register_buffer("scaled_qkv_weight", None, persistent=False)
        self.register_buffer("scaled_qkv_bias", None, persistent=False)
        self.register_load_state_dict_post_hook(self._pack_q_scale)
        self._pack_q_scale(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_q_scale(module, _incompatible):
        module.scaled_qkv_weight = module.qkv.weight.clone()
        module.scaled_qkv_weight[: module.channels].mul_(module.scale)
        module.scaled_qkv_bias = None if module.qkv.bias is None else module.qkv.bias.clone()
        if module.scaled_qkv_bias is not None:
            module.scaled_qkv_bias[: module.channels].mul_(module.scale)

    def prepare_indices(self, point):
        count = point.feat.shape[0]
        single_batch = point.offset.numel() == 1
        self.patch_size = min(
            count if single_batch else offset2bincount(point.offset).min().item(),
            self.patch_size_max,
        )
        # A full-cloud patch without RPE is permutation equivariant. Keep row
        # order unchanged instead of gathering and immediately undoing a sort.
        if single_batch and 0 < count <= self.patch_size_max:
            return None, None
        # For a single cloud with complete patches, pad/unpad are identities.
        # Use the current point's views, not a cache shared across point clouds.
        if single_batch and self.patch_size > 0 and count % self.patch_size == 0:
            return (
                point.serialized_order[self.order_index],
                point.serialized_inverse[self.order_index],
            )
        pad, unpad, _ = self.get_padding_and_inverse(point)
        return (
            point.serialized_order[self.order_index][pad],
            unpad[point.serialized_inverse[self.order_index]],
        )

    def forward(self, point):
        if point.feat.device.type != "cpu" or point.feat.dtype != torch.float32:
            raise RuntimeError("The attention-only surrounding encoder must be CPU FP32")
        order, inverse = self.prepare_indices(point)
        features = point.feat.to(device=self.qkv.weight.device, dtype=torch.float16)
        point.feat = self.forward_features(features, order, inverse).to(device="cpu", dtype=torch.float32)
        return point

    def forward_features(self, features, order, inverse):
        if self.training:
            raise RuntimeError("Ascend PTV3 is inference-only; call eval() first")
        if features.device.type != "npu" or features.dtype != torch.float16:
            raise RuntimeError("Attention features must remain NPU FP16")
        for parameter in (self.qkv.weight, self.qkv.bias, self.proj.weight, self.proj.bias):
            if parameter is not None and (parameter.device.type != "npu" or parameter.dtype != torch.float16):
                raise RuntimeError("Attention parameters must remain NPU FP16")

        h, k, c = self.num_heads, self.patch_size, self.channels

        device = self.qkv.weight.device
        qkv = torch.nn.functional.linear(features, self.scaled_qkv_weight, self.scaled_qkv_bias)
        if order is not None:
            order = order.to(device=device, dtype=torch.int32)
            inverse = inverse.to(device=device, dtype=torch.int32)
            qkv = qkv.index_select(0, order)
        if c // h % 16 == 0 and c // h <= 512 and h <= 256 and 1 <= k <= 65535 and qkv.shape[0] // k <= 128:
            q, key, value = qkv.reshape(-1, k, 3, c).unbind(dim=2)
            q, key, value = q.contiguous(), key.contiguous(), value.contiguous()
            features = torch_npu.npu_prompt_flash_attention(
                q,
                key,
                value,
                num_heads=h,
                input_layout="BSH",
                scale_value=1.0,
                pre_tokens=2147483647,
                next_tokens=2147483647,
            ).reshape(-1, c)
        else:
            q, key, value = qkv.reshape(-1, k, 3, h, c // h).permute(2, 0, 3, 1, 4).unbind(0)
            q, key, value = q.contiguous(), key.contiguous(), value.contiguous()
            features = (self.softmax(q @ key.transpose(-2, -1)) @ value).transpose(1, 2).reshape(-1, c)
        if inverse is not None:
            features = features.index_select(0, inverse)
        features = self.proj(features)
        return features


class PointTransformerV3AttentionOnly(PointTransformerV3Vanilla):
    """Previous attention-only implementation, retained as the experiment control."""

    def __init__(self, **kwargs):
        for option in ("enable_flash", "enable_rpe", "upcast_attention", "upcast_softmax"):
            if kwargs.get(option, False):
                raise ValueError(f"Ascend PTV3 does not support {option}=True")
            kwargs[option] = False
        torch.npu.set_device(NPU_DEVICE)
        torch.npu.set_compile_mode(jit_compile=NPU_JIT_COMPILE)
        super().__init__(**kwargs)
        self.execution_config = {
            "dense_resident": False,
            "attention_dtype": "fp16",
            "jit_compile": NPU_JIT_COMPILE,
        }
        for stage in self.enc:
            for block in stage.children():
                if hasattr(block, "attn"):
                    block.attn = AscendSerializedAttention(block.attn)


class AscendFFN(torch.nn.Module):
    """FP16 FFN with the original checkpoint keys and optional CANN FFN fusion."""

    def __init__(self, source):
        super().__init__()
        self.fc1 = source.fc1.to(device=NPU_DEVICE, dtype=torch.float16)
        self.fc2 = source.fc2.to(device=NPU_DEVICE, dtype=torch.float16)
        self.act = source.act
        self.use_fused = FUSE_FFN
        self.register_buffer("weight1", None, persistent=False)
        self.register_buffer("weight2", None, persistent=False)
        self.register_load_state_dict_post_hook(self._pack_weights)
        self._pack_weights(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_weights(module, _incompatible):
        # FFN expects [C, 4C] and [4C, C], unlike nn.Linear's transposed layout.
        # Repack after checkpoint reloads; never transpose weights per request.
        if module.use_fused:
            module.weight1 = module.fc1.weight.T.contiguous()
            module.weight2 = module.fc2.weight.T.contiguous()

    def forward(self, features):
        if self.training or features.dtype != torch.float16 or features.device.type != "npu":
            raise RuntimeError("FFN requires eval mode and NPU FP16 features")
        if self.use_fused:
            return torch_npu.npu_ffn(
                features,
                self.weight1,
                self.weight2,
                "gelu",
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                inner_precise=1,
            )
        return self.fc2(self.act(self.fc1(features)))


class AscendBlock(VanillaPointModule):
    """FP16 CPE and dense region, resident across consecutive blocks.

    A block uploads float32 CPU features as float16, runs CPE, attention and
    FFN on the device, and hands float32 CPU features back. When the next
    module is another such block, that round trip is pure overhead: float16 to
    float32 and back is lossless, so ``resident`` keeps the features on the
    device and the results are bitwise identical. It also removes the
    device-to-host copy, which is a synchronization point: a request keeps the
    device busy for 78 of its 325 ms, and the host waits at every one of those
    boundaries for the queue to drain.
    """

    def __init__(self, source):
        super().__init__()
        if not source.pre_norm:
            raise ValueError("The resident Ascend block requires pre_norm=True")
        self.channels = source.channels
        self.pre_norm = source.pre_norm
        for name, module in source.named_children():
            self.add_module(name, module)
        self.cpe_conv = SubMCPEConv(source.cpe_conv)
        self.norm1.to(device=NPU_DEVICE, dtype=torch.float16)
        self.norm2.to(device=NPU_DEVICE, dtype=torch.float16)
        self.mlp = AscendFFN(source.mlp)
        self.fuse_add_norm = FUSE_ADD_LAYER_NORM
        # Set by the encoder once the stage's children are known.
        self.resident = False
        for name in (
            "cpe_linear_weight_npu",
            "cpe_linear_bias_npu",
            "cpe_norm_weight_npu",
            "cpe_norm_bias_npu",
            "cpe_projected_weight_npu",
            "cpe_projected_bias_npu",
        ):
            self.register_buffer(name, None, persistent=False)
        self.register_load_state_dict_post_hook(self._pack_cpe_weights)
        self._pack_cpe_weights(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_cpe_weights(module, _incompatible):
        if getattr(module.cpe_conv, "_supported_shape", False):
            for name, value in (
                ("cpe_linear_weight_npu", module.cpe_linear.weight),
                ("cpe_linear_bias_npu", module.cpe_linear.bias),
                ("cpe_norm_weight_npu", module.cpe_norm.weight),
                ("cpe_norm_bias_npu", module.cpe_norm.bias),
            ):
                setattr(module, name, None if value is None else value.to(NPU_DEVICE, torch.float16).contiguous())
            linear = module.cpe_linear.weight
            weight = module.cpe_conv.weight @ linear.T
            bias = module.cpe_linear.bias
            if module.cpe_conv.bias is not None:
                bias = torch.nn.functional.linear(module.cpe_conv.bias, linear, bias)
            module.cpe_projected_weight_npu = weight.to(NPU_DEVICE, torch.float16).contiguous()
            module.cpe_projected_bias_npu = None if bias is None else bias.to(NPU_DEVICE, torch.float16)

    def forward_attention(self, point):
        if self.training:
            raise RuntimeError("Ascend PTV3 is inference-only; call eval() first")
        cpu_input = point.feat.device.type == "cpu" and point.feat.dtype == torch.float32
        if point.feat.device.type == "npu" and point.feat.dtype == torch.float16:
            if point.feat.device != self.norm1.weight.device:
                raise RuntimeError("CPE features must share the attention NPU device")
        elif not cpu_input:
            raise RuntimeError("CPE must produce CPU FP32 or NPU FP16 features")
        order, inverse = self.attn.prepare_indices(point)
        residual = point.feat.to(device=NPU_DEVICE, dtype=torch.float16) if cpu_input else point.feat
        normalized = point.pop("_cpe_normalized", None)
        if normalized is None:
            normalized = torch_npu.npu_layer_norm_eval(
                residual,
                self.norm1.normalized_shape,
                self.norm1.weight,
                self.norm1.bias,
                self.norm1.eps,
            )
        point.feat = self.attn.forward_features(normalized, order, inverse)
        point["_attention_residual"] = residual
        return point

    def forward_ffn(self, point):
        residual = point.pop("_attention_residual")
        if self.fuse_add_norm:
            # y and residual_sum are FP16. FP32 mean/rstd are unused statistics,
            # not an upcast feature path between attention and FFN.
            normalized, _, _, residual = torch_npu.npu_add_layer_norm(
                residual,
                point.feat,
                self.norm2.weight,
                self.norm2.bias,
                epsilon=self.norm2.eps,
                additional_output=True,
            )
        else:
            residual = residual + point.feat
            normalized = torch_npu.npu_layer_norm_eval(
                residual,
                self.norm2.normalized_shape,
                self.norm2.weight,
                self.norm2.bias,
                self.norm2.eps,
            )
        output = self.mlp(normalized)
        if output.dtype != torch.float16 or residual.dtype != torch.float16:
            raise RuntimeError("FFN output and residual must remain FP16")
        features = residual + output
        # Hand float16 straight to the next block; only the module that leaves
        # the dense region needs CPU float32.
        point.feat = features if self.resident else features.to(device="cpu", dtype=torch.float32)
        return point

    def forward_cpe(self, point):
        uploaded = point.feat.device.type == "npu" and point.feat.dtype == torch.float16
        if self.training or not (uploaded or (point.feat.device.type == "cpu" and point.feat.dtype == torch.float32)):
            raise RuntimeError("CPE block requires eval mode and CPU FP32 or resident NPU FP16 features")
        if not getattr(self.cpe_conv, "_supported_shape", False) or not 1 <= point.feat.shape[0] <= 4096:
            if uploaded:
                raise RuntimeError("The reference CPE path requires CPU FP32 features")
            cpe = self.cpe_conv(point.feat, point.grid_coord, point.batch, point)
            point.feat = point.feat + self.cpe_norm(self.cpe_linear(cpe))
            return point
        residual = point.feat if uploaded else point.feat.to(NPU_DEVICE, torch.float16)
        cpe = self.cpe_conv(
            residual,
            point.grid_coord,
            point.batch,
            point,
            projected_weight=self.cpe_projected_weight_npu,
            projected_bias=self.cpe_projected_bias_npu,
        )
        cpe = torch_npu.npu_layer_norm_eval(
            cpe, self.cpe_norm.normalized_shape, self.cpe_norm_weight_npu, self.cpe_norm_bias_npu, self.cpe_norm.eps
        )
        normalized, _, _, point.feat = torch_npu.npu_add_layer_norm(
            residual,
            cpe,
            self.norm1.weight,
            self.norm1.bias,
            epsilon=self.norm1.eps,
            additional_output=True,
        )
        point["_cpe_normalized"] = normalized
        return point

    def forward(self, point):
        point = self.forward_cpe(point)
        point = self.forward_attention(point)
        return self.forward_ffn(point)


class PointTransformerV3Ascend(PointTransformerV3AttentionOnly):
    """Hybrid SubM CPE with resident FP16 attention, residual/norm and FFN."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_config.update(
            dense_resident=True,
            norm_residual_ffn_dtype="fp16",
            fuse_add_layer_norm=FUSE_ADD_LAYER_NORM,
            fuse_ffn=FUSE_FFN,
            ffn_inner_precise=1 if FUSE_FFN else None,
            cpe_map_cache="per_point_per_forward",
            cpe_compute="npu_fp16_supported_shapes",
            cpe_map="cpu_representatives_npu_query",
            cpe_post_ops="npu_fp16_supported_shapes",
            serialization_compute="cpu_grid_depth_batch_sort_npu_spatial_codes",
            embedding_compute="cpu_fp32_additive_hash",
            pooling_compute="cpu_fp32_segment_reduce",
            block_features_resident_within_stage=True,
            attention_single_patch_no_permutation=True,
            attention_q_scale_packed=True,
            cpe_norm1_fused=True,
            cpe_projection_folded="cpu_fp32_then_fp16",
            attention_compute="pfa_bsh_with_fp16_shape_fallback",
            pooling_norm_fold="positive_eval_affine",
            pooling_torchscript=True,
            stage0_map_prefetch="canonical_n_ge_1024",
            geometry_sharing="per_request_scope_between_encoders",
        )
        self.embedding.conv = AscendEmbeddingConv(self.embedding.conv)
        for stage in self.enc:
            for name, block in list(stage.named_children()):
                if hasattr(block, "attn"):
                    setattr(stage, name, AscendBlock(block))
                elif isinstance(block, VanillaSerializedPooling):
                    setattr(stage, name, AscendSerializedPooling(block))
            # Consecutive blocks exchange device features; the last one in a
            # stage feeds pooling or the pooled read-out, which need CPU float32.
            children = list(stage.children())
            for index, block in enumerate(children[:-1]):
                if isinstance(block, AscendBlock) and isinstance(children[index + 1], AscendBlock):
                    block.resident = True

    @torch.inference_mode()
    def serialize_point(self, data_dict):
        offset = data_dict.get("offset")
        if (
            "batch" not in data_dict
            and "feat" in data_dict
            and isinstance(offset, torch.Tensor)
            and offset.device.type == "cpu"
            and offset.dtype == torch.int64
            and offset.shape == (1,)
            and offset.item() == len(data_dict["feat"])
        ):
            data_dict = dict(data_dict, batch=torch.zeros(len(data_dict["feat"]), dtype=torch.long))
        point = VanillaPoint(data_dict)
        if "grid_coord" not in point:
            assert {"grid_size", "coord"}.issubset(point.keys())
            point.grid_coord = torch.div(
                point.coord - point.coord.min(0)[0], point.grid_size, rounding_mode="trunc"
            ).int()
        grid = point.grid_coord
        depth = int(grid.max()).bit_length()
        orders = ("z", "z-trans", "hilbert", "hilbert-trans")
        if not (
            tuple(self.order) == orders
            and grid.device.type == "cpu"
            and grid.dtype in (torch.int32, torch.int64)
            and grid.shape == (len(point.batch), 3)
            and 1 <= len(grid) <= 4096
            and 1 <= depth <= 16
            and grid.min() >= 0
        ):
            # Preserve reference handling of non-default orders and out-of-domain
            # inputs, including its depth-zero failure. Never catch NPU errors.
            point.serialization(order=self.order, depth=depth, shuffle_orders=self.shuffle_orders)
            return point
        assert depth * 3 + len(point.offset).bit_length() <= 63
        signature = ("serialize", len(grid), depth, orders, self.shuffle_orders)
        identity = (grid, point.batch)
        shared = _geometry_take(signature, identity, self.shuffle_orders)
        if shared is not None:
            code, order, inverse = shared
        else:
            from torch_models.graspgenx_ascend_310p.custom_ops.grid_encode import grid_encode

            spatial = grid_encode(grid.to(NPU_DEVICE, torch.int32).contiguous(), depth)
            code = spatial.cpu().T.contiguous() | (point.batch.long() << (depth * 3))
            order = torch.argsort(code)
            inverse = torch.zeros_like(order).scatter_(1, order, torch.arange(code.shape[1]).repeat(code.shape[0], 1))
            if self.shuffle_orders:
                permutation = torch.randperm(code.shape[0])
                code, order, inverse = code[permutation], order[permutation], inverse[permutation]
            _geometry_put(signature, identity, (code, order, inverse), self.shuffle_orders)
        point.update(serialized_depth=depth, serialized_code=code, serialized_order=order, serialized_inverse=inverse)
        # Queue the first CPE query before the independent CPU stem computation.
        stage = getattr(getattr(self, "enc", None), "enc0", None)
        conv = getattr(getattr(stage, "block0", None), "cpe_conv", None)
        if len(grid) >= 1024 and getattr(conv, "_supported_shape", False):
            conv._get_npu_map(point.grid_coord, point.batch, point)
        return point

    def forward(self, data_dict):
        point = self.serialize_point(data_dict)
        point = self.embedding(point)
        point = self.enc(point)
        pooled = self.pool_features(point)
        return self.projection(pooled)

    @staticmethod
    def pool_features(point):
        return torch.segment_reduce(point.feat, "mean", offsets=torch.nn.functional.pad(point.offset, (1, 0)))


@torch.jit.script
def _pool_geometry(
    coord: torch.Tensor, grid: torch.Tensor, batch: torch.Tensor, codes: torch.Tensor, depth: int, shuffle: bool
):
    """Cluster, reorder and pool everything that depends only on coordinates."""
    code = codes >> (depth * 3)
    _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
    _, indices = torch.sort(cluster)
    indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    heads = indices[indptr[:-1]]
    coordinates = torch.segment_reduce(coord[indices], "mean", offsets=indptr)
    code = code[:, heads]
    order = torch.argsort(code)
    inverse = torch.zeros_like(order).scatter_(
        1, order, torch.arange(code.shape[1], device=order.device).repeat(code.shape[0], 1)
    )
    if shuffle:
        permutation = torch.randperm(code.shape[0])
        code, order, inverse = code[permutation], order[permutation], inverse[permutation]
    return (indices, indptr, coordinates, grid[heads] >> depth, batch[heads], code, order, inverse, cluster)


@torch.jit.script
def _pool_features(
    feat: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    reduce: str,
):
    projected = torch.nn.functional.linear(feat, weight, bias)
    return torch.segment_reduce(projected[indices], reduce, offsets=indptr)


class AscendSerializedPooling(VanillaSerializedPooling):
    """Reference CPU FP32 pooling with native CSR reductions, not expanded IDs."""

    def __init__(self, source):
        torch.nn.Module.__init__(self)
        for name in ("in_channels", "out_channels", "stride", "reduce", "shuffle_orders", "traceable"):
            setattr(self, name, getattr(source, name))
        self.proj, self.norm, self.act = source.proj, source.norm, source.act
        self.register_buffer("fused_proj_weight", None, persistent=False)
        self.register_buffer("fused_proj_bias", None, persistent=False)
        self.register_load_state_dict_post_hook(self._pack_norm)
        self._pack_norm(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_norm(module, _incompatible):
        module.fused_proj_weight = module.fused_proj_bias = None
        norm = module.norm
        if not (
            module.reduce in ("max", "min")
            and isinstance(norm, torch.nn.BatchNorm1d)
            and norm.affine
            and norm.track_running_stats
        ):
            return
        scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
        # Only increasing affine transforms commute with both max and min.
        if not bool((scale >= 0).all()):
            return
        module.fused_proj_weight = module.proj.weight * scale[:, None]
        bias = module.proj.bias if module.proj.bias is not None else 0
        module.fused_proj_bias = (bias - norm.running_mean) * scale + norm.bias

    def forward(self, point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        if self.training or torch.is_grad_enabled() or (self.norm is not None and self.norm.training):
            # Training/optimizer paths retire the fold until a checkpoint reload.
            self.fused_proj_weight = self.fused_proj_bias = None
        fused_norm = self.fused_proj_weight is not None
        # Distinct point counts give each pooling level its own entry, so the
        # four levels of one encoder never evict each other.
        signature = ("pooling", len(point.feat), pooling_depth, self.shuffle_orders)
        identity = (point.coord, point.grid_coord, point.batch, point.serialized_code)
        geometry = _geometry_take(signature, identity, self.shuffle_orders)
        if geometry is None:
            geometry = _geometry_put(
                signature,
                identity,
                _pool_geometry(
                    point.coord,
                    point.grid_coord,
                    point.batch,
                    point.serialized_code,
                    pooling_depth,
                    self.shuffle_orders,
                ),
                self.shuffle_orders,
            )
        indices, indptr, coordinates, grid, batch, code, order, inverse, cluster = geometry
        features = _pool_features(
            point.feat,
            self.fused_proj_weight if fused_norm else self.proj.weight,
            self.fused_proj_bias if fused_norm else self.proj.bias,
            indices,
            indptr,
            self.reduce,
        )
        result = dict(
            feat=features,
            coord=coordinates,
            grid_coord=grid,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=batch,
        )
        for key in ("condition", "context"):
            if key in point:
                result[key] = point[key]
        if self.traceable:
            result.update(pooling_inverse=cluster, pooling_parent=point)
        point = VanillaPoint(result)
        if self.norm is not None and not fused_norm:
            point.feat = self.norm(point.feat)
        if self.act is not None:
            point.feat = self.act(point.feat)
        return point


class CachedCPEConv(HashSparseConv3d):
    """CPU CPE with a point-local map shared by blocks with identical geometry."""

    def __init__(self, source):
        # Reuse parameters/buffers: wrapping must not change keys or consume RNG.
        torch.nn.Module.__init__(self)
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size
        self.weight = source.weight
        self.bias = source.bias
        self.register_buffer("offsets", source.offsets)

    def _get_neighbor_map(self, grid_coord, batch, point):
        n, volume = grid_coord.shape[0], self.offsets.shape[0]
        cache_key = f"_cpe_hash_map_{self.kernel_size}"
        if cache_key not in point:
            # Preserve vanilla's sort/searchsorted tie-breaking, including duplicate
            # voxels and hash collisions. Geometry stays fixed within this stage.
            sorted_keys, sort_idx = self._hash(batch, grid_coord).sort()
            neighbor_coords = grid_coord.unsqueeze(1) + self.offsets.unsqueeze(0)
            neighbor_batch = batch.unsqueeze(1).expand(-1, volume)
            keys = self._hash(neighbor_batch.reshape(-1), neighbor_coords.reshape(-1, 3))
            positions = torch.searchsorted(sorted_keys, keys).clamp(max=n - 1)
            point[cache_key] = {
                "indices": sort_idx[positions],
                "found": sorted_keys[positions] == keys,
            }
        return point[cache_key]

    def forward(self, feat, grid_coord, batch, point):
        if self.training or feat.device.type != "cpu" or feat.dtype != torch.float32:
            raise RuntimeError("Cached CPE requires eval mode and CPU FP32 features")
        cache = self._get_neighbor_map(grid_coord, batch, point)
        neighbors = feat[cache["indices"]] * cache["found"].unsqueeze(-1)
        output = torch.einsum(
            "nki,kio->no",
            neighbors.view(feat.shape[0], self.offsets.shape[0], self.in_channels),
            self.weight,
        )
        return output if self.bias is None else output + self.bias


class AscendEmbeddingConv(CachedCPEConv):
    """CPU FP32 stem using additive integer hashes instead of NxKx3 coordinates."""

    def __init__(self, source):
        super().__init__(source)
        self.register_buffer("offset_keys", None, persistent=False)
        self.register_load_state_dict_post_hook(self._pack_offsets)
        self._pack_offsets(self, None)

    @staticmethod
    def _pack_offsets(module, _incompatible):
        module.offset_keys = module._hash(torch.zeros(len(module.offsets), dtype=torch.long), module.offsets)

    def forward(self, feat, grid_coord, batch):
        if grid_coord.dtype not in (torch.int32, torch.int64):
            return HashSparseConv3d.forward(self, feat, grid_coord, batch)
        signature = ("embedding_gather", len(feat), self.kernel_size, self.in_channels)
        identity = (grid_coord, batch, feat)
        gathered = _geometry_take(signature, identity)
        if gathered is None:
            keys = self._hash(batch, grid_coord)
            sorted_keys, rows = keys.sort()
            # Integer hash arithmetic is linear, including int64 wraparound.
            queries = (keys[:, None] + self.offset_keys[None, :]).reshape(-1)
            positions = torch.searchsorted(sorted_keys, queries).clamp(max=len(feat) - 1)
            neighbors = feat[rows[positions]] * (sorted_keys[positions] == queries).unsqueeze(-1)
            gathered = _geometry_put(
                signature,
                identity,
                neighbors.reshape(len(feat), -1),
            )
        output = gathered @ self.weight.reshape(-1, self.out_channels)
        return output if self.bias is None else output + self.bias


class SubMCPEConv(CachedCPEConv):
    """Default CPE: reference representatives on CPU, map/conv on the NPU."""

    def __init__(self, source):
        super().__init__(source)
        self._supported_shape = self.kernel_size in (1, 3, 5) and all(
            16 <= c <= 512 and c % 16 == 0 for c in (self.in_channels, self.out_channels)
        )
        self.register_buffer("npu_weight", None, persistent=False)
        self.register_buffer("npu_bias", None, persistent=False)
        if self._supported_shape:
            from torch_models.graspgenx_ascend_310p.custom_ops import submconv3d  # noqa: F401
        self.register_load_state_dict_post_hook(self._pack_weights)
        self._pack_weights(self, None)

    @staticmethod
    @torch.no_grad()
    def _pack_weights(module, _incompatible):
        if module._supported_shape:
            module.npu_weight = module.weight.to(NPU_DEVICE, torch.float16).contiguous()
            module.npu_bias = None if module.bias is None else module.bias.to(NPU_DEVICE, torch.float16).contiguous()

    def _get_npu_map(self, grid_coord, batch, point):
        cache_key = f"_cpe_npu_map_{self.kernel_size}"
        if cache_key not in point:
            signature = ("cpe_npu_map", len(grid_coord), self.kernel_size)
            identity = (grid_coord, batch)
            shared = _geometry_take(signature, identity)
            if shared is not None:
                point[cache_key] = shared
                return shared
            coordinates = torch.cat((batch[:, None], grid_coord), dim=1)
            if coordinates.min() < -(2**31) or coordinates.max() >= 2**31:
                # Keep reference semantics outside the builder's int32 range.
                cache = self._get_neighbor_map(grid_coord, batch, point)
                n, volume = grid_coord.shape[0], self.kernel_size**3
                packed = torch.full((n, (volume + 7) // 8 * 8), -1, dtype=torch.int32)
                packed[:, :volume] = cache["indices"].view(n, volume)
                packed[:, :volume].masked_fill_(~cache["found"].view(n, volume), -1)
                point[cache_key] = packed.to(NPU_DEVICE)
            else:
                # Keep CPU sort's chosen row for every hash, including collisions.
                keys, rows = self._hash(batch, grid_coord).sort()
                first = torch.ones_like(keys, dtype=torch.bool)
                first[1:] = keys[1:] != keys[:-1]
                point[cache_key] = torch.ops.graspgenx_subm.build_subm_map(
                    coordinates.to(NPU_DEVICE, torch.int32),
                    self.kernel_size,
                    keys[first].to(NPU_DEVICE),
                    rows[first].to(NPU_DEVICE, torch.int32),
                )
            _geometry_put(signature, identity, point[cache_key])
        return point[cache_key]

    def forward(self, feat, grid_coord, batch, point, *, projected_weight=None, projected_bias=None):
        if not self._supported_shape or not 1 <= feat.shape[0] <= 4096:
            return super().forward(feat, grid_coord, batch, point)
        cpu_input = feat.device.type == "cpu" and feat.dtype == torch.float32
        npu_input = feat.device.type == "npu" and feat.dtype == torch.float16
        if self.training or not (cpu_input or npu_input):
            raise RuntimeError("SubM CPE requires eval mode and CPU FP32 or NPU FP16 features")
        if cpu_input and projected_weight is not None:
            raise RuntimeError("Projected CPE weights require NPU FP16 features")
        if npu_input and feat.device != self.npu_weight.device:
            raise RuntimeError("CPE features must share the convolution NPU device")
        neighbors = self._get_npu_map(grid_coord, batch, point)
        output = torch.ops.graspgenx_subm.subm_conv3d(
            feat.to(NPU_DEVICE, torch.float16) if cpu_input else feat,
            neighbors,
            self.npu_weight if projected_weight is None else projected_weight,
        )
        if cpu_input:
            output = output.to(device="cpu", dtype=torch.float32)
            return output if self.bias is None else output + self.bias
        bias = self.npu_bias if projected_weight is None else projected_bias
        return output if bias is None else output + bias
