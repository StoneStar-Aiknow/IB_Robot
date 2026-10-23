#include "build_subm_map_tiling.h"
#include "graph/types.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* ctx) {
    const auto& shape = ctx->GetInputShape(0)->GetStorageShape();
    const auto k = *ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 4 ||
        shape.GetDim(0) < 1 || shape.GetDim(0) > 4096 ||
        (k != 1 && k != 3 && k != 5)) return ge::GRAPH_FAILED;
    const auto* keys = ctx->GetOptionalInputShape(1);
    const auto* rows = ctx->GetOptionalInputShape(2);
    if ((keys == nullptr) != (rows == nullptr)) return ge::GRAPH_FAILED;
    uint32_t key_count = 0;
    if (keys != nullptr) {
        const auto& ks = keys->GetStorageShape();
        const auto& rs = rows->GetStorageShape();
        const auto* kd = ctx->GetOptionalInputDesc(1);
        const auto* rd = ctx->GetOptionalInputDesc(2);
        if (ks.GetDimNum() != 1 || rs.GetDimNum() != 1 ||
            ks.GetDim(0) < 1 || ks.GetDim(0) > shape.GetDim(0) ||
            rs.GetDim(0) != ks.GetDim(0) || kd == nullptr || rd == nullptr ||
            kd->GetDataType() != ge::DT_INT64 || rd->GetDataType() != ge::DT_INT32)
            return ge::GRAPH_FAILED;
        key_count = ks.GetDim(0);
    }
    const auto& out = ctx->GetOutputShape(0)->GetStorageShape();
    if (out.GetDimNum() != 2 || out.GetDim(0) != shape.GetDim(0) ||
        out.GetDim(1) != (k * k * k + 7) / 8 * 8) return ge::GRAPH_FAILED;
    BuildSubmMapTilingData data;
    data.set_points(shape.GetDim(0));
    data.set_kernel(k);
    data.set_width((k * k * k + 7) / 8 * 8);
    data.set_key_count(key_count);
    uint32_t capacity = 0;
    // P1 dense/spread sweeps: K=5 queries cost more than scanning at small N.
    const int64_t scan_limit = k == 5 ? 256 : 128;
    if (key_count == 0 && k != 1 && shape.GetDim(0) > scan_limit) {
        capacity = 1;
        while (capacity < 2 * shape.GetDim(0)) capacity <<= 1;
    }
    data.set_table_capacity(capacity);
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    ctx->SetBlockDim(std::min<uint32_t>(shape.GetDim(0), platform.GetCoreNumAic()));
    ctx->SetTilingKey(0);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class BuildSubmMap : public OpDef {
public:
    explicit BuildSubmMap(const char* name) : OpDef(name) {
        Input("indices").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("sorted_keys").ParamType(OPTIONAL).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("source_rows").ParamType(OPTIONAL).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("neighbors").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("kernel_size").AttrType(REQUIRED).Int();
        SetInferShape([](gert::InferShapeContext* ctx) {
            const auto* in = ctx->GetInputShape(0);
            const auto* keys = ctx->GetOptionalInputShape(1);
            const auto* rows = ctx->GetOptionalInputShape(2);
            auto* out = ctx->GetOutputShape(0);
            const auto* attrs = ctx->GetAttrs();
            const auto* kernel = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(0);
            const int64_t k = kernel == nullptr ? -1 : *kernel;
            const auto unknown_rank = [](const gert::Shape* shape) {
                return shape != nullptr && shape->GetDimNum() == 1 &&
                       shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
            };
            const bool in_unknown = unknown_rank(in);
            const int64_t n = in != nullptr && in->GetDimNum() == 2 ? in->GetDim(0) : ge::UNKNOWN_DIM;
            // JIT-off GE also probes zero-sized shapes. Concrete size limits
            // belong to the bridge and runtime tiling, not this shape derivation.
            if (in == nullptr || out == nullptr || kernel == nullptr ||
                (k != 1 && k != 3 && k != 5) ||
                (!in_unknown && in->GetDimNum() != 2) ||
                ((keys == nullptr) != (rows == nullptr))) return ge::GRAPH_FAILED;
            if (keys != nullptr &&
                ((!unknown_rank(keys) && keys->GetDimNum() != 1) ||
                 (!unknown_rank(rows) && rows->GetDimNum() != 1))) return ge::GRAPH_FAILED;
            out->SetDimNum(2);
            out->SetDim(0, n);
            out->SetDim(1, (k * k * k + 7) / 8 * 8);
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext* ctx) {
            ctx->SetOutputDataType(0, ge::DT_INT32);
            return ge::GRAPH_SUCCESS;
        });
        AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
    }
};
OP_ADD(BuildSubmMap);
}
