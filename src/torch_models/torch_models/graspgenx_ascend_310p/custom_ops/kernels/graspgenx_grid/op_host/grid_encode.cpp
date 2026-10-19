#include "grid_encode_tiling.h"
#include "graph/types.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* ctx) {
    const auto* input = ctx->GetInputShape(0);
    const auto* output = ctx->GetOutputShape(0);
    const auto* input_desc = ctx->GetInputDesc(0);
    const auto* output_desc = ctx->GetOutputDesc(0);
    const auto* attrs = ctx->GetAttrs();
    const auto* depth = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(0);
    if (input == nullptr || output == nullptr || input_desc == nullptr ||
        output_desc == nullptr || depth == nullptr || *depth < 1 || *depth > 16 ||
        input_desc->GetDataType() != ge::DT_INT32 ||
        output_desc->GetDataType() != ge::DT_INT64 ||
        input_desc->GetStorageFormat() != ge::FORMAT_ND ||
        output_desc->GetStorageFormat() != ge::FORMAT_ND) return ge::GRAPH_FAILED;
    const auto& shape = input->GetStorageShape();
    const auto& out = output->GetStorageShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(1) != 3 ||
        shape.GetDim(0) < 1 || shape.GetDim(0) > 4096 ||
        out.GetDimNum() != 2 || out.GetDim(0) != shape.GetDim(0) ||
        out.GetDim(1) != 4) return ge::GRAPH_FAILED;

    const uint32_t points = shape.GetDim(0);
    const uint32_t groups = (points + 7) / 8;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    const uint32_t cores = std::min<uint32_t>(8, std::min<uint32_t>(groups, platform.GetCoreNumAic()));
    if (cores == 0) return ge::GRAPH_FAILED;
    GridEncodeTilingData data;
    data.set_points(points);
    data.set_depth(*depth);
    // Eight xyz rows occupy 96 bytes, so every core's DMA start is aligned.
    data.set_block_points((groups + cores - 1) / cores * 8);
    ctx->SetBlockDim(cores);
    ctx->SetTilingKey(0);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class GridEncode : public OpDef {
public:
    explicit GridEncode(const char* name) : OpDef(name) {
        Input("grid_coord").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("spatial_codes").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("depth").AttrType(REQUIRED).Int();
        SetInferShape([](gert::InferShapeContext* ctx) {
            const auto* in = ctx->GetInputShape(0);
            auto* out = ctx->GetOutputShape(0);
            const auto* attrs = ctx->GetAttrs();
            const auto* depth = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(0);
            if (in == nullptr || out == nullptr || depth == nullptr ||
                *depth < 1 || *depth > 16) return ge::GRAPH_FAILED;
            const bool unknown_rank = in->GetDimNum() == 1 && in->GetDim(0) == ge::UNKNOWN_DIM_NUM;
            if (!unknown_rank && in->GetDimNum() != 2) return ge::GRAPH_FAILED;
            // JIT-off GE probes zeros/unknowns, including the coordinate width.
            // Derive the output here; enforce real [N, 3] limits at runtime.
            out->SetDimNum(2);
            out->SetDim(0, unknown_rank ? ge::UNKNOWN_DIM : in->GetDim(0));
            out->SetDim(1, 4);
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext* ctx) {
            ctx->SetOutputDataType(0, ge::DT_INT64);
            return ge::GRAPH_SUCCESS;
        });
        AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
    }
};
OP_ADD(GridEncode);
}
