#include "subm_conv3d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* ctx) {
    const auto& x = ctx->GetInputShape(0)->GetStorageShape();
    const auto& map = ctx->GetInputShape(1)->GetStorageShape();
    const auto& w = ctx->GetInputShape(2)->GetStorageShape();
    if (x.GetDimNum() != 2 || map.GetDimNum() != 2 || w.GetDimNum() != 3)
        return ge::GRAPH_FAILED;
    const int64_t n = x.GetDim(0), ci = x.GetDim(1), co = w.GetDim(2), kv = w.GetDim(0);
    if (n < 1 || n > 4096 || ci < 16 || ci > 512 || ci % 16 ||
        co < 16 || co > 512 || co % 16 || w.GetDim(1) != ci ||
        (kv != 1 && kv != 27 && kv != 125) || map.GetDim(0) != n ||
        map.GetDim(1) != (kv + 7) / 8 * 8) return ge::GRAPH_FAILED;
    const auto& out = ctx->GetOutputShape(0)->GetStorageShape();
    if (out.GetDimNum() != 2 || out.GetDim(0) != n || out.GetDim(1) != co)
        return ge::GRAPH_FAILED;
    SubmConv3dTilingData data;
    data.set_points(n);
    data.set_cin(ci);
    data.set_cout(co);
    data.set_volume(kv);
    data.set_width(map.GetDim(1));
    const uint32_t channel_tiles = std::min<int64_t>(4, co / 16);
    data.set_channel_tiles(channel_tiles);
    data.set_jobs((n + 15) / 16 * ((co / 16 + channel_tiles - 1) / channel_tiles));
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    ctx->SetBlockDim(std::min<uint32_t>(data.get_jobs(), platform.GetCoreNumAic()));
    ctx->SetTilingKey(0);
    ctx->GetWorkspaceSizes(1)[0] = 0;
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class SubmConv3d : public OpDef {
public:
    explicit SubmConv3d(const char* name) : OpDef(name) {
        Input("features").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("neighbors").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape([](gert::InferShapeContext* ctx) {
            const auto* x = ctx->GetInputShape(0);
            const auto* w = ctx->GetInputShape(2);
            if (x->GetDimNum() != 2 || w->GetDimNum() != 3) return ge::GRAPH_FAILED;
            auto* out = ctx->GetOutputShape(0);
            out->SetDimNum(2);
            out->SetDim(0, x->GetDim(0));
            out->SetDim(1, w->GetDim(2));
            return ge::GRAPH_SUCCESS;
        });
        SetInferDataType([](gert::InferDataTypeContext* ctx) {
            ctx->SetOutputDataType(0, ge::DT_FLOAT16);
            return ge::GRAPH_SUCCESS;
        });
        AICore().SetTiling(optiling::Tiling).AddConfig("ascend310p");
    }
};
OP_ADD(SubmConv3d);
}
