#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SubmConv3dTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, points);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, volume);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, channel_tiles);
    TILING_DATA_FIELD_DEF(uint32_t, jobs);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SubmConv3d, SubmConv3dTilingData)
}
