#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridEncodeTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, points);
    TILING_DATA_FIELD_DEF(uint32_t, depth);
    TILING_DATA_FIELD_DEF(uint32_t, block_points);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GridEncode, GridEncodeTilingData)
}
