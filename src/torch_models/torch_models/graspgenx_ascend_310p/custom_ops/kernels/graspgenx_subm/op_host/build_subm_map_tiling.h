#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BuildSubmMapTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, points);
    TILING_DATA_FIELD_DEF(uint32_t, kernel);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, key_count);
    TILING_DATA_FIELD_DEF(uint32_t, table_capacity);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BuildSubmMap, BuildSubmMapTilingData)
}
