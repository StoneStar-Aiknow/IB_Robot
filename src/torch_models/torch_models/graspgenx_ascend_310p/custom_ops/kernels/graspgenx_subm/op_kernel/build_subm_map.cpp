#include "kernel_operator.h"
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Sync() {
    SetFlag<event>(0);
    WaitFlag<event>(0);
}

__aicore__ inline int64_t ReferenceHash(int64_t batch, int64_t x, int64_t y, int64_t z) {
    // Int32 coordinates plus radius <= 2 cannot overflow this signed int64 sum.
    return batch * 334214467LL + x * 73856093LL + y * 19349669LL + z * 83492791LL;
}

__aicore__ inline uint32_t Bucket(int64_t batch, int64_t x, int64_t y, int64_t z,
                                uint32_t mask) {
    const uint64_t key = static_cast<uint64_t>(ReferenceHash(batch, x, y, z));
    return static_cast<uint32_t>(key ^ (key >> 32)) & mask;
}

extern "C" __global__ __aicore__ void build_subm_map(
    GM_ADDR indices, GM_ADDR sorted_keys, GM_ADDR source_rows,
    GM_ADDR neighbors, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    TPipe pipe;
    TBuf<TPosition::VECCALC> coordsBuf, rowBuf, keysBuf, lookupBuf;
    pipe.InitBuffer(rowBuf, t.width * sizeof(int32_t));
    auto row = rowBuf.Get<int32_t>();
    GlobalTensor<int32_t> input, output;
    input.SetGlobalBuffer((__gm__ int32_t*)indices);
    output.SetGlobalBuffer((__gm__ int32_t*)neighbors);

    if (t.key_count == 0 && t.kernel == 1) {
        Duplicate(row, int32_t(-1), t.width);
        Sync<HardEvent::V_S>();
        for (uint32_t i = GetBlockIdx(); i < t.points; i += GetBlockNum()) {
            row.SetValue(0, i);
            Sync<HardEvent::S_MTE3>();
            DataCopy(output[i * t.width], row, t.width);
            Sync<HardEvent::MTE3_S>();
        }
        return;
    }

    pipe.InitBuffer(coordsBuf, (t.points + 1) / 2 * 8 * sizeof(int32_t));
    auto coords = coordsBuf.Get<int32_t>();
    // 310P DMA uses 32-byte blocks. Scalar tails never over-read user storage.
    const uint32_t aligned = t.points / 2 * 8;
    if (aligned) DataCopy(coords, input, aligned);
    Sync<HardEvent::MTE2_S>();
    for (uint32_t i = aligned; i < t.points * 4; ++i) coords.SetValue(i, input.GetValue(i));

    LocalTensor<int64_t> keys;
    LocalTensor<int32_t> lookup;
    if (t.key_count != 0) {
        pipe.InitBuffer(keysBuf, (t.key_count + 3) / 4 * 4 * sizeof(int64_t));
        pipe.InitBuffer(lookupBuf, (t.key_count + 7) / 8 * 8 * sizeof(int32_t));
        keys = keysBuf.Get<int64_t>();
        lookup = lookupBuf.Get<int32_t>();
        GlobalTensor<int64_t> inputKeys;
        GlobalTensor<int32_t> inputRows;
        inputKeys.SetGlobalBuffer((__gm__ int64_t*)sorted_keys);
        inputRows.SetGlobalBuffer((__gm__ int32_t*)source_rows);
        const uint32_t alignedKeys = t.key_count / 4 * 4;
        const uint32_t alignedRows = t.key_count / 8 * 8;
        if (alignedKeys) DataCopy(keys, inputKeys, alignedKeys);
        if (alignedRows) DataCopy(lookup, inputRows, alignedRows);
        Sync<HardEvent::MTE2_S>();
        for (uint32_t j = alignedKeys; j < t.key_count; ++j)
            keys.SetValue(j, inputKeys.GetValue(j));
        for (uint32_t j = alignedRows; j < t.key_count; ++j)
            lookup.SetValue(j, inputRows.GetValue(j));
    } else if (t.table_capacity != 0) {
        pipe.InitBuffer(lookupBuf, t.table_capacity * sizeof(int32_t));
        lookup = lookupBuf.Get<int32_t>();
        Duplicate(lookup, int32_t(-1), t.table_capacity);
        Sync<HardEvent::V_S>();
        // Each core owns its table: <= 1/2 load, row-index sentinel, no atomics.
        const uint32_t mask = t.table_capacity - 1;
        for (uint32_t j = 0; j < t.points; ++j) {
            uint32_t slot = Bucket(coords.GetValue(j * 4), coords.GetValue(j * 4 + 1),
                                   coords.GetValue(j * 4 + 2), coords.GetValue(j * 4 + 3), mask);
            while (lookup.GetValue(slot) != -1) slot = (slot + 1) & mask;
            lookup.SetValue(slot, j);
        }
    }

    const int64_t radius = t.kernel / 2;
    for (uint32_t i = GetBlockIdx(); i < t.points; i += GetBlockNum()) {
        Duplicate(row, int32_t(-1), t.width);
        Sync<HardEvent::V_S>();
        const int32_t batch = coords.GetValue(i * 4);
        const int64_t x = coords.GetValue(i * 4 + 1);
        const int64_t y = coords.GetValue(i * 4 + 2);
        const int64_t z = coords.GetValue(i * 4 + 3);
        if (t.key_count == 0 && t.table_capacity == 0) {
            for (uint32_t j = 0; j < t.points; ++j) {
                if (coords.GetValue(j * 4) != batch) continue;
                const int64_t dx = int64_t(coords.GetValue(j * 4 + 1)) - x + radius;
                const int64_t dy = int64_t(coords.GetValue(j * 4 + 2)) - y + radius;
                const int64_t dz = int64_t(coords.GetValue(j * 4 + 3)) - z + radius;
                if (dx >= 0 && dx < t.kernel && dy >= 0 && dy < t.kernel && dz >= 0 && dz < t.kernel)
                    row.SetValue((dx * t.kernel + dy) * t.kernel + dz, j);
            }
        } else {
            uint32_t column = 0;
            for (int64_t dx = -radius; dx <= radius; ++dx) {
                for (int64_t dy = -radius; dy <= radius; ++dy) {
                    for (int64_t dz = -radius; dz <= radius; ++dz, ++column) {
                        if (t.key_count != 0) {
                            const int64_t key = ReferenceHash(batch, x + dx, y + dy, z + dz);
                            uint32_t lo = 0, hi = t.key_count;
                            while (lo < hi) {
                                const uint32_t mid = lo + (hi - lo) / 2;
                                if (keys.GetValue(mid) < key) lo = mid + 1;
                                else hi = mid;
                            }
                            // No coordinate check here: reference hash collisions are intentional.
                            if (lo < t.key_count && keys.GetValue(lo) == key)
                                row.SetValue(column, lookup.GetValue(lo));
                        } else {
                            const uint32_t mask = t.table_capacity - 1;
                            uint32_t slot = Bucket(batch, x + dx, y + dy, z + dz, mask);
                            int32_t source = lookup.GetValue(slot);
                            while (source != -1) {
                                if (coords.GetValue(source * 4) == batch &&
                                    int64_t(coords.GetValue(source * 4 + 1)) == x + dx &&
                                    int64_t(coords.GetValue(source * 4 + 2)) == y + dy &&
                                    int64_t(coords.GetValue(source * 4 + 3)) == z + dz) {
                                    row.SetValue(column, source);
                                    break;
                                }
                                slot = (slot + 1) & mask;
                                source = lookup.GetValue(slot);
                            }
                        }
                    }
                }
            }
        }
        Sync<HardEvent::S_MTE3>();
        DataCopy(output[i * t.width], row, t.width);
        Sync<HardEvent::MTE3_V>();
    }
}
