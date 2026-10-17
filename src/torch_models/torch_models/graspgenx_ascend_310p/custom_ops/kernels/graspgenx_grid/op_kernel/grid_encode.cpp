#include "kernel_operator.h"
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Sync() {
    SetFlag<event>(0);
    WaitFlag<event>(0);
}

__aicore__ inline uint64_t Interleave(uint32_t x, uint32_t y, uint32_t z, uint32_t depth) {
    uint64_t code = 0;
    for (int32_t bit = static_cast<int32_t>(depth) - 1; bit >= 0; --bit) {
        code = (code << 3) | (((x >> bit) & 1U) << 2) |
               (((y >> bit) & 1U) << 1) | ((z >> bit) & 1U);
    }
    return code;
}

__aicore__ inline uint64_t Hilbert(uint32_t x, uint32_t y, uint32_t z, uint32_t depth) {
    // Exact integer form of ptv3_vanilla's boolean-bit loop, in x,y,z order.
    // At bit q, a set axis bit inverts x's lower bits; otherwise exchange
    // that axis's lower bits with x. The final bit has an empty lower mask.
    for (uint32_t q = 1U << (depth - 1); q > 1; q >>= 1) {
        const uint32_t lower = q - 1;
        if (x & q) x ^= lower;
        if (y & q) {
            x ^= lower;
        } else {
            const uint32_t exchange = (x ^ y) & lower;
            x ^= exchange;
            y ^= exchange;
        }
        if (z & q) {
            x ^= lower;
        } else {
            const uint32_t exchange = (x ^ z) & lower;
            x ^= exchange;
            z ^= exchange;
        }
    }
    // The reference interleaves MSB-first, then Gray-decodes the whole string.
    uint64_t code = Interleave(x, y, z, depth);
    code ^= code >> 1;
    code ^= code >> 2;
    code ^= code >> 4;
    code ^= code >> 8;
    code ^= code >> 16;
    code ^= code >> 32;
    return code;
}

extern "C" __global__ __aicore__ void grid_encode(
    GM_ADDR grid_coord, GM_ADDR spatial_codes, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(t, tiling);
    const uint32_t groups = (t.points + 7) / 8;
    const uint32_t begin = groups * GetBlockIdx() / GetBlockNum() * 8;
    const uint32_t group_end = groups * (GetBlockIdx() + 1) / GetBlockNum() * 8;
    const uint32_t end = group_end < t.points ? group_end : t.points;
    if (begin >= end) return;
    const uint32_t count = end - begin;

    TPipe pipe;
    TBuf<TPosition::VECCALC> coordsBuf, codesBuf;
    pipe.InitBuffer(coordsBuf, t.block_points * 3 * sizeof(int32_t));
    pipe.InitBuffer(codesBuf, t.block_points * 4 * sizeof(int64_t));
    auto coords = coordsBuf.Get<int32_t>();
    auto codes = codesBuf.Get<int64_t>();
    GlobalTensor<int32_t> input;
    GlobalTensor<int64_t> output;
    input.SetGlobalBuffer((__gm__ int32_t*)grid_coord);
    output.SetGlobalBuffer((__gm__ int64_t*)spatial_codes);

    // 310P copies whole 32-byte blocks only. Never read beyond [N, 3].
    // A contiguous storage-offset view can have an unaligned base: use scalars.
    const uint32_t aligned = (reinterpret_cast<uint64_t>(grid_coord) & 31U) == 0
                                 ? count * 3 / 8 * 8 : 0;
    if (aligned != 0) DataCopy(coords, input[begin * 3], aligned);
    Sync<HardEvent::MTE2_S>();
    for (uint32_t i = aligned; i < count * 3; ++i)
        coords.SetValue(i, input.GetValue(begin * 3 + i));

    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t x = static_cast<uint32_t>(coords.GetValue(i * 3));
        const uint32_t y = static_cast<uint32_t>(coords.GetValue(i * 3 + 1));
        const uint32_t z = static_cast<uint32_t>(coords.GetValue(i * 3 + 2));
        codes.SetValue(i * 4, static_cast<int64_t>(Interleave(x, y, z, t.depth)));
        codes.SetValue(i * 4 + 1, static_cast<int64_t>(Interleave(y, x, z, t.depth)));
        codes.SetValue(i * 4 + 2, static_cast<int64_t>(Hilbert(x, y, z, t.depth)));
        codes.SetValue(i * 4 + 3, static_cast<int64_t>(Hilbert(y, x, z, t.depth)));
    }
    Sync<HardEvent::S_MTE3>();
    DataCopy(output[begin * 4], codes, count * 4);
    // No buffer is reused; complete the only write before releasing the pipe.
    Sync<HardEvent::MTE3_S>();
}
