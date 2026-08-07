// Architecture-neutral raw-pointer image-kernel probes.
// The caller supplies non-overlapping buffers and a blend weight in [0, 256].
// Length is last so the same entry points can be called from FFI harnesses.

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using usize = __SIZE_TYPE__;

extern "C" void blend_u8_scalar(
    u8* dst, const u8* a, const u8* b, u16 weight, usize len) {
    const u32 wide_weight = static_cast<u32>(weight);
    const u32 inverse_weight = 256U - wide_weight;
    for (usize i = 0; i < len; ++i) {
        const u32 weighted_a = static_cast<u32>(a[i]) * inverse_weight;
        const u32 weighted_b = static_cast<u32>(b[i]) * wide_weight;
        dst[i] = static_cast<u8>((weighted_a + weighted_b + 128U) >> 8U);
    }
}

extern "C" void convolve3_u8_scalar(
    u8* dst, const u8* src, usize len) {
    if (len == 0U) return;
    if (len == 1U) {
        dst[0] = src[0];
        return;
    }

    dst[0] = static_cast<u8>(
        (3U * static_cast<u32>(src[0]) + static_cast<u32>(src[1]) + 2U) >> 2U);
    for (usize i = 1; i + 1U < len; ++i) {
        const u32 sum = static_cast<u32>(src[i - 1U])
            + 2U * static_cast<u32>(src[i])
            + static_cast<u32>(src[i + 1U])
            + 2U;
        dst[i] = static_cast<u8>(sum >> 2U);
    }
    dst[len - 1U] = static_cast<u8>(
        (static_cast<u32>(src[len - 2U])
            + 3U * static_cast<u32>(src[len - 1U])
            + 2U) >> 2U);
}

extern "C" void convolve5_u8_scalar(
    u8* dst, const u8* src, usize len) {
    if (len == 0U) return;
    if (len < 5U) {
        for (usize i = 0; i < len; ++i) {
            const usize left2 = i < 2U ? 0U : i - 2U;
            const usize left1 = i == 0U ? 0U : i - 1U;
            const usize right1 = i + 1U < len ? i + 1U : len - 1U;
            const usize right2 = len - 1U - i >= 2U ? i + 2U : len - 1U;
            const u32 sum = static_cast<u32>(src[left2])
                + 4U * static_cast<u32>(src[left1])
                + 6U * static_cast<u32>(src[i])
                + 4U * static_cast<u32>(src[right1])
                + static_cast<u32>(src[right2])
                + 8U;
            dst[i] = static_cast<u8>(sum >> 4U);
        }
        return;
    }

    dst[0] = static_cast<u8>(
        (11U * static_cast<u32>(src[0])
            + 4U * static_cast<u32>(src[1])
            + static_cast<u32>(src[2])
            + 8U) >> 4U);
    dst[1] = static_cast<u8>(
        (5U * static_cast<u32>(src[0])
            + 6U * static_cast<u32>(src[1])
            + 4U * static_cast<u32>(src[2])
            + static_cast<u32>(src[3])
            + 8U) >> 4U);
    for (usize i = 2; i + 2U < len; ++i) {
        const u32 sum = static_cast<u32>(src[i - 2U])
            + 4U * static_cast<u32>(src[i - 1U])
            + 6U * static_cast<u32>(src[i])
            + 4U * static_cast<u32>(src[i + 1U])
            + static_cast<u32>(src[i + 2U])
            + 8U;
        dst[i] = static_cast<u8>(sum >> 4U);
    }
    dst[len - 2U] = static_cast<u8>(
        (static_cast<u32>(src[len - 4U])
            + 4U * static_cast<u32>(src[len - 3U])
            + 6U * static_cast<u32>(src[len - 2U])
            + 5U * static_cast<u32>(src[len - 1U])
            + 8U) >> 4U);
    dst[len - 1U] = static_cast<u8>(
        (static_cast<u32>(src[len - 3U])
            + 4U * static_cast<u32>(src[len - 2U])
            + 11U * static_cast<u32>(src[len - 1U])
            + 8U) >> 4U);
}
