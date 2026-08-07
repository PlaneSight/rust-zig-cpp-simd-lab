// Architecture-neutral widened saturating-add probes for cross-target codegen.
using i8 = signed char;
using i16 = short;
using i32 = int;
using i64 = long long;
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;
using usize = __SIZE_TYPE__;

extern "C" void sat_add_u8_widened(u8* dst, const u8* a, const u8* b,
                                   usize len) {
    constexpr u16 max_value = 255u;
    for (usize i = 0; i < len; ++i) {
        const u16 widened_sum = static_cast<u16>(a[i]) +
                                static_cast<u16>(b[i]);
        dst[i] = static_cast<u8>(
            widened_sum > max_value ? max_value : widened_sum);
    }
}

extern "C" void sat_add_i8_widened(i8* dst, const i8* a, const i8* b,
                                   usize len) {
    constexpr i16 min_value = -128;
    constexpr i16 max_value = 127;
    for (usize i = 0; i < len; ++i) {
        const i16 widened_sum = static_cast<i16>(a[i]) +
                                static_cast<i16>(b[i]);
        dst[i] = static_cast<i8>(
            widened_sum < min_value ? min_value
            : widened_sum > max_value ? max_value
                                      : widened_sum);
    }
}

extern "C" void sat_add_u16_widened(u16* dst, const u16* a, const u16* b,
                                    usize len) {
    constexpr u32 max_value = 65535u;
    for (usize i = 0; i < len; ++i) {
        const u32 widened_sum = static_cast<u32>(a[i]) +
                                static_cast<u32>(b[i]);
        dst[i] = static_cast<u16>(
            widened_sum > max_value ? max_value : widened_sum);
    }
}

extern "C" void sat_add_i16_widened(i16* dst, const i16* a, const i16* b,
                                    usize len) {
    constexpr i32 min_value = -32768;
    constexpr i32 max_value = 32767;
    for (usize i = 0; i < len; ++i) {
        const i32 widened_sum = static_cast<i32>(a[i]) +
                                static_cast<i32>(b[i]);
        dst[i] = static_cast<i16>(
            widened_sum < min_value ? min_value
            : widened_sum > max_value ? max_value
                                      : widened_sum);
    }
}

extern "C" void sat_add_u32_widened(u32* dst, const u32* a, const u32* b,
                                    usize len) {
    constexpr u64 max_value = static_cast<u64>(~static_cast<u32>(0));
    for (usize i = 0; i < len; ++i) {
        const u64 widened_sum = static_cast<u64>(a[i]) +
                                static_cast<u64>(b[i]);
        dst[i] = static_cast<u32>(
            widened_sum > max_value ? max_value : widened_sum);
    }
}

extern "C" void sat_add_i32_widened(i32* dst, const i32* a, const i32* b,
                                    usize len) {
    constexpr i64 min_value = -2147483648LL;
    constexpr i64 max_value = 2147483647LL;
    for (usize i = 0; i < len; ++i) {
        const i64 widened_sum = static_cast<i64>(a[i]) +
                                static_cast<i64>(b[i]);
        dst[i] = static_cast<i32>(
            widened_sum < min_value ? min_value
            : widened_sum > max_value ? max_value
                                      : widened_sum);
    }
}

extern "C" void sat_add_u64_widened(u64* dst, const u64* a, const u64* b,
                                    usize len) {
    constexpr u64 max_value = ~static_cast<u64>(0);
    for (usize i = 0; i < len; ++i) {
        const u64 lhs = a[i];
        const u64 rhs = b[i];
        dst[i] = lhs > max_value - rhs ? max_value : lhs + rhs;
    }
}

extern "C" void sat_add_i64_widened(i64* dst, const i64* a, const i64* b,
                                    usize len) {
    constexpr i64 max_value = static_cast<i64>(~static_cast<u64>(0) >> 1);
    constexpr i64 min_value = -max_value - 1;
    for (usize i = 0; i < len; ++i) {
        const i64 lhs = a[i];
        const i64 rhs = b[i];
        if (rhs > 0 && lhs > max_value - rhs) {
            dst[i] = max_value;
        } else if (rhs < 0 && lhs < min_value - rhs) {
            dst[i] = min_value;
        } else {
            dst[i] = lhs + rhs;
        }
    }
}
