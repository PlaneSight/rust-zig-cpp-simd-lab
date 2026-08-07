#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

extern "C" void sat_sub_u8_scalar(std::uint8_t* dst,
                                  const std::uint8_t* a,
                                  const std::uint8_t* b,
                                  std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const auto lhs = static_cast<std::uint16_t>(a[i]);
        const auto rhs = static_cast<std::uint16_t>(b[i]);
        dst[i] = static_cast<std::uint8_t>(lhs < rhs ? 0U : lhs - rhs);
    }
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
extern "C" __attribute__((target("avx2")))
void sat_sub_u8_avx2(std::uint8_t* dst,
                     const std::uint8_t* a,
                     const std::uint8_t* b,
                     std::size_t len) {
    std::size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        const __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i),
                            _mm256_subs_epu8(va, vb));
    }
    for (; i < len; ++i) {
        const auto lhs = static_cast<std::uint16_t>(a[i]);
        const auto rhs = static_cast<std::uint16_t>(b[i]);
        dst[i] = static_cast<std::uint8_t>(lhs < rhs ? 0U : lhs - rhs);
    }
}
#endif

extern "C" void sat_sub_i8_scalar(std::int8_t* dst,
                                  const std::int8_t* a,
                                  const std::int8_t* b,
                                  std::size_t len) {
    constexpr std::int32_t min_value =
        static_cast<std::int32_t>(std::numeric_limits<std::int8_t>::min());
    constexpr std::int32_t max_value =
        static_cast<std::int32_t>(std::numeric_limits<std::int8_t>::max());
    for (std::size_t i = 0; i < len; ++i) {
        const auto difference = static_cast<std::int32_t>(a[i]) -
                                static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int8_t>(
            difference < min_value ? min_value
            : difference > max_value ? max_value
                                      : difference);
    }
}

extern "C" void sat_sub_u16_scalar(std::uint16_t* dst,
                                   const std::uint16_t* a,
                                   const std::uint16_t* b,
                                   std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const auto lhs = static_cast<std::uint32_t>(a[i]);
        const auto rhs = static_cast<std::uint32_t>(b[i]);
        dst[i] = static_cast<std::uint16_t>(lhs < rhs ? 0U : lhs - rhs);
    }
}

extern "C" void sat_sub_i16_scalar(std::int16_t* dst,
                                   const std::int16_t* a,
                                   const std::int16_t* b,
                                   std::size_t len) {
    constexpr std::int32_t min_value =
        static_cast<std::int32_t>(std::numeric_limits<std::int16_t>::min());
    constexpr std::int32_t max_value =
        static_cast<std::int32_t>(std::numeric_limits<std::int16_t>::max());
    for (std::size_t i = 0; i < len; ++i) {
        const auto difference = static_cast<std::int32_t>(a[i]) -
                                static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int16_t>(
            difference < min_value ? min_value
            : difference > max_value ? max_value
                                      : difference);
    }
}

extern "C" void sat_sub_u32_scalar(std::uint32_t* dst,
                                   const std::uint32_t* a,
                                   const std::uint32_t* b,
                                   std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const auto lhs = static_cast<std::uint64_t>(a[i]);
        const auto rhs = static_cast<std::uint64_t>(b[i]);
        dst[i] = static_cast<std::uint32_t>(lhs < rhs ? 0U : lhs - rhs);
    }
}

extern "C" void sat_sub_i32_scalar(std::int32_t* dst,
                                   const std::int32_t* a,
                                   const std::int32_t* b,
                                   std::size_t len) {
    constexpr std::int64_t min_value =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min());
    constexpr std::int64_t max_value =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    for (std::size_t i = 0; i < len; ++i) {
        const auto difference = static_cast<std::int64_t>(a[i]) -
                                static_cast<std::int64_t>(b[i]);
        dst[i] = static_cast<std::int32_t>(
            difference < min_value ? min_value
            : difference > max_value ? max_value
                                      : difference);
    }
}

extern "C" void sat_sub_u64_scalar(std::uint64_t* dst,
                                   const std::uint64_t* a,
                                   const std::uint64_t* b,
                                   std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = a[i] < b[i] ? 0U : a[i] - b[i];
    }
}

extern "C" void sat_sub_i64_scalar(std::int64_t* dst,
                                   const std::int64_t* a,
                                   const std::int64_t* b,
                                   std::size_t len) {
    constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
    constexpr auto max_value = std::numeric_limits<std::int64_t>::max();
    for (std::size_t i = 0; i < len; ++i) {
        const auto lhs = a[i];
        const auto rhs = b[i];
        if (rhs > 0 && lhs < min_value + rhs) {
            dst[i] = min_value;
        } else if (rhs < 0 && lhs > max_value + rhs) {
            dst[i] = max_value;
        } else {
            dst[i] = lhs - rhs;
        }
    }
}
