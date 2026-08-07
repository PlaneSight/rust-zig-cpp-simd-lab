#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

extern "C" void sat_add_u8_scalar(std::uint8_t* dst,
                                  const std::uint8_t* a,
                                  const std::uint8_t* b,
                                  std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const unsigned sum = static_cast<unsigned>(a[i]) + static_cast<unsigned>(b[i]);
        dst[i] = static_cast<std::uint8_t>(sum > 255u ? 255u : sum);
    }
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
extern "C" __attribute__((target("avx2")))
void sat_add_u8_avx2(std::uint8_t* dst,
                     const std::uint8_t* a,
                     const std::uint8_t* b,
                     std::size_t len) {
    std::size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_adds_epu8(va, vb));
    }
    for (; i < len; ++i) {
        const unsigned sum = static_cast<unsigned>(a[i]) + static_cast<unsigned>(b[i]);
        dst[i] = static_cast<std::uint8_t>(sum > 255u ? 255u : sum);
    }
}
#endif
extern "C" void sat_add_i8_scalar(std::int8_t* dst,
                                  const std::int8_t* a,
                                  const std::int8_t* b,
                                  std::size_t len) {
    const std::int32_t max_value = static_cast<std::int32_t>(std::numeric_limits<std::int8_t>::max());
    const std::int32_t min_value = static_cast<std::int32_t>(std::numeric_limits<std::int8_t>::min());
    for (std::size_t i = 0; i < len; ++i) {
        const std::int32_t sum = static_cast<std::int32_t>(a[i]) + static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int8_t>(sum > max_value ? max_value : sum < min_value ? min_value : sum);
    }
}

extern "C" void sat_add_u16_scalar(std::uint16_t* dst,
                                   const std::uint16_t* a,
                                   const std::uint16_t* b,
                                   std::size_t len) {
    const std::uint32_t max_value = static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max());
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint32_t sum = static_cast<std::uint32_t>(a[i]) + static_cast<std::uint32_t>(b[i]);
        dst[i] = static_cast<std::uint16_t>(sum > max_value ? max_value : sum);
    }
}

extern "C" void sat_add_i16_scalar(std::int16_t* dst,
                                   const std::int16_t* a,
                                   const std::int16_t* b,
                                   std::size_t len) {
    const std::int32_t max_value = static_cast<std::int32_t>(std::numeric_limits<std::int16_t>::max());
    const std::int32_t min_value = static_cast<std::int32_t>(std::numeric_limits<std::int16_t>::min());
    for (std::size_t i = 0; i < len; ++i) {
        const std::int32_t sum = static_cast<std::int32_t>(a[i]) + static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int16_t>(sum > max_value ? max_value : sum < min_value ? min_value : sum);
    }
}

extern "C" void sat_add_u32_scalar(std::uint32_t* dst,
                                   const std::uint32_t* a,
                                   const std::uint32_t* b,
                                   std::size_t len) {
    const std::uint64_t max_value = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
    for (std::size_t i = 0; i < len; ++i) {
        const std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + static_cast<std::uint64_t>(b[i]);
        dst[i] = static_cast<std::uint32_t>(sum > max_value ? max_value : sum);
    }
}

extern "C" void sat_add_i32_scalar(std::int32_t* dst,
                                   const std::int32_t* a,
                                   const std::int32_t* b,
                                   std::size_t len) {
    const std::int32_t max_value = std::numeric_limits<std::int32_t>::max();
    const std::int32_t min_value = std::numeric_limits<std::int32_t>::min();
    for (std::size_t i = 0; i < len; ++i) {
        const std::int32_t lhs = a[i];
        const std::int32_t rhs = b[i];
        if (rhs > 0 && lhs > max_value - rhs) {
            dst[i] = max_value;
        } else if (rhs < 0 && lhs < min_value - rhs) {
            dst[i] = min_value;
        } else {
            dst[i] = lhs + rhs;
        }
    }
}

extern "C" void sat_add_u64_scalar(std::uint64_t* dst,
                                   const std::uint64_t* a,
                                   const std::uint64_t* b,
                                   std::size_t len) {
    const std::uint64_t max_value = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < len; ++i) {
        if (a[i] > max_value - b[i]) {
            dst[i] = max_value;
        } else {
            dst[i] = a[i] + b[i];
        }
    }
}

extern "C" void sat_add_i64_scalar(std::int64_t* dst,
                                   const std::int64_t* a,
                                   const std::int64_t* b,
                                   std::size_t len) {
    const std::int64_t max_value = std::numeric_limits<std::int64_t>::max();
    const std::int64_t min_value = std::numeric_limits<std::int64_t>::min();
    for (std::size_t i = 0; i < len; ++i) {
        const std::int64_t lhs = a[i];
        const std::int64_t rhs = b[i];
        if (rhs > 0 && lhs > max_value - rhs) {
            dst[i] = max_value;
        } else if (rhs < 0 && lhs < min_value - rhs) {
            dst[i] = min_value;
        } else {
            dst[i] = lhs + rhs;
        }
    }
}
