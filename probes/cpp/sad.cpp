#include <cstddef>
#include <cstdint>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

extern "C" std::uint64_t sad_u8_scalar(const std::uint8_t* a,
                                        const std::uint8_t* b,
                                        std::size_t len) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < len; ++i) {
        const auto x = a[i];
        const auto y = b[i];
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

extern "C" std::uint64_t sad_u16_scalar(const std::uint16_t* a,
                                        const std::uint16_t* b,
                                        std::size_t len) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < len; ++i) {
        const auto x = static_cast<std::uint32_t>(a[i]);
        const auto y = static_cast<std::uint32_t>(b[i]);
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(_M_X64))
extern "C" __attribute__((target("avx2")))
std::uint64_t sad_u8_avx2(const std::uint8_t* a,
                          const std::uint8_t* b,
                          std::size_t len) {
    __m256i acc = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        const auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
    }
    alignas(32) std::uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    std::uint64_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < len; ++i) {
        const auto x = a[i];
        const auto y = b[i];
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

extern "C" __attribute__((target("avx2")))
std::uint64_t sad_u16_avx2(const std::uint16_t* a,
                           const std::uint16_t* b,
                           std::size_t len) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 16 <= len; i += 16) {
        const auto va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a + i));
        const auto vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b + i));
        const auto hi = _mm256_max_epu16(va, vb);
        const auto lo = _mm256_min_epu16(va, vb);
        const auto diff = _mm256_sub_epi16(hi, lo);
        const auto diff_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(diff));
        const auto diff_hi = _mm256_cvtepu16_epi32(
            _mm256_extracti128_si256(diff, 1));
        acc0 = _mm256_add_epi64(
            acc0, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff_lo)));
        acc1 = _mm256_add_epi64(
            acc1, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(diff_lo, 1)));
        acc2 = _mm256_add_epi64(
            acc2, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff_hi)));
        acc3 = _mm256_add_epi64(
            acc3, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(diff_hi, 1)));
    }

    alignas(32) std::uint64_t lanes[16];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 4), acc1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 8), acc2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 12), acc3);
    std::uint64_t sum = 0;
    for (const auto lane : lanes) sum += lane;
    for (; i < len; ++i) {
        const auto x = static_cast<std::uint32_t>(a[i]);
        const auto y = static_cast<std::uint32_t>(b[i]);
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}
#endif
