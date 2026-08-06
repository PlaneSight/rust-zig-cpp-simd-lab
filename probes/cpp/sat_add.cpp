#include <cstddef>
#include <cstdint>

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
