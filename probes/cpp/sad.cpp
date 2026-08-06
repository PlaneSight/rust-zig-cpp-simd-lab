#include <cstddef>
#include <cstdint>
#include <immintrin.h>

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

#if defined(__GNUC__) || defined(__clang__)
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
#endif
