#include "kernels.hpp"

#include <cassert>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace simd_lab {

void axpy_scalar(std::span<float> dst, std::span<const float> x,
                 std::span<const float> y, float a) {
    assert(dst.size() == x.size() && x.size() == y.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = std::fma(a, x[i], y[i]);
    }
}

float squared_error_scalar(std::span<const float> a,
                           std::span<const float> b) {
    assert(a.size() == b.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

std::uint64_t sad_u8_scalar(std::span<const std::uint8_t> a,
                            std::span<const std::uint8_t> b) {
    assert(a.size() == b.size());
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto x = a[i];
        const auto y = b[i];
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
__attribute__((target("avx2,fma")))
static float squared_error_avx2(std::span<const float> a,
                                std::span<const float> b) {
    __m256 acc = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= a.size(); i += 8) {
        const __m256 va = _mm256_loadu_ps(a.data() + i);
        const __m256 vb = _mm256_loadu_ps(b.data() + i);
        const __m256 d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    float sum = 0.0f;
    for (float lane : lanes) sum += lane;
    for (; i < a.size(); ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

__attribute__((target("avx2")))
static std::uint64_t sad_u8_avx2(std::span<const std::uint8_t> a,
                                 std::span<const std::uint8_t> b) {
    __m256i acc = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 32 <= a.size(); i += 32) {
        const __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a.data() + i));
        const __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b.data() + i));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
    }

    alignas(32) std::uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    std::uint64_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < a.size(); ++i) {
        const auto x = a[i];
        const auto y = b[i];
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}
#endif

float squared_error_best(std::span<const float> a,
                         std::span<const float> b) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return squared_error_avx2(a, b);
    }
#endif
    return squared_error_scalar(a, b);
}

std::uint64_t sad_u8_best(std::span<const std::uint8_t> a,
                          std::span<const std::uint8_t> b) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) {
        return sad_u8_avx2(a, b);
    }
#endif
    return sad_u8_scalar(a, b);
}

} // namespace simd_lab
