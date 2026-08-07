#include "kernels.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// This translation unit is compiled with /arch:AVX2 only on MSVC. Baseline
// dispatch remains in kernels.cpp and f16c.cpp, so unsupported CPUs never enter
// functions whose prologues or bodies may contain AVX instructions.
namespace simd_lab {

double squared_error_avx2_msvc(std::span<const float> a,
                               std::span<const float> b) noexcept {
    assert(a.size() == b.size());
    __m256d acc_lo = _mm256_setzero_pd();
    __m256d acc_hi = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= a.size(); i += 8) {
        const __m256 va = _mm256_loadu_ps(a.data() + i);
        const __m256 vb = _mm256_loadu_ps(b.data() + i);
        const __m256 d = _mm256_sub_ps(va, vb);
        const __m256d d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(d));
        const __m256d d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(d, 1));
        acc_lo = _mm256_fmadd_pd(d_lo, d_lo, acc_lo);
        acc_hi = _mm256_fmadd_pd(d_hi, d_hi, acc_hi);
    }

    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, _mm256_add_pd(acc_lo, acc_hi));
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < a.size(); ++i) {
        const auto d = static_cast<double>(a[i] - b[i]);
        sum += d * d;
    }
    return sum;
}

std::uint64_t sad_u8_avx2_msvc(std::span<const std::uint8_t> a,
                               std::span<const std::uint8_t> b) noexcept {
    assert(a.size() == b.size());
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

void clamp_f16c_msvc(std::uint16_t* dst, const std::uint16_t* c,
                     const std::uint16_t* lo, const std::uint16_t* hi,
                     std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; i += 8) {
        const __m128i hc = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(c + i));
        const __m128i hlo = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(lo + i));
        const __m128i hhi = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(hi + i));
        const __m256 vc = _mm256_cvtph_ps(hc);
        const __m256 vlo = _mm256_cvtph_ps(hlo);
        const __m256 vhi = _mm256_cvtph_ps(hhi);
        const __m256 out = _mm256_max_ps(vlo, _mm256_min_ps(vc, vhi));
        const __m128i half = _mm256_cvtps_ph(
            out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), half);
    }
}

} // namespace simd_lab
