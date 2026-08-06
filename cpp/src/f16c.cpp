#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace simd_lab {

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
__attribute__((target("avx,f16c")))
static void clamp8_f16c_block(const std::uint16_t* c,
                              const std::uint16_t* lo,
                              const std::uint16_t* hi,
                              std::uint16_t* dst) {
    const __m128i hc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c));
    const __m128i hlo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lo));
    const __m128i hhi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(hi));

    const __m256 vc = _mm256_cvtph_ps(hc);
    const __m256 vlo = _mm256_cvtph_ps(hlo);
    const __m256 vhi = _mm256_cvtph_ps(hhi);
    const __m256 out = _mm256_max_ps(vlo, _mm256_min_ps(vc, vhi));
    const __m128i half = _mm256_cvtps_ph(out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), half);
}
#endif

// Binary16 storage -> f32 arithmetic -> binary16 storage. Returns false when
// the CPU cannot execute F16C or when n is not a whole SIMD block.
bool clamp_f16c(std::uint16_t* dst, const std::uint16_t* c,
                const std::uint16_t* lo, const std::uint16_t* hi,
                std::size_t n) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (!__builtin_cpu_supports("avx") || !__builtin_cpu_supports("f16c")) {
        return false;
    }
    if ((n % 8) != 0) return false;
    for (std::size_t i = 0; i < n; i += 8) {
        clamp8_f16c_block(c + i, lo + i, hi + i, dst + i);
    }
    return true;
#else
    (void)dst; (void)c; (void)lo; (void)hi; (void)n;
    return false;
#endif
}

} // namespace simd_lab
