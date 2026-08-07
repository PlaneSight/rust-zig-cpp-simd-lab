#include "kernels.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
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
    const __m128i half = _mm256_cvtps_ph(
        out, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), half);
}
#endif

#if defined(_MSC_VER) && defined(_M_X64)
void clamp_f16c_msvc(std::uint16_t* dst, const std::uint16_t* c,
                     const std::uint16_t* lo, const std::uint16_t* hi,
                     std::size_t n) noexcept;

static bool cpu_has_avx_f16c() noexcept {
    int registers[4]{};
    __cpuidex(registers, 1, 0);
    constexpr int osxsave = 1 << 27;
    constexpr int avx = 1 << 28;
    constexpr int f16c = 1 << 29;
    if ((registers[2] & (osxsave | avx | f16c)) !=
        (osxsave | avx | f16c)) {
        return false;
    }
    return (_xgetbv(0) & 0x6) == 0x6;
}
#endif

bool clamp_f16c(std::uint16_t* dst, const std::uint16_t* c,
                const std::uint16_t* lo, const std::uint16_t* hi,
                std::size_t n) {
    // Rejection is transactional: validate the complete-block contract before
    // checking features or executing a single store.
    if ((n % 8) != 0) return false;

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (!__builtin_cpu_supports("avx") || !__builtin_cpu_supports("f16c")) {
        return false;
    }
    for (std::size_t i = 0; i < n; i += 8) {
        clamp8_f16c_block(c + i, lo + i, hi + i, dst + i);
    }
    return true;
#elif defined(_MSC_VER) && defined(_M_X64)
    if (!cpu_has_avx_f16c()) return false;
    clamp_f16c_msvc(dst, c, lo, hi, n);
    return true;
#else
    (void)dst;
    (void)c;
    (void)lo;
    (void)hi;
    (void)n;
    return false;
#endif
}

} // namespace simd_lab
