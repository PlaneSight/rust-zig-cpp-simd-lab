//! x86 F16C experiments.
//!
//! Storage is represented as IEEE-754 binary16 bit patterns (`u16`). F16C is
//! used only at the boundary: convert eight half values to f32, do the full
//! clamp in f32, then narrow once. This is deliberately distinct from native
//! f16 arithmetic semantics.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn clamp8_f16c_block(c: *const u16, lo: *const u16, hi: *const u16, dst: *mut u16) {
    unsafe {
        let hc = _mm_loadu_si128(c.cast());
        let hlo = _mm_loadu_si128(lo.cast());
        let hhi = _mm_loadu_si128(hi.cast());

        let vc = _mm256_cvtph_ps(hc);
        let vlo = _mm256_cvtph_ps(hlo);
        let vhi = _mm256_cvtph_ps(hhi);
        let out = _mm256_max_ps(vlo, _mm256_min_ps(vc, vhi));
        let half = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(out);
        _mm_storeu_si128(dst.cast(), half);
    }
}

/// Clamp binary16 storage using F16C conversion and f32 arithmetic.
///
/// Returns `false` when the current CPU cannot execute the kernel.
pub fn clamp_f16c(dst: &mut [u16], c: &[u16], lo: &[u16], hi: &[u16]) -> bool {
    assert_eq!(dst.len(), c.len());
    assert_eq!(c.len(), lo.len());
    assert_eq!(lo.len(), hi.len());

    #[cfg(target_arch = "x86_64")]
    {
        if !(std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return false;
        }

        let mut i = 0;
        while i + 8 <= dst.len() {
            unsafe {
                clamp8_f16c_block(
                    c.as_ptr().add(i),
                    lo.as_ptr().add(i),
                    hi.as_ptr().add(i),
                    dst.as_mut_ptr().add(i),
                );
            }
            i += 8;
        }

        // Keep the experiment SIMD-only for now. Runtime benchmarks choose
        // lengths divisible by eight; tails will get a shared scalar half
        // conversion helper when non-multiple sizes enter the matrix.
        if i != dst.len() {
            return false;
        }
        true
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, c, lo, hi);
        false
    }
}
