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
    // SAFETY: the caller supplies four valid eight-element regions. Unaligned
    // loads/stores are used, and this function's target-feature contract
    // guarantees AVX and F16C support.
    unsafe {
        let hc = _mm_loadu_si128(c.cast());
        let hlo = _mm_loadu_si128(lo.cast());
        let hhi = _mm_loadu_si128(hi.cast());

        let vc = _mm256_cvtph_ps(hc);
        let vlo = _mm256_cvtph_ps(hlo);
        let vhi = _mm256_cvtph_ps(hhi);
        let out = _mm256_max_ps(vlo, _mm256_min_ps(vc, vhi));
        // Rust models VCVTPS2PH's rounding control as a three-bit immediate.
        // Round-to-nearest is zero; MXCSR exception masking remains unchanged.
        let half = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(out);
        _mm_storeu_si128(dst.cast(), half);
    }
}

/// Clamp binary16 storage using F16C conversion and f32 arithmetic.
///
/// Returns `false` without modifying `dst` when the length is not a complete
/// eight-lane block or when the current CPU cannot execute the kernel.
pub fn clamp_f16c(dst: &mut [u16], c: &[u16], lo: &[u16], hi: &[u16]) -> bool {
    assert_eq!(dst.len(), c.len());
    assert_eq!(c.len(), lo.len());
    assert_eq!(lo.len(), hi.len());

    if dst.len() % 8 != 0 {
        return false;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if !(std::arch::is_x86_feature_detected!("avx")
            && std::arch::is_x86_feature_detected!("f16c"))
        {
            return false;
        }

        for i in (0..dst.len()).step_by(8) {
            // SAFETY: the preflight divisibility check makes every eight-lane
            // region complete, and runtime detection proves AVX+F16C support.
            unsafe {
                clamp8_f16c_block(
                    c.as_ptr().add(i),
                    lo.as_ptr().add(i),
                    hi.as_ptr().add(i),
                    dst.as_mut_ptr().add(i),
                );
            }
        }
        true
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (dst, c, lo, hi);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_VALUES: [u16; 8] = [
        0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400,
    ];

    #[test]
    fn incomplete_block_is_rejected_without_writes() {
        for len in (1..64).filter(|len| len % 8 != 0) {
            let c: Vec<u16> = (0..len).map(|i| HALF_VALUES[i & 7]).collect();
            let lo = vec![0x3800; len];
            let hi = vec![0x4000; len];
            let mut dst = vec![0xdead; len];

            assert!(!clamp_f16c(&mut dst, &c, &lo, &hi));
            assert!(dst.iter().all(|&value| value == 0xdead), "len={len}");
        }
    }

    #[test]
    fn randomized_complete_blocks_match_scalar_clamp() {
        for blocks in 0..33 {
            let len = blocks * 8;
            let c: Vec<u16> = (0..len)
                .map(|i| HALF_VALUES[(i * 5 + blocks) & 7])
                .collect();
            let lo = vec![0x3800; len];
            let hi = vec![0x4000; len];
            let mut dst = vec![0; len];

            if !clamp_f16c(&mut dst, &c, &lo, &hi) {
                return;
            }

            let expected: Vec<u16> = c
                .iter()
                .map(|&value| value.clamp(0x3800, 0x4000))
                .collect();
            assert_eq!(dst, expected, "len={len}");
        }
    }
}
