pub mod f16c;

pub fn axpy_scalar(dst: &mut [f32], x: &[f32], y: &[f32], a: f32) {
    assert_eq!(dst.len(), x.len());
    assert_eq!(x.len(), y.len());

    for ((d, &xv), &yv) in dst.iter_mut().zip(x).zip(y) {
        *d = a.mul_add(xv, yv);
    }
}

/// Reference squared error for f32 inputs with f64 accumulation.
///
/// Subtraction retains f32 semantics; only the product and reduction are
/// widened. This keeps the oracle relevant to the SIMD input operation while
/// avoiding the linear f32 accumulation drift seen on large buffers.
pub fn squared_error_scalar(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x - y);
            d * d
        })
        .sum()
}

pub fn sad_u8_scalar(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn squared_error_avx2(a: &[f32], b: &[f32]) -> f64 {
    use core::arch::x86_64::*;

    assert_eq!(a.len(), b.len());

    let mut i = 0;
    let mut acc_lo = _mm256_setzero_pd();
    let mut acc_hi = _mm256_setzero_pd();

    while i + 8 <= a.len() {
        // SAFETY: i..i+8 is in bounds for both equally-sized slices. The
        // function target-feature contract guarantees AVX2 and FMA support.
        let (va, vb) = unsafe {
            (
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
            )
        };
        let d = _mm256_sub_ps(va, vb);
        let d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(d));
        let d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(d));
        acc_lo = _mm256_fmadd_pd(d_lo, d_lo, acc_lo);
        acc_hi = _mm256_fmadd_pd(d_hi, d_hi, acc_hi);
        i += 8;
    }

    let mut lanes = [0.0_f64; 4];
    // SAFETY: lanes provides four writable f64 values and unaligned stores are
    // valid for any ordinary Rust array address.
    unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), _mm256_add_pd(acc_lo, acc_hi)) };
    let mut sum = lanes.into_iter().sum::<f64>();

    while i < a.len() {
        let d = f64::from(a[i] - b[i]);
        sum += d * d;
        i += 1;
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sad_u8_avx2(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let mut i = 0;
    let mut acc = _mm256_setzero_si256();

    while i + 32 <= a.len() {
        // SAFETY: i..i+32 is in bounds for both equally-sized slices. Unaligned
        // loads are used, and the function requires AVX2 at its call boundary.
        let (va, vb) = unsafe {
            (
                _mm256_loadu_si256(a.as_ptr().add(i).cast()),
                _mm256_loadu_si256(b.as_ptr().add(i).cast()),
            )
        };
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
        i += 32;
    }

    let mut lanes = [0_u64; 4];
    // SAFETY: lanes provides 32 writable bytes; the store is unaligned.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
    let mut sum = lanes.into_iter().sum::<u64>();

    while i < a.len() {
        sum += u64::from(a[i].abs_diff(b[i]));
        i += 1;
    }
    sum
}

pub fn squared_error_best(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: runtime feature detection establishes the target-feature
            // precondition; the callee validates equal slice lengths.
            return unsafe { squared_error_avx2(a, b) };
        }
    }

    squared_error_scalar(a, b)
}

pub fn sad_u8_best(a: &[u8], b: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection establishes the target-feature
            // precondition; the callee validates equal slice lengths.
            return unsafe { sad_u8_avx2(a, b) };
        }
    }
    sad_u8_scalar(a, b)
}

pub fn dispatch_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return "avx2+fma";
        }
    }
    "scalar"
}

#[cfg(test)]
mod tests {
    use super::*;

    struct XorShift64(u64);

    impl XorShift64 {
        fn next(&mut self) -> u64 {
            let mut value = self.0;
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            self.0 = value;
            value
        }

        fn f32(&mut self) -> f32 {
            let unit = (self.next() >> 40) as f32 / (1_u32 << 24) as f32;
            unit.mul_add(8.0, -4.0)
        }
    }

    #[test]
    fn axpy_matches_expected() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];
        let mut dst = [0.0; 4];
        axpy_scalar(&mut dst, &x, &y, 2.0);
        assert_eq!(dst, [7.0, 10.0, 13.0, 16.0]);
    }

    #[test]
    fn randomized_simd_paths_match_references() {
        let mut rng = XorShift64(0x8f3c_a516_d27b_49e1);

        for trial in 0..256 {
            let len = if trial < 64 {
                trial
            } else {
                (rng.next() as usize) % 2049
            };
            let a: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
            let b: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
            let reference = squared_error_scalar(&a, &b);
            let candidate = squared_error_best(&a, &b);
            let relative_error = (reference - candidate).abs() / reference.abs().max(1.0);
            assert!(
                relative_error <= 1e-12,
                "len={len}: reference={reference}, candidate={candidate}, relative_error={relative_error}"
            );

            let bytes_a: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let bytes_b: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            assert_eq!(
                sad_u8_scalar(&bytes_a, &bytes_b),
                sad_u8_best(&bytes_a, &bytes_b),
                "len={len}"
            );
        }
    }
}
