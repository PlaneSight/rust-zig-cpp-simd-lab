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

/// Adds unsigned bytes element-wise with saturation at `u8::MAX`.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_u8_scalar(dst: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}
/// Computes an f32 dot product with f64 products and accumulation.
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0_f64;
    let mut i = 0;
    while i < a.len() {
        let x = f64::from(a[i]);
        let y = f64::from(b[i]);
        sum += x * y;
        i += 1;
    }
    sum
}

/// Computes an f64 dot product with f64 accumulation.
pub fn dot_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0.0_f64;
    let mut i = 0;
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Computes an i16 dot product with widened i64 products and accumulation.
pub fn dot_i16_scalar(a: &[i16], b: &[i16]) -> i64 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0_i64;
    let mut i = 0;
    while i < a.len() {
        let x = i64::from(a[i]);
        let y = i64::from(b[i]);
        sum += x * y;
        i += 1;
    }
    sum
}

/// Computes a mixed u8/i8 dot product with exact i64 products and accumulation.
pub fn dot_u8_i8_scalar(a: &[u8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len());

    let mut sum = 0_i64;
    let mut i = 0;
    while i < a.len() {
        let x = i64::from(a[i]);
        let y = i64::from(b[i]);
        sum += x * y;
        i += 1;
    }
    sum
}

/// Multiplies u8 pairs after widening each operand to u16.
pub fn widen_mul_u8_u16_scalar(dst: &mut [u16], a: &[u8], b: &[u8]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i < dst.len() {
        let x = u16::from(a[i]);
        let y = u16::from(b[i]);
        dst[i] = x * y;
        i += 1;
    }
}

/// Multiplies i8 pairs after widening each operand to i16.
pub fn widen_mul_i8_i16_scalar(dst: &mut [i16], a: &[i8], b: &[i8]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i < dst.len() {
        let x = i16::from(a[i]);
        let y = i16::from(b[i]);
        dst[i] = x * y;
        i += 1;
    }
}

/// Multiplies u16 pairs after widening each operand to u32.
pub fn widen_mul_u16_u32_scalar(dst: &mut [u32], a: &[u16], b: &[u16]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i < dst.len() {
        let x = u32::from(a[i]);
        let y = u32::from(b[i]);
        dst[i] = x * y;
        i += 1;
    }
}

/// Multiplies i16 pairs after widening each operand to i32.
pub fn widen_mul_i16_i32_scalar(dst: &mut [i32], a: &[i16], b: &[i16]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i < dst.len() {
        let x = i32::from(a[i]);
        let y = i32::from(b[i]);
        dst[i] = x * y;
        i += 1;
    }
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

/// Adds unsigned bytes with AVX2 packed saturation and a scalar tail.
///
/// # Safety
///
/// The caller must ensure that the current CPU and operating system support
/// AVX2. Slice length and non-aliasing invariants are enforced by the safe
/// slice types and validated before the first vector load.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn sat_add_u8_avx2(dst: &mut [u8], a: &[u8], b: &[u8]) {
    use core::arch::x86_64::*;

    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i + 32 <= dst.len() {
        // SAFETY: i..i+32 is in bounds for all three equally-sized slices.
        // The loads and store accept unaligned addresses, and the function's
        // target-feature contract requires AVX2.
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
            let result = _mm256_adds_epu8(va, vb);
            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), result);
        }
        i += 32;
    }

    while i < dst.len() {
        dst[i] = a[i].saturating_add(b[i]);
        i += 1;
    }
}

pub fn squared_error_best(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
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

pub fn sat_add_u8_best(dst: &mut [u8], a: &[u8], b: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: runtime feature detection establishes the target-feature
            // precondition; the callee validates all slice lengths.
            unsafe { sat_add_u8_avx2(dst, a, b) };
            return;
        }
    }
    sat_add_u8_scalar(dst, a, b);
}

pub fn sat_add_u8_dispatch_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    "scalar"
}

pub fn dispatch_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
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

            let mut expected = vec![0_u8; len];
            let mut candidate = vec![0_u8; len];
            for ((out, &x), &y) in expected.iter_mut().zip(&bytes_a).zip(&bytes_b) {
                let widened_sum = u16::from(x) + u16::from(y);
                *out = widened_sum.min(u16::from(u8::MAX)) as u8;
            }
            sat_add_u8_best(&mut candidate, &bytes_a, &bytes_b);
            assert_eq!(candidate, expected, "len={len}");
        }
    }

    #[test]
    fn saturating_add_covers_every_u8_pair() {
        const PAIRS: usize = 256 * 256;
        let mut a = vec![0_u8; PAIRS];
        let mut b = vec![0_u8; PAIRS];
        let mut expected = vec![0_u8; PAIRS];
        let mut candidate = vec![0_u8; PAIRS];

        for x in 0_u16..=u16::from(u8::MAX) {
            for y in 0_u16..=u16::from(u8::MAX) {
                let index = usize::from(x) * 256 + usize::from(y);
                a[index] = x as u8;
                b[index] = y as u8;
                expected[index] = (x + y).min(u16::from(u8::MAX)) as u8;
            }
        }

        sat_add_u8_scalar(&mut candidate, &a, &b);
        assert_eq!(candidate, expected);
        candidate.fill(0);
        sat_add_u8_best(&mut candidate, &a, &b);
        assert_eq!(candidate, expected);
    }

    #[test]
    fn dot_and_widen_cover_extrema_and_pathological_lengths() {
        const LENGTHS: [usize; 14] = [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127];
        for &len in &LENGTHS {
            let a_f32: Vec<f32> = (0..len)
                .map(|i| [f32::MAX, f32::MIN, 1.5, -2.25][i & 3])
                .collect();
            let b_f32: Vec<f32> = (0..len).map(|i| [1.0, 1.0, 2.0, -4.0][i & 3]).collect();
            let mut expected_f32 = 0.0_f64;
            for i in 0..len {
                expected_f32 += f64::from(a_f32[i]) * f64::from(b_f32[i]);
            }
            let candidate_f32 = dot_f32_scalar(&a_f32, &b_f32);
            let error_f32 = (expected_f32 - candidate_f32).abs() / expected_f32.abs().max(1.0);
            assert!(
                error_f32 <= 1e-12,
                "len={len}: f32 relative error={error_f32}"
            );

            let a_f64: Vec<f64> = (0..len)
                .map(|i| [f64::MAX, f64::MIN, 1.0e150, -1.0e150][i & 3])
                .collect();
            let b_f64: Vec<f64> = (0..len)
                .map(|i| [1.0, 1.0, 1.0e-150, -1.0e-150][i & 3])
                .collect();
            let mut expected_f64 = 0.0_f64;
            for i in 0..len {
                expected_f64 += a_f64[i] * b_f64[i];
            }
            let candidate_f64 = dot_f64_scalar(&a_f64, &b_f64);
            let error_f64 = (expected_f64 - candidate_f64).abs() / expected_f64.abs().max(1.0);
            assert!(
                error_f64 <= 1e-12,
                "len={len}: f64 relative error={error_f64}"
            );

            let a_i16: Vec<i16> = (0..len)
                .map(|i| [i16::MIN, -1, 1, i16::MAX][i & 3])
                .collect();
            let b_i16: Vec<i16> = (0..len)
                .map(|i| [i16::MAX, i16::MIN, -1, i16::MAX][i & 3])
                .collect();
            let mut expected_i16 = 0_i64;
            for i in 0..len {
                expected_i16 += i64::from(a_i16[i]) * i64::from(b_i16[i]);
            }
            assert_eq!(dot_i16_scalar(&a_i16, &b_i16), expected_i16);

            let a_u8: Vec<u8> = (0..len).map(|i| [0, 1, 127, u8::MAX][i & 3]).collect();
            let b_i8: Vec<i8> = (0..len).map(|i| [i8::MIN, -1, 1, i8::MAX][i & 3]).collect();
            let mut expected_mixed = 0_i64;
            for i in 0..len {
                expected_mixed += i64::from(a_u8[i]) * i64::from(b_i8[i]);
            }
            assert_eq!(dot_u8_i8_scalar(&a_u8, &b_i8), expected_mixed);

            let a_u8_w: Vec<u8> = (0..len).map(|i| [0, 1, 127, u8::MAX][i & 3]).collect();
            let b_u8_w: Vec<u8> = (0..len).map(|i| [u8::MAX, 254, 128, 1][i & 3]).collect();
            let mut expected_u8_u16 = vec![0_u16; len];
            for i in 0..len {
                expected_u8_u16[i] = u16::from(a_u8_w[i]) * u16::from(b_u8_w[i]);
            }
            let mut candidate_u8_u16 = vec![0_u16; len];
            widen_mul_u8_u16_scalar(&mut candidate_u8_u16, &a_u8_w, &b_u8_w);
            assert_eq!(candidate_u8_u16, expected_u8_u16);

            let a_i8: Vec<i8> = (0..len).map(|i| [i8::MIN, -1, 1, i8::MAX][i & 3]).collect();
            let b_i8_w: Vec<i8> = (0..len).map(|i| [i8::MIN, 1, -1, 0][i & 3]).collect();
            let mut expected_i8_i16 = vec![0_i16; len];
            for i in 0..len {
                expected_i8_i16[i] = i16::from(a_i8[i]) * i16::from(b_i8_w[i]);
            }
            let mut candidate_i8_i16 = vec![0_i16; len];
            widen_mul_i8_i16_scalar(&mut candidate_i8_i16, &a_i8, &b_i8_w);
            assert_eq!(candidate_i8_i16, expected_i8_i16);

            let a_u16: Vec<u16> = (0..len).map(|i| [0, 1, 32_768, u16::MAX][i & 3]).collect();
            let b_u16: Vec<u16> = (0..len).map(|i| [u16::MAX, 2, 3, 1][i & 3]).collect();
            let mut expected_u16_u32 = vec![0_u32; len];
            for i in 0..len {
                expected_u16_u32[i] = u32::from(a_u16[i]) * u32::from(b_u16[i]);
            }
            let mut candidate_u16_u32 = vec![0_u32; len];
            widen_mul_u16_u32_scalar(&mut candidate_u16_u32, &a_u16, &b_u16);
            assert_eq!(candidate_u16_u32, expected_u16_u32);

            let a_i16_w: Vec<i16> = (0..len)
                .map(|i| [i16::MIN, -1, 1, i16::MAX][i & 3])
                .collect();
            let b_i16_w: Vec<i16> = (0..len).map(|i| [i16::MIN, 1, -1, 0][i & 3]).collect();
            let mut expected_i16_i32 = vec![0_i32; len];
            for i in 0..len {
                expected_i16_i32[i] = i32::from(a_i16_w[i]) * i32::from(b_i16_w[i]);
            }
            let mut candidate_i16_i32 = vec![0_i32; len];
            widen_mul_i16_i32_scalar(&mut candidate_i16_i32, &a_i16_w, &b_i16_w);
            assert_eq!(candidate_i16_i32, expected_i16_i32);
        }
    }

    #[test]
    fn randomized_dot_and_widen_differential_coverage() {
        let mut rng = XorShift64(0x31d2_9a77_40e6_b5c1);
        for trial in 0..256 {
            let len = if trial == 0 {
                2049
            } else {
                (rng.next() as usize) % 2050
            };
            let a_f32: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
            let b_f32: Vec<f32> = (0..len).map(|_| rng.f32()).collect();
            let mut expected_f32 = 0.0_f64;
            for i in 0..len {
                expected_f32 += f64::from(a_f32[i]) * f64::from(b_f32[i]);
            }
            let candidate_f32 = dot_f32_scalar(&a_f32, &b_f32);
            assert!((expected_f32 - candidate_f32).abs() / expected_f32.abs().max(1.0) <= 1e-12);

            let a_f64: Vec<f64> = (0..len)
                .map(|_| (rng.next() as i64 as f64) / 1.0e12)
                .collect();
            let b_f64: Vec<f64> = (0..len)
                .map(|_| (rng.next() as i64 as f64) / 1.0e12)
                .collect();
            let mut expected_f64 = 0.0_f64;
            for i in 0..len {
                expected_f64 += a_f64[i] * b_f64[i];
            }
            let candidate_f64 = dot_f64_scalar(&a_f64, &b_f64);
            assert!((expected_f64 - candidate_f64).abs() / expected_f64.abs().max(1.0) <= 1e-12);

            let a_i16: Vec<i16> = (0..len).map(|_| rng.next() as i16).collect();
            let b_i16: Vec<i16> = (0..len).map(|_| rng.next() as i16).collect();
            let mut expected_i16 = 0_i64;
            for i in 0..len {
                expected_i16 += i64::from(a_i16[i]) * i64::from(b_i16[i]);
            }
            assert_eq!(dot_i16_scalar(&a_i16, &b_i16), expected_i16);

            let a_u8: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let b_i8: Vec<i8> = (0..len).map(|_| rng.next() as i8).collect();
            let mut expected_mixed = 0_i64;
            for i in 0..len {
                expected_mixed += i64::from(a_u8[i]) * i64::from(b_i8[i]);
            }
            assert_eq!(dot_u8_i8_scalar(&a_u8, &b_i8), expected_mixed);

            let a_u8_w: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let b_u8_w: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let mut expected_u8_u16 = vec![0_u16; len];
            for i in 0..len {
                expected_u8_u16[i] = u16::from(a_u8_w[i]) * u16::from(b_u8_w[i]);
            }
            let mut candidate_u8_u16 = vec![0_u16; len];
            widen_mul_u8_u16_scalar(&mut candidate_u8_u16, &a_u8_w, &b_u8_w);
            assert_eq!(candidate_u8_u16, expected_u8_u16);

            let a_i8: Vec<i8> = (0..len).map(|_| rng.next() as i8).collect();
            let b_i8_w: Vec<i8> = (0..len).map(|_| rng.next() as i8).collect();
            let mut expected_i8_i16 = vec![0_i16; len];
            for i in 0..len {
                expected_i8_i16[i] = i16::from(a_i8[i]) * i16::from(b_i8_w[i]);
            }
            let mut candidate_i8_i16 = vec![0_i16; len];
            widen_mul_i8_i16_scalar(&mut candidate_i8_i16, &a_i8, &b_i8_w);
            assert_eq!(candidate_i8_i16, expected_i8_i16);

            let a_u16: Vec<u16> = (0..len).map(|_| rng.next() as u16).collect();
            let b_u16: Vec<u16> = (0..len).map(|_| rng.next() as u16).collect();
            let mut expected_u16_u32 = vec![0_u32; len];
            for i in 0..len {
                expected_u16_u32[i] = u32::from(a_u16[i]) * u32::from(b_u16[i]);
            }
            let mut candidate_u16_u32 = vec![0_u32; len];
            widen_mul_u16_u32_scalar(&mut candidate_u16_u32, &a_u16, &b_u16);
            assert_eq!(candidate_u16_u32, expected_u16_u32);

            let a_i16_w: Vec<i16> = (0..len).map(|_| rng.next() as i16).collect();
            let b_i16_w: Vec<i16> = (0..len).map(|_| rng.next() as i16).collect();
            let mut expected_i16_i32 = vec![0_i32; len];
            for i in 0..len {
                expected_i16_i32[i] = i32::from(a_i16_w[i]) * i32::from(b_i16_w[i]);
            }
            let mut candidate_i16_i32 = vec![0_i32; len];
            widen_mul_i16_i32_scalar(&mut candidate_i16_i32, &a_i16_w, &b_i16_w);
            assert_eq!(candidate_i16_i32, expected_i16_i32);
        }
    }
}
