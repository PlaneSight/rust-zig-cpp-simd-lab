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
/// Adds signed bytes element-wise with saturation at the `i8` bounds.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_i8_scalar(dst: &mut [i8], a: &[i8], b: &[i8]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}

/// Adds unsigned 16-bit integers element-wise with saturation at `u16::MAX`.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_u16_scalar(dst: &mut [u16], a: &[u16], b: &[u16]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}

/// Adds signed 16-bit integers element-wise with saturation at the `i16` bounds.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_i16_scalar(dst: &mut [i16], a: &[i16], b: &[i16]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}

/// Adds unsigned 32-bit integers element-wise with saturation at `u32::MAX`.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_u32_scalar(dst: &mut [u32], a: &[u32], b: &[u32]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}

/// Adds signed 32-bit integers element-wise with saturation at the `i32` bounds.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_i32_scalar(dst: &mut [i32], a: &[i32], b: &[i32]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}

/// Adds unsigned 64-bit integers element-wise with saturation at `u64::MAX`.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_u64_scalar(dst: &mut [u64], a: &[u64], b: &[u64]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    for ((out, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *out = x.saturating_add(y);
    }
}

/// Adds signed 64-bit integers element-wise with saturation at the `i64` bounds.
///
/// The three slices must have identical lengths. Rust's borrowing rules make
/// this an out-of-place transform: `dst` cannot overlap either input.
pub fn sat_add_i64_scalar(dst: &mut [i64], a: &[i64], b: &[i64]) {
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
/// Multiplies u32 pairs after widening each operand to u64.
pub fn widen_mul_u32_u64_scalar(dst: &mut [u64], a: &[u32], b: &[u32]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i < dst.len() {
        let x = u64::from(a[i]);
        let y = u64::from(b[i]);
        dst[i] = x * y;
        i += 1;
    }
}

/// Multiplies i32 pairs after widening each operand to i64.
pub fn widen_mul_i32_i64_scalar(dst: &mut [i64], a: &[i32], b: &[i32]) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());

    let mut i = 0;
    while i < dst.len() {
        let x = i64::from(a[i]);
        let y = i64::from(b[i]);
        dst[i] = x * y;
        i += 1;
    }
}

/// Widens each `u8` element to `u16` without changing its numeric value.
pub fn widen_u8_to_u16_scalar(dst: &mut [u16], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = u16::from(value);
    }
}

/// Widens each `u8` element to `u32` without changing its numeric value.
pub fn widen_u8_to_u32_scalar(dst: &mut [u32], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = u32::from(value);
    }
}

/// Sign-extends each `i8` element to `i16`.
pub fn widen_i8_to_i16_scalar(dst: &mut [i16], src: &[i8]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = i16::from(value);
    }
}

/// Widens each `u16` element to `u32` without changing its numeric value.
pub fn widen_u16_to_u32_scalar(dst: &mut [u32], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = u32::from(value);
    }
}

/// Sign-extends each `i16` element to `i32`.
pub fn widen_i16_to_i32_scalar(dst: &mut [i32], src: &[i16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = i32::from(value);
    }
}

/// Converts `u16` values to exactly represented `f32` values.
pub fn convert_u16_to_f32_scalar(dst: &mut [f32], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = f32::from(value);
    }
}

/// Converts `i16` values to exactly represented `f32` values.
pub fn convert_i16_to_f32_scalar(dst: &mut [f32], src: &[i16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = f32::from(value);
    }
}

/// Applies an affine transform after converting each `u8` to `f32`.
pub fn convert_u8_f32_affine_scalar(
    dst: &mut [f32],
    src: &[u8],
    scale: f32,
    bias: f32,
) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = f32::from(value) * scale + bias;
    }
}

/// Converts `f32` to `u16` with total, explicit saturation.
///
/// NaN and values at or below zero produce zero. Values at or above 65535
/// produce `u16::MAX`; finite values in between truncate toward zero.
pub fn f32_to_u16_sat_scalar(dst: &mut [u16], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = if value.is_nan() || value <= 0.0 {
            0
        } else if value >= f32::from(u16::MAX) {
            u16::MAX
        } else {
            value as u16
        };
    }
}

/// Converts finite `f32` values in `[0, 255]` to `u8` by truncating.
pub fn convert_f32_u8_trunc_scalar(dst: &mut [u8], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        assert!(value.is_finite() && (0.0..=255.0).contains(&value));
        *out = value as u8;
    }
}

/// Converts finite `f32` values in `[0, 255]` to `u8`, rounding ties upward.
pub fn convert_f32_u8_round_scalar(dst: &mut [u8], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        assert!(value.is_finite() && (0.0..=255.0).contains(&value));
        *out = (value + 0.5).floor() as u8;
    }
}

/// Converts `f32` values to `u8` with explicit NaN/infinity-safe saturation.
pub fn convert_f32_u8_sat_scalar(dst: &mut [u8], src: &[f32]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = if value.is_nan() || value <= 0.0 {
            0
        } else if value >= 255.0 {
            u8::MAX
        } else {
            value as u8
        };
    }
}

/// Narrows `u16` values by retaining their low eight bits.
pub fn narrow_u16_to_u8_trunc_scalar(dst: &mut [u8], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = (value & u16::from(u8::MAX)) as u8;
    }
}

/// Converts full-range `u16` values to `u8` with integer round-to-nearest.
pub fn narrow_u16_to_u8_round_scalar(dst: &mut [u8], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = ((u32::from(value) + 128) / 257) as u8;
    }
}

/// Narrows `u16` values to `u8`, clamping values above 255.
pub fn narrow_u16_to_u8_sat_scalar(dst: &mut [u8], src: &[u16]) {
    assert_eq!(dst.len(), src.len());
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = value.min(u16::from(u8::MAX)) as u8;
    }
}

/// Packs four bytes per output word in logical little-endian order.
pub fn pack_u8x4_to_u32_scalar(dst: &mut [u32], src: &[u8]) {
    let expected_src_len = dst
        .len()
        .checked_mul(4)
        .expect("destination length overflows packed source length");
    assert_eq!(src.len(), expected_src_len);
    for (out, bytes) in dst.iter_mut().zip(src.chunks_exact(4)) {
        *out = u32::from(bytes[0])
            | (u32::from(bytes[1]) << 8)
            | (u32::from(bytes[2]) << 16)
            | (u32::from(bytes[3]) << 24);
    }
}

/// Unpacks logical little-endian words into four bytes per output group.
pub fn unpack_u32_to_u8x4_scalar(dst: &mut [u8], src: &[u32]) {
    let expected_dst_len = src
        .len()
        .checked_mul(4)
        .expect("source length overflows unpacked destination length");
    assert_eq!(dst.len(), expected_dst_len);
    for (&word, bytes) in src.iter().zip(dst.chunks_exact_mut(4)) {
        bytes[0] = (word & 0xff) as u8;
        bytes[1] = ((word >> 8) & 0xff) as u8;
        bytes[2] = ((word >> 16) & 0xff) as u8;
        bytes[3] = ((word >> 24) & 0xff) as u8;
    }
}

/// Blends two byte slices with a 16-bit fixed-point weight.
///
/// The three slices must have identical lengths. `weight` is an inclusive
/// fixed-point weight in `0..=256`: zero selects `a`, while 256 selects `b`.
/// The weighted sum is rounded to the nearest byte with ties rounded upward.
/// Rust's borrowing rules make this an out-of-place transform: `dst` cannot
/// overlap either input.
pub fn blend_u8_scalar(dst: &mut [u8], a: &[u8], b: &[u8], weight: u16) {
    assert_eq!(dst.len(), a.len());
    assert_eq!(a.len(), b.len());
    assert!(weight <= 256);

    let weight = u32::from(weight);
    let inverse_weight = 256_u32 - weight;
    let mut i = 0;
    while i < dst.len() {
        let weighted_sum =
            u32::from(a[i]) * inverse_weight + u32::from(b[i]) * weight + 128;
        dst[i] = (weighted_sum >> 8) as u8;
        i += 1;
    }
}

/// Applies a three-tap `[1, 2, 1] / 4` horizontal filter to bytes.
///
/// The destination and source slices must have identical lengths. Source
/// indices outside the slice are clamped to the nearest endpoint, and the
/// rounded result is written for every destination element.
/// Rust's borrowing rules make this an out-of-place transform: `dst` cannot
/// overlap `src`.
pub fn convolve3_u8_scalar(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    let n = src.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        dst[0] = src[0];
        return;
    }

    dst[0] =
        ((3 * u32::from(src[0]) + u32::from(src[1]) + 2) >> 2) as u8;
    let mut i = 1;
    while i < n - 1 {
        let weighted_sum = u32::from(src[i - 1])
            + 2 * u32::from(src[i])
            + u32::from(src[i + 1])
            + 2;
        dst[i] = (weighted_sum >> 2) as u8;
        i += 1;
    }
    dst[n - 1] =
        ((u32::from(src[n - 2]) + 3 * u32::from(src[n - 1]) + 2) >> 2) as u8;
}

/// Applies a five-tap `[1, 4, 6, 4, 1] / 16` horizontal filter to bytes.
///
/// The destination and source slices must have identical lengths. Source
/// indices outside the slice are clamped to the nearest endpoint, and the
/// rounded result is written for every destination element.
/// Rust's borrowing rules make this an out-of-place transform: `dst` cannot
/// overlap `src`.
pub fn convolve5_u8_scalar(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len());
    let n = src.len();
    if n == 0 {
        return;
    }
    if n < 5 {
        let last = n - 1;
        for i in 0..n {
            let left2 = i.saturating_sub(2);
            let left1 = i.saturating_sub(1);
            let right1 = i.saturating_add(1).min(last);
            let right2 = i.saturating_add(2).min(last);
            let weighted_sum = u32::from(src[left2])
                + 4 * u32::from(src[left1])
                + 6 * u32::from(src[i])
                + 4 * u32::from(src[right1])
                + u32::from(src[right2])
                + 8;
            dst[i] = (weighted_sum >> 4) as u8;
        }
        return;
    }

    dst[0] =
        ((11 * u32::from(src[0]) + 4 * u32::from(src[1]) + u32::from(src[2]) + 8) >> 4)
            as u8;
    dst[1] = ((5 * u32::from(src[0])
        + 6 * u32::from(src[1])
        + 4 * u32::from(src[2])
        + u32::from(src[3])
        + 8)
        >> 4) as u8;
    let mut i = 2;
    while i < n - 2 {
        let weighted_sum = u32::from(src[i - 2])
            + 4 * u32::from(src[i - 1])
            + 6 * u32::from(src[i])
            + 4 * u32::from(src[i + 1])
            + u32::from(src[i + 2])
            + 8;
        dst[i] = (weighted_sum >> 4) as u8;
        i += 1;
    }
    dst[n - 2] = ((
        u32::from(src[n - 4])
            + 4 * u32::from(src[n - 3])
            + 6 * u32::from(src[n - 2])
            + 5 * u32::from(src[n - 1])
            + 8
    ) >> 4) as u8;
    dst[n - 1] = ((
        u32::from(src[n - 3])
            + 4 * u32::from(src[n - 2])
            + 11 * u32::from(src[n - 1])
            + 8
    ) >> 4) as u8;
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
            let a_u32: Vec<u32> = (0..len).map(|i| [0, 1, 0x8000_0000, u32::MAX][i & 3]).collect();
            let b_u32: Vec<u32> =
                (0..len).map(|i| [u32::MAX, u32::MAX, 2, 1][i & 3]).collect();
            let mut expected_u32_u64 = vec![0_u64; len];
            for i in 0..len {
                expected_u32_u64[i] = u64::from(a_u32[i]) * u64::from(b_u32[i]);
            }
            let mut candidate_u32_u64 = vec![0_u64; len];
            widen_mul_u32_u64_scalar(&mut candidate_u32_u64, &a_u32, &b_u32);
            assert_eq!(candidate_u32_u64, expected_u32_u64);

            let a_i32: Vec<i32> = (0..len).map(|i| [i32::MIN, -1, 1, i32::MAX][i & 3]).collect();
            let b_i32: Vec<i32> = (0..len).map(|i| [i32::MIN, 1, -1, 0][i & 3]).collect();
            let mut expected_i32_i64 = vec![0_i64; len];
            for i in 0..len {
                expected_i32_i64[i] = i64::from(a_i32[i]) * i64::from(b_i32[i]);
            }
            let mut candidate_i32_i64 = vec![0_i64; len];
            widen_mul_i32_i64_scalar(&mut candidate_i32_i64, &a_i32, &b_i32);
            assert_eq!(candidate_i32_i64, expected_i32_i64);
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
            let a_u32_w: Vec<u32> = (0..len).map(|_| rng.next() as u32).collect();
            let b_u32_w: Vec<u32> = (0..len).map(|_| rng.next() as u32).collect();
            let mut expected_u32_u64 = vec![0_u64; len];
            for i in 0..len {
                expected_u32_u64[i] = u64::from(a_u32_w[i]) * u64::from(b_u32_w[i]);
            }
            let mut candidate_u32_u64 = vec![0_u64; len];
            widen_mul_u32_u64_scalar(&mut candidate_u32_u64, &a_u32_w, &b_u32_w);
            assert_eq!(candidate_u32_u64, expected_u32_u64);

            let a_i32_w: Vec<i32> = (0..len).map(|_| rng.next() as i32).collect();
            let b_i32_w: Vec<i32> = (0..len).map(|_| rng.next() as i32).collect();
            let mut expected_i32_i64 = vec![0_i64; len];
            for i in 0..len {
                expected_i32_i64[i] = i64::from(a_i32_w[i]) * i64::from(b_i32_w[i]);
            }
            let mut candidate_i32_i64 = vec![0_i64; len];
            widen_mul_i32_i64_scalar(&mut candidate_i32_i64, &a_i32_w, &b_i32_w);
            assert_eq!(candidate_i32_i64, expected_i32_i64);
        }
    }
    #[test]
    fn saturating_add_integer_apis_match_independent_widened_references() {
        const LENGTHS: [usize; 20] = [
            0, 1, 2, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 31, 32, 33, 63, 64, 65,
        ];
        let mut rng = XorShift64(0x4a21_8c73_d5e9_0b6f);

        macro_rules! check_signed {
            ($len:ident, $ty:ty, $wide:ty, $function:ident) => {{
                let a: Vec<$ty> = (0..$len)
                    .map(|i| match i {
                        0 => <$ty>::MIN,
                        1 => <$ty>::MIN + 1,
                        2 => -1,
                        3 => 0,
                        4 => 1,
                        5 => <$ty>::MAX - 1,
                        6 => <$ty>::MAX,
                        _ => rng.next() as $ty,
                    })
                    .collect();
                let b: Vec<$ty> = (0..$len)
                    .map(|i| match i {
                        0 => <$ty>::MIN,
                        1 => <$ty>::MAX,
                        2 => <$ty>::MAX,
                        3 => <$ty>::MIN,
                        4 => -1,
                        5 => 1,
                        6 => <$ty>::MAX,
                        _ => rng.next() as $ty,
                    })
                    .collect();
                let expected: Vec<$ty> = a
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| {
                        let sum = <$wide>::from(x) + <$wide>::from(y);
                        sum.clamp(<$wide>::from(<$ty>::MIN), <$wide>::from(<$ty>::MAX)) as $ty
                    })
                    .collect();
                let mut candidate = vec![0 as $ty; $len];
                $function(&mut candidate, &a, &b);
                assert_eq!(
                    candidate,
                    expected,
                    "signed type {} len={}",
                    stringify!($ty),
                    $len
                );
            }};
        }
        macro_rules! check_unsigned {
            ($len:ident, $ty:ty, $wide:ty, $function:ident) => {{
                let a: Vec<$ty> = (0..$len)
                    .map(|i| match i {
                        0 => 0,
                        1 => 1,
                        2 => 2,
                        3 => <$ty>::MAX / 2,
                        4 => <$ty>::MAX - 1,
                        5 => <$ty>::MAX,
                        6 => <$ty>::MAX,
                        _ => rng.next() as $ty,
                    })
                    .collect();
                let b: Vec<$ty> = (0..$len)
                    .map(|i| match i {
                        0 => <$ty>::MAX,
                        1 => <$ty>::MAX,
                        2 => <$ty>::MAX,
                        3 => 1,
                        4 => 2,
                        5 => <$ty>::MAX,
                        6 => 0,
                        _ => rng.next() as $ty,
                    })
                    .collect();
                let expected: Vec<$ty> = a
                    .iter()
                    .zip(&b)
                    .map(|(&x, &y)| {
                        let sum = <$wide>::from(x) + <$wide>::from(y);
                        sum.min(<$wide>::from(<$ty>::MAX)) as $ty
                    })
                    .collect();
                let mut candidate = vec![0 as $ty; $len];
                $function(&mut candidate, &a, &b);
                assert_eq!(
                    candidate,
                    expected,
                    "unsigned type {} len={}",
                    stringify!($ty),
                    $len
                );
            }};
        }

        for &len in &LENGTHS {
            check_signed!(len, i8, i128, sat_add_i8_scalar);
            check_unsigned!(len, u16, u128, sat_add_u16_scalar);
            check_signed!(len, i16, i128, sat_add_i16_scalar);
            check_unsigned!(len, u32, u128, sat_add_u32_scalar);
            check_signed!(len, i32, i128, sat_add_i32_scalar);
            check_unsigned!(len, u64, u128, sat_add_u64_scalar);
            check_signed!(len, i64, i128, sat_add_i64_scalar);
        }


        let i64_a = [i64::MAX, i64::MIN, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1];
        let i64_b = [1_i64, -1, i64::MAX, i64::MIN, 1, -1];
        let mut i64_dst = [0_i64; 6];
        sat_add_i64_scalar(&mut i64_dst, &i64_a, &i64_b);
        assert_eq!(
            i64_dst,
            [i64::MAX, i64::MIN, i64::MAX, i64::MIN, i64::MAX, i64::MIN]
        );
    }
    #[test]
    fn mixed_width_scalar_edge_lengths_extrema_and_packings() {
        const LENGTHS: [usize; 16] = [
            0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129,
        ];

        for &len in &LENGTHS {
            let u8_src: Vec<u8> = (0..len)
                .map(|i| [0, 1, 127, 128, 254, u8::MAX][i % 6])
                .collect();
            let mut u16_actual = vec![0_u16; len];
            let mut u32_actual = vec![0_u32; len];
            widen_u8_to_u16_scalar(&mut u16_actual, &u8_src);
            widen_u8_to_u32_scalar(&mut u32_actual, &u8_src);
            assert_eq!(
                u16_actual,
                u8_src.iter().map(|&value| u16::from(value)).collect::<Vec<_>>()
            );
            assert_eq!(
                u32_actual,
                u8_src.iter().map(|&value| u32::from(value)).collect::<Vec<_>>()
            );

            let i8_src: Vec<i8> = (0..len)
                .map(|i| [i8::MIN, -1, 0, 1, i8::MAX][i % 5])
                .collect();
            let mut i16_actual = vec![0_i16; len];
            widen_i8_to_i16_scalar(&mut i16_actual, &i8_src);
            assert_eq!(
                i16_actual,
                i8_src.iter().map(|&value| i16::from(value)).collect::<Vec<_>>()
            );

            let u16_src: Vec<u16> = (0..len)
                .map(|i| [0, 1, 255, 256, 32_768, u16::MAX][i % 6])
                .collect();
            let mut u32_from_u16 = vec![0_u32; len];
            let mut f32_from_u16 = vec![0.0_f32; len];
            widen_u16_to_u32_scalar(&mut u32_from_u16, &u16_src);
            convert_u16_to_f32_scalar(&mut f32_from_u16, &u16_src);
            assert_eq!(
                u32_from_u16,
                u16_src.iter().map(|&value| u32::from(value)).collect::<Vec<_>>()
            );
            assert_eq!(
                f32_from_u16,
                u16_src.iter().map(|&value| f32::from(value)).collect::<Vec<_>>()
            );

            let i16_src: Vec<i16> = (0..len)
                .map(|i| [i16::MIN, -1, 0, 1, i16::MAX][i % 5])
                .collect();
            let mut i32_actual = vec![0_i32; len];
            let mut f32_from_i16 = vec![0.0_f32; len];
            widen_i16_to_i32_scalar(&mut i32_actual, &i16_src);
            convert_i16_to_f32_scalar(&mut f32_from_i16, &i16_src);
            assert_eq!(
                i32_actual,
                i16_src.iter().map(|&value| i32::from(value)).collect::<Vec<_>>()
            );
            assert_eq!(
                f32_from_i16,
                i16_src.iter().map(|&value| f32::from(value)).collect::<Vec<_>>()
            );

            let mut affine_actual = vec![0.0_f32; len];
            convert_u8_f32_affine_scalar(&mut affine_actual, &u8_src, 1.25, -3.5);
            let affine_expected: Vec<f32> = u8_src
                .iter()
                .map(|&value| f32::from(value) * 1.25 - 3.5)
                .collect();
            assert_eq!(affine_actual, affine_expected);

            let f32_to_u16_src: Vec<f32> = (0..len)
                .map(|i| {
                    [
                        f32::NAN,
                        f32::NEG_INFINITY,
                        -0.5,
                        -0.0,
                        0.5,
                        1.75,
                        65_534.75,
                        65_535.0,
                        f32::INFINITY,
                        f32::MAX,
                    ][i % 10]
                })
                .collect();
            let mut f32_to_u16_actual = vec![0_u16; len];
            f32_to_u16_sat_scalar(&mut f32_to_u16_actual, &f32_to_u16_src);
            let f32_to_u16_expected: Vec<u16> = f32_to_u16_src
                .iter()
                .map(|&value| {
                    if value.is_nan() || value <= 0.0 {
                        0
                    } else if value >= f32::from(u16::MAX) {
                        u16::MAX
                    } else {
                        value as u16
                    }
                })
                .collect();
            assert_eq!(f32_to_u16_actual, f32_to_u16_expected);

            let valid_f32_src: Vec<f32> = (0..len)
                .map(|i| [0.0, 0.25, 0.5, 1.49, 1.5, 1.51, 254.5, 255.0][i % 8])
                .collect();
            let mut trunc_actual = vec![0_u8; len];
            let mut round_actual = vec![0_u8; len];
            convert_f32_u8_trunc_scalar(&mut trunc_actual, &valid_f32_src);
            convert_f32_u8_round_scalar(&mut round_actual, &valid_f32_src);
            assert_eq!(
                trunc_actual,
                valid_f32_src.iter().map(|&value| value as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                round_actual,
                valid_f32_src
                    .iter()
                    .map(|&value| (value + 0.5).floor() as u8)
                    .collect::<Vec<_>>()
            );

            let saturating_f32_src: Vec<f32> = (0..len)
                .map(|i| {
                    [
                        f32::NAN,
                        f32::NEG_INFINITY,
                        -0.5,
                        -0.0,
                        0.5,
                        128.75,
                        254.99,
                        255.0,
                        f32::INFINITY,
                    ][i % 9]
                })
                .collect();
            let mut saturating_actual = vec![0_u8; len];
            convert_f32_u8_sat_scalar(&mut saturating_actual, &saturating_f32_src);
            let saturating_expected: Vec<u8> = saturating_f32_src
                .iter()
                .map(|&value| {
                    if value.is_nan() || value <= 0.0 {
                        0
                    } else if value >= 255.0 {
                        u8::MAX
                    } else {
                        value as u8
                    }
                })
                .collect();
            assert_eq!(saturating_actual, saturating_expected);

            let mut narrow_trunc = vec![0_u8; len];
            let mut narrow_round = vec![0_u8; len];
            let mut narrow_sat = vec![0_u8; len];
            narrow_u16_to_u8_trunc_scalar(&mut narrow_trunc, &u16_src);
            narrow_u16_to_u8_round_scalar(&mut narrow_round, &u16_src);
            narrow_u16_to_u8_sat_scalar(&mut narrow_sat, &u16_src);
            assert_eq!(
                narrow_trunc,
                u16_src
                    .iter()
                    .map(|&value| (value & 0xff) as u8)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                narrow_round,
                u16_src
                    .iter()
                    .map(|&value| ((u32::from(value) + 128) / 257) as u8)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                narrow_sat,
                u16_src
                    .iter()
                    .map(|&value| value.min(u16::from(u8::MAX)) as u8)
                    .collect::<Vec<_>>()
            );
        }

        const GROUPS: [usize; 12] = [0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 33];
        for &groups in &GROUPS {
            let bytes: Vec<u8> = (0..groups * 4)
                .map(|i| [0, 1, 2, 127, 128, 254, 255][i % 7])
                .collect();
            let mut packed = vec![0_u32; groups];
            pack_u8x4_to_u32_scalar(&mut packed, &bytes);
            let expected_packed: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    u32::from(chunk[0])
                        | (u32::from(chunk[1]) << 8)
                        | (u32::from(chunk[2]) << 16)
                        | (u32::from(chunk[3]) << 24)
                })
                .collect();
            assert_eq!(packed, expected_packed, "groups={groups}");

            let mut unpacked = vec![0_u8; groups * 4];
            unpack_u32_to_u8x4_scalar(&mut unpacked, &packed);
            assert_eq!(unpacked, bytes, "groups={groups}");
        }
    }

    #[test]
    fn mixed_width_scalar_randomized_differential_coverage() {
        let mut rng = XorShift64(0x9d4c_2a71_f803_b6e5);
        for trial in 0..256 {
            let len = if trial < 32 {
                trial
            } else {
                (rng.next() as usize) % 258
            };
            let u8_src: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let u16_src: Vec<u16> = (0..len).map(|_| rng.next() as u16).collect();
            let i8_src: Vec<i8> = (0..len).map(|_| rng.next() as i8).collect();
            let i16_src: Vec<i16> = (0..len).map(|_| rng.next() as i16).collect();

            let mut u16_actual = vec![0_u16; len];
            let mut u32_actual = vec![0_u32; len];
            let mut i16_actual = vec![0_i16; len];
            let mut u32_from_u16 = vec![0_u32; len];
            let mut i32_actual = vec![0_i32; len];
            widen_u8_to_u16_scalar(&mut u16_actual, &u8_src);
            widen_u8_to_u32_scalar(&mut u32_actual, &u8_src);
            widen_i8_to_i16_scalar(&mut i16_actual, &i8_src);
            widen_u16_to_u32_scalar(&mut u32_from_u16, &u16_src);
            widen_i16_to_i32_scalar(&mut i32_actual, &i16_src);
            assert_eq!(u16_actual, u8_src.iter().map(|&x| u16::from(x)).collect::<Vec<_>>());
            assert_eq!(u32_actual, u8_src.iter().map(|&x| u32::from(x)).collect::<Vec<_>>());
            assert_eq!(i16_actual, i8_src.iter().map(|&x| i16::from(x)).collect::<Vec<_>>());
            assert_eq!(
                u32_from_u16,
                u16_src.iter().map(|&x| u32::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(i32_actual, i16_src.iter().map(|&x| i32::from(x)).collect::<Vec<_>>());

            let mut converted_u16 = vec![0.0_f32; len];
            let mut converted_i16 = vec![0.0_f32; len];
            convert_u16_to_f32_scalar(&mut converted_u16, &u16_src);
            convert_i16_to_f32_scalar(&mut converted_i16, &i16_src);
            assert_eq!(
                converted_u16,
                u16_src.iter().map(|&x| f32::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                converted_i16,
                i16_src.iter().map(|&x| f32::from(x)).collect::<Vec<_>>()
            );

            let mut affine_actual = vec![0.0_f32; len];
            convert_u8_f32_affine_scalar(&mut affine_actual, &u8_src, -0.75, 11.25);
            for (i, &value) in u8_src.iter().enumerate() {
                assert_eq!(affine_actual[i], f32::from(value) * -0.75 + 11.25);
            }

            let f32_u16_src: Vec<f32> = (0..len)
                .map(|_| (rng.next() % 70_001) as f32 - 2_000.0)
                .collect();
            let mut f32_u16_actual = vec![0_u16; len];
            f32_to_u16_sat_scalar(&mut f32_u16_actual, &f32_u16_src);
            for (actual, &value) in f32_u16_actual.iter().zip(&f32_u16_src) {
                let expected = if value <= 0.0 {
                    0
                } else if value >= f32::from(u16::MAX) {
                    u16::MAX
                } else {
                    value as u16
                };
                assert_eq!(*actual, expected);
            }

            let valid_f32_src: Vec<f32> = (0..len)
                .map(|_| (rng.next() % 255_001) as f32 / 1_000.0)
                .collect();
            let mut trunc_actual = vec![0_u8; len];
            let mut round_actual = vec![0_u8; len];
            convert_f32_u8_trunc_scalar(&mut trunc_actual, &valid_f32_src);
            convert_f32_u8_round_scalar(&mut round_actual, &valid_f32_src);
            for (i, &value) in valid_f32_src.iter().enumerate() {
                assert_eq!(trunc_actual[i], value as u8);
                assert_eq!(round_actual[i], (value + 0.5).floor() as u8);
            }

            let mut sat_actual = vec![0_u8; len];
            convert_f32_u8_sat_scalar(&mut sat_actual, &f32_u16_src);
            for (actual, &value) in sat_actual.iter().zip(&f32_u16_src) {
                let expected = if value <= 0.0 {
                    0
                } else if value >= 255.0 {
                    u8::MAX
                } else {
                    value as u8
                };
                assert_eq!(*actual, expected);
            }

            let mut narrow_trunc = vec![0_u8; len];
            let mut narrow_round = vec![0_u8; len];
            let mut narrow_sat = vec![0_u8; len];
            narrow_u16_to_u8_trunc_scalar(&mut narrow_trunc, &u16_src);
            narrow_u16_to_u8_round_scalar(&mut narrow_round, &u16_src);
            narrow_u16_to_u8_sat_scalar(&mut narrow_sat, &u16_src);
            for (i, &value) in u16_src.iter().enumerate() {
                assert_eq!(narrow_trunc[i], (value & 0xff) as u8);
                assert_eq!(narrow_round[i], ((u32::from(value) + 128) / 257) as u8);
                assert_eq!(narrow_sat[i], value.min(u16::from(u8::MAX)) as u8);
            }

            let groups = (rng.next() as usize) % 66;
            let bytes: Vec<u8> = (0..groups * 4).map(|_| rng.next() as u8).collect();
            let mut packed = vec![0_u32; groups];
            pack_u8x4_to_u32_scalar(&mut packed, &bytes);
            let mut unpacked = vec![0_u8; groups * 4];
            unpack_u32_to_u8x4_scalar(&mut unpacked, &packed);
            assert_eq!(unpacked, bytes, "trial={trial}, groups={groups}");
        }
    }

    fn reference_blend_u8(a: &[u8], b: &[u8], weight: u16) -> Vec<u8> {
        let weight = u32::from(weight);
        let inverse_weight = 256_u32 - weight;
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                ((u32::from(x) * inverse_weight + u32::from(y) * weight + 128) >> 8) as u8
            })
            .collect()
    }

    fn reference_convolve3_u8(src: &[u8]) -> Vec<u8> {
        if src.is_empty() {
            return Vec::new();
        }
        let last = src.len() - 1;
        (0..src.len())
            .map(|i| {
                let left = src[i.saturating_sub(1)];
                let right = src[i.saturating_add(1).min(last)];
                ((u32::from(left) + 2 * u32::from(src[i]) + u32::from(right) + 2) >> 2) as u8
            })
            .collect()
    }

    fn reference_convolve5_u8(src: &[u8]) -> Vec<u8> {
        if src.is_empty() {
            return Vec::new();
        }
        let last = src.len() - 1;
        (0..src.len())
            .map(|i| {
                let left2 = src[i.saturating_sub(2)];
                let left = src[i.saturating_sub(1)];
                let right = src[i.saturating_add(1).min(last)];
                let right2 = src[i.saturating_add(2).min(last)];
                ((u32::from(left2)
                    + 4 * u32::from(left)
                    + 6 * u32::from(src[i])
                    + 4 * u32::from(right)
                    + u32::from(right2)
                    + 8)
                    >> 4) as u8
            })
            .collect()
    }

    fn assert_convolutions_match_references(src: &[u8]) {
        let expected3 = reference_convolve3_u8(src);
        let expected5 = reference_convolve5_u8(src);
        let mut actual3 = vec![0_u8; src.len()];
        let mut actual5 = vec![0_u8; src.len()];
        convolve3_u8_scalar(&mut actual3, src);
        convolve5_u8_scalar(&mut actual5, src);
        assert_eq!(actual3, expected3, "convolve3 len={}", src.len());
        assert_eq!(actual5, expected5, "convolve5 len={}", src.len());
    }

    #[test]
    fn image_kernels_cover_edges_extrema_and_random_lengths() {
        const LENGTHS: [usize; 19] = [
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127,
        ];
        const WEIGHTS: [u16; 6] = [0, 1, 77, 128, 255, 256];
        let mut rng = XorShift64(0x71e3_4a29_c805_d6f1);

        for &len in &LENGTHS {
            let a: Vec<u8> = (0..len)
                .map(|i| match i {
                    0 => 0,
                    1 => u8::MAX,
                    2 => 1,
                    3 => u8::MAX - 1,
                    _ => rng.next() as u8,
                })
                .collect();
            let b: Vec<u8> = (0..len)
                .map(|i| match i {
                    0 => u8::MAX,
                    1 => 0,
                    2 => u8::MAX,
                    3 => 1,
                    _ => rng.next() as u8,
                })
                .collect();
            for &weight in &WEIGHTS {
                let expected = reference_blend_u8(&a, &b, weight);
                let mut actual = vec![0_u8; len];
                blend_u8_scalar(&mut actual, &a, &b, weight);
                assert_eq!(actual, expected, "blend len={len}, weight={weight}");
            }

            let source: Vec<u8> = (0..len)
                .map(|i| match i {
                    0 => 0,
                    1 => u8::MAX,
                    2 => 1,
                    3 => u8::MAX - 1,
                    _ => rng.next() as u8,
                })
                .collect();
            assert_convolutions_match_references(&source);
            for &value in &[0_u8, u8::MAX] {
                assert_convolutions_match_references(&vec![value; len]);
            }
        }

        for trial in 0..256 {
            let len = if trial < 6 {
                trial
            } else {
                (rng.next() as usize) % 258
            };
            let a: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let b: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let weight = (rng.next() % 257) as u16;
            let expected = reference_blend_u8(&a, &b, weight);
            let mut actual = vec![0_u8; len];
            blend_u8_scalar(&mut actual, &a, &b, weight);
            assert_eq!(actual, expected, "random blend trial={trial}, len={len}");

            let source: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            assert_convolutions_match_references(&source);
        }
    }

    #[test]
    #[should_panic]
    fn blend_u8_rejects_weight_above_fixed_point_range() {
        let mut dst = [0_u8; 1];
        blend_u8_scalar(&mut dst, &[0], &[u8::MAX], 257);
    }
    #[test]
    #[should_panic]
    fn mixed_width_f32_u8_checked_conversions_reject_invalid_inputs() {
        let mut dst = [0_u8; 1];
        convert_f32_u8_trunc_scalar(&mut dst, &[f32::NAN]);
    }
}
