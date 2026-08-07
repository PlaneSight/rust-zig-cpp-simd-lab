#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn sad_u8_scalar(a: *const u8, b: *const u8, len: usize) -> u64 {
    if len == 0 {
        return 0;
    }

    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y) as u64).sum()
}

#[no_mangle]
pub extern "C" fn sad_u16_scalar(a: *const u16, b: *const u16, len: usize) -> u64 {
    let mut sum = 0_u64;
    for i in 0..len {
        // SAFETY: callers provide valid input arrays of at least `len` u16s.
        // No pointer is dereferenced when len is zero.
        let (x, y) = unsafe { (*a.add(i), *b.add(i)) };
        sum += u64::from(x.abs_diff(y));
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[no_mangle]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sad_u8_avx2(a: *const u8, b: *const u8, len: usize) -> u64 {
    use core::arch::x86_64::*;

    let mut i = 0usize;
    let mut acc = _mm256_setzero_si256();
    while i + 32 <= len {
        let va = unsafe { _mm256_loadu_si256(a.add(i).cast()) };
        let vb = unsafe { _mm256_loadu_si256(b.add(i).cast()) };
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
        i += 32;
    }

    let mut lanes = [0_u64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc) };
    let mut sum = lanes.into_iter().sum::<u64>();
    while i < len {
        sum += unsafe { *a.add(i) }.abs_diff(unsafe { *b.add(i) }) as u64;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[no_mangle]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sad_u16_avx2(a: *const u16, b: *const u16, len: usize) -> u64 {
    use core::arch::x86_64::*;

    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut acc2 = _mm256_setzero_si256();
    let mut acc3 = _mm256_setzero_si256();

    while i + 16 <= len {
        // SAFETY: i..i+16 is in bounds for both caller-provided arrays.
        // Unaligned loads are valid, and this function requires AVX2.
        let (va, vb) = unsafe {
            (
                _mm256_loadu_si256(a.add(i).cast()),
                _mm256_loadu_si256(b.add(i).cast()),
            )
        };
        let diff = _mm256_sub_epi16(_mm256_max_epu16(va, vb), _mm256_min_epu16(va, vb));
        let diff_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(diff));
        let diff_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(diff));
        acc0 = _mm256_add_epi64(
            acc0,
            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff_lo)),
        );
        acc1 = _mm256_add_epi64(
            acc1,
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(diff_lo)),
        );
        acc2 = _mm256_add_epi64(
            acc2,
            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff_hi)),
        );
        acc3 = _mm256_add_epi64(
            acc3,
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(diff_hi)),
        );
        i += 16;
    }

    let mut lanes = [0_u64; 16];
    // SAFETY: lanes provides 16 writable u64 values and unaligned stores are
    // valid for any ordinary Rust array address.
    unsafe {
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), acc0);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(4).cast(), acc1);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(8).cast(), acc2);
        _mm256_storeu_si256(lanes.as_mut_ptr().add(12).cast(), acc3);
    }
    let mut sum = lanes.into_iter().sum::<u64>();

    while i < len {
        // SAFETY: the scalar tail remains within both caller-provided arrays.
        let (x, y) = unsafe { (*a.add(i), *b.add(i)) };
        sum += u64::from(x.abs_diff(y));
        i += 1;
    }
    sum
}
