#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn sad_u8_scalar(a: *const u8, b: *const u8, len: usize) -> u64 {
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    a.iter().zip(b).map(|(&x, &y)| x.abs_diff(y) as u64).sum()
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
