#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn sat_add_u8_scalar(dst: *mut u8, a: *const u8, b: *const u8, len: usize) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i8_scalar(dst: *mut i8, a: *const i8, b: *const i8, len: usize) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_u16_scalar(
    dst: *mut u16,
    a: *const u16,
    b: *const u16,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i16_scalar(
    dst: *mut i16,
    a: *const i16,
    b: *const i16,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_u32_scalar(
    dst: *mut u32,
    a: *const u32,
    b: *const u32,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i32_scalar(
    dst: *mut i32,
    a: *const i32,
    b: *const i32,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_u64_scalar(
    dst: *mut u64,
    a: *const u64,
    b: *const u64,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i64_scalar(
    dst: *mut i64,
    a: *const i64,
    b: *const i64,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].saturating_add(b[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[no_mangle]
#[target_feature(enable = "avx2")]
pub unsafe extern "C" fn sat_add_u8_avx2(dst: *mut u8, a: *const u8, b: *const u8, len: usize) {
    use core::arch::x86_64::*;
    let mut i = 0usize;
    while i + 32 <= len {
        let va = unsafe { _mm256_loadu_si256(a.add(i).cast()) };
        let vb = unsafe { _mm256_loadu_si256(b.add(i).cast()) };
        let out = _mm256_adds_epu8(va, vb);
        unsafe { _mm256_storeu_si256(dst.add(i).cast(), out) };
        i += 32;
    }
    while i < len {
        unsafe { *dst.add(i) = (*a.add(i)).saturating_add(*b.add(i)) };
        i += 1;
    }
}
