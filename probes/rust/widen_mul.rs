#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn widen_mul_u8_u16_scalar(
    dst: *mut u16,
    a: *const u8,
    b: *const u8,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let x = u16::from(a[i]);
        let y = u16::from(b[i]);
        dst[i] = x * y;
    }
}

#[no_mangle]
pub extern "C" fn widen_mul_i8_i16_scalar(
    dst: *mut i16,
    a: *const i8,
    b: *const i8,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let x = i16::from(a[i]);
        let y = i16::from(b[i]);
        dst[i] = x * y;
    }
}

#[no_mangle]
pub extern "C" fn widen_mul_u16_u32_scalar(
    dst: *mut u32,
    a: *const u16,
    b: *const u16,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let x = u32::from(a[i]);
        let y = u32::from(b[i]);
        dst[i] = x * y;
    }
}

#[no_mangle]
pub extern "C" fn widen_mul_i16_i32_scalar(
    dst: *mut i32,
    a: *const i16,
    b: *const i16,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let x = i32::from(a[i]);
        let y = i32::from(b[i]);
        dst[i] = x * y;
    }
}
#[no_mangle]
pub extern "C" fn widen_mul_u32_u64_scalar(
    dst: *mut u64,
    a: *const u32,
    b: *const u32,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let x = u64::from(a[i]);
        let y = u64::from(b[i]);
        dst[i] = x * y;
    }
}

#[no_mangle]
pub extern "C" fn widen_mul_i32_i64_scalar(
    dst: *mut i64,
    a: *const i32,
    b: *const i32,
    len: usize,
) {
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let x = i64::from(a[i]);
        let y = i64::from(b[i]);
        dst[i] = x * y;
    }
}
