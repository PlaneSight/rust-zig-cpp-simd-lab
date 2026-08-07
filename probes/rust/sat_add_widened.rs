#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn sat_add_u8_widened(
    dst: *mut u8,
    a: *const u8,
    b: *const u8,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let sum = u16::from(a[i]) + u16::from(b[i]);
        dst[i] = sum.min(u16::from(u8::MAX)) as u8;
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i8_widened(
    dst: *mut i8,
    a: *const i8,
    b: *const i8,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let sum = i16::from(a[i]) + i16::from(b[i]);
        dst[i] = sum.clamp(i16::from(i8::MIN), i16::from(i8::MAX)) as i8;
    }
}

#[no_mangle]
pub extern "C" fn sat_add_u16_widened(
    dst: *mut u16,
    a: *const u16,
    b: *const u16,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let sum = u32::from(a[i]) + u32::from(b[i]);
        dst[i] = sum.min(u32::from(u16::MAX)) as u16;
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i16_widened(
    dst: *mut i16,
    a: *const i16,
    b: *const i16,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let sum = i32::from(a[i]) + i32::from(b[i]);
        dst[i] = sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}

#[no_mangle]
pub extern "C" fn sat_add_u32_widened(
    dst: *mut u32,
    a: *const u32,
    b: *const u32,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let sum = u64::from(a[i]) + u64::from(b[i]);
        dst[i] = sum.min(u64::from(u32::MAX)) as u32;
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i32_widened(
    dst: *mut i32,
    a: *const i32,
    b: *const i32,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        let sum = i64::from(a[i]) + i64::from(b[i]);
        dst[i] = sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }
}

#[no_mangle]
pub extern "C" fn sat_add_u64_widened(
    dst: *mut u64,
    a: *const u64,
    b: *const u64,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i].checked_add(b[i]).unwrap_or(u64::MAX);
    }
}

#[no_mangle]
pub extern "C" fn sat_add_i64_widened(
    dst: *mut i64,
    a: *const i64,
    b: *const i64,
    len: usize,
) {
    if len == 0 {
        return;
    }
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    for i in 0..len {
        dst[i] = a[i]
            .checked_add(b[i])
            .unwrap_or(if a[i].is_negative() { i64::MIN } else { i64::MAX });
    }
}
