#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn dot_f32_scalar(a: *const f32, b: *const f32, len: usize) -> f64 {
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    let mut sum = 0.0_f64;
    for i in 0..len {
        let x = f64::from(a[i]);
        let y = f64::from(b[i]);
        sum += x * y;
    }
    sum
}

#[no_mangle]
pub extern "C" fn dot_f64_scalar(a: *const f64, b: *const f64, len: usize) -> f64 {
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    let mut sum = 0.0_f64;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

#[no_mangle]
pub extern "C" fn dot_i16_scalar(a: *const i16, b: *const i16, len: usize) -> i64 {
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    let mut sum = 0_i64;
    for i in 0..len {
        let x = i64::from(a[i]);
        let y = i64::from(b[i]);
        sum += x * y;
    }
    sum
}

#[no_mangle]
pub extern "C" fn dot_u8_i8_scalar(a: *const u8, b: *const i8, len: usize) -> i64 {
    let a = unsafe { core::slice::from_raw_parts(a, len) };
    let b = unsafe { core::slice::from_raw_parts(b, len) };
    let mut sum = 0_i64;
    for i in 0..len {
        let x = i64::from(a[i]);
        let y = i64::from(b[i]);
        sum += x * y;
    }
    sum
}
