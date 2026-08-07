#![no_std]
#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn widen_u8_to_u16_scalar(dst: *mut u16, src: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = u16::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn widen_u8_to_u32_scalar(dst: *mut u32, src: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = u32::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn widen_i8_to_i16_scalar(dst: *mut i16, src: *const i8, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = i16::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn widen_u16_to_u32_scalar(dst: *mut u32, src: *const u16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = u32::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn widen_i16_to_i32_scalar(dst: *mut i32, src: *const i16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = i32::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn convert_u16_to_f32_scalar(dst: *mut f32, src: *const u16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = f32::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn convert_i16_to_f32_scalar(dst: *mut f32, src: *const i16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = f32::from(src[i]);
    }
}

#[no_mangle]
pub extern "C" fn convert_u8_f32_affine_scalar(dst: *mut f32, src: *const u8, scale: f32, bias: f32, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = f32::from(src[i]) * scale + bias;
    }
}

#[no_mangle]
pub extern "C" fn f32_to_u16_sat_scalar(dst: *mut u16, src: *const f32, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        let value = src[i];
        dst[i] = if value.is_nan() || value <= 0.0 {
            0
        } else if value >= f32::from(u16::MAX) {
            u16::MAX
        } else {
            value as u16
        };
    }
}

#[no_mangle]
pub extern "C" fn convert_f32_u8_trunc_scalar(dst: *mut u8, src: *const f32, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        let value = src[i];
        assert!(value.is_finite() && (0.0..=255.0).contains(&value));
        dst[i] = value as u8;
    }
}

#[no_mangle]
pub extern "C" fn convert_f32_u8_round_scalar(dst: *mut u8, src: *const f32, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        let value = src[i];
        assert!(value.is_finite() && (0.0..=255.0).contains(&value));
        dst[i] = (value + 0.5) as u8;
    }
}

#[no_mangle]
pub extern "C" fn convert_f32_u8_sat_scalar(dst: *mut u8, src: *const f32, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        let value = src[i];
        dst[i] = if value.is_nan() || value <= 0.0 {
            0
        } else if value >= 255.0 {
            u8::MAX
        } else {
            value as u8
        };
    }
}

#[no_mangle]
pub extern "C" fn narrow_u16_to_u8_trunc_scalar(dst: *mut u8, src: *const u16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = (src[i] & u16::from(u8::MAX)) as u8;
    }
}

#[no_mangle]
pub extern "C" fn narrow_u16_to_u8_round_scalar(dst: *mut u8, src: *const u16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = ((u32::from(src[i]) + 128) / 257) as u8;
    }
}

#[no_mangle]
pub extern "C" fn narrow_u16_to_u8_sat_scalar(dst: *mut u8, src: *const u16, len: usize) {
    if len == 0 {
        return;
    }
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` elements and writable `dst`
    // elements. The two ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        dst[i] = src[i].min(u16::from(u8::MAX)) as u8;
    }
}

#[no_mangle]
pub extern "C" fn pack_u8x4_to_u32_scalar(dst: *mut u32, src: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    let src_len = len.checked_mul(4).expect("packed source length overflows usize");
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` writable `dst` words and `4 * len` readable
    // `src` bytes. The ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
    let src = unsafe { core::slice::from_raw_parts(src, src_len) };
    for i in 0..len {
        let base = i * 4;
        dst[i] = u32::from(src[base])
            | (u32::from(src[base + 1]) << 8)
            | (u32::from(src[base + 2]) << 16)
            | (u32::from(src[base + 3]) << 24);
    }
}

#[no_mangle]
pub extern "C" fn unpack_u32_to_u8x4_scalar(dst: *mut u8, src: *const u32, len: usize) {
    if len == 0 {
        return;
    }
    let dst_len = len.checked_mul(4).expect("unpacked destination length overflows usize");
    // SAFETY: For `len > 0`, the caller must provide non-null, properly
    // aligned pointers to `len` readable `src` words and `4 * len` writable
    // `dst` bytes. The ranges must not overlap.
    let dst = unsafe { core::slice::from_raw_parts_mut(dst, dst_len) };
    let src = unsafe { core::slice::from_raw_parts(src, len) };
    for i in 0..len {
        let base = i * 4;
        let word = src[i];
        dst[base] = (word & 0xff) as u8;
        dst[base + 1] = ((word >> 8) & 0xff) as u8;
        dst[base + 2] = ((word >> 16) & 0xff) as u8;
        dst[base + 3] = ((word >> 24) & 0xff) as u8;
    }
}
