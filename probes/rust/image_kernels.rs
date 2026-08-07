#![no_std]
#![allow(clippy::missing_safety_doc)]

#[no_mangle]
pub extern "C" fn blend_u8_scalar(dst: *mut u8, a: *const u8, b: *const u8, weight: u16, len: usize) {
  assert!(weight <= 256);
  if len == 0 {
    return;
  }
  // SAFETY: For `len > 0`, the caller must provide non-null, properly
  // aligned pointers to `len` readable `a`/`b` elements and writable `dst`
  // elements. The ranges must not overlap.
  let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
  let a = unsafe { core::slice::from_raw_parts(a, len) };
  let b = unsafe { core::slice::from_raw_parts(b, len) };
  let wide_weight = u32::from(weight);
  let inverse_weight = 256_u32 - wide_weight;
  for i in 0..len {
    let weighted_a = u32::from(a[i]) * inverse_weight;
    let weighted_b = u32::from(b[i]) * wide_weight;
    dst[i] = ((weighted_a + weighted_b + 128) >> 8) as u8;
  }
}

#[no_mangle]
pub extern "C" fn convolve3_u8_scalar(dst: *mut u8, src: *const u8, len: usize) {
  if len == 0 {
    return;
  }
  // SAFETY: For `len > 0`, the caller must provide non-null, properly
  // aligned pointers to `len` readable `src` elements and writable `dst`
  // elements. The ranges must not overlap.
  let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
  let src = unsafe { core::slice::from_raw_parts(src, len) };
  if len == 1 {
    dst[0] = src[0];
    return;
  }

  dst[0] = ((3 * u32::from(src[0]) + u32::from(src[1]) + 2) >> 2) as u8;
  for i in 1..len - 1 {
    let sum = u32::from(src[i - 1]) + 2 * u32::from(src[i]) + u32::from(src[i + 1]) + 2;
    dst[i] = (sum >> 2) as u8;
  }
  dst[len - 1] = ((u32::from(src[len - 2]) + 3 * u32::from(src[len - 1]) + 2) >> 2) as u8;
}

#[no_mangle]
pub extern "C" fn convolve5_u8_scalar(dst: *mut u8, src: *const u8, len: usize) {
  if len == 0 {
    return;
  }
  // SAFETY: For `len > 0`, the caller must provide non-null, properly
  // aligned pointers to `len` readable `src` elements and writable `dst`
  // elements. The ranges must not overlap.
  let dst = unsafe { core::slice::from_raw_parts_mut(dst, len) };
  let src = unsafe { core::slice::from_raw_parts(src, len) };
  if len < 5 {
    for i in 0..len {
      let left2 = i.saturating_sub(2);
      let left1 = i.saturating_sub(1);
      let right1 = if i + 1 < len { i + 1 } else { len - 1 };
      let right2 = if len - 1 - i >= 2 { i + 2 } else { len - 1 };
      let sum = u32::from(src[left2])
        + 4 * u32::from(src[left1])
        + 6 * u32::from(src[i])
        + 4 * u32::from(src[right1])
        + u32::from(src[right2])
        + 8;
      dst[i] = (sum >> 4) as u8;
    }
    return;
  }

  dst[0] = ((11 * u32::from(src[0]) + 4 * u32::from(src[1]) + u32::from(src[2]) + 8) >> 4) as u8;
  dst[1] = ((5 * u32::from(src[0]) + 6 * u32::from(src[1]) + 4 * u32::from(src[2]) + u32::from(src[3]) + 8) >> 4) as u8;
  for i in 2..len - 2 {
    let sum = u32::from(src[i - 2])
      + 4 * u32::from(src[i - 1])
      + 6 * u32::from(src[i])
      + 4 * u32::from(src[i + 1])
      + u32::from(src[i + 2])
      + 8;
    dst[i] = (sum >> 4) as u8;
  }
  dst[len - 2] = ((u32::from(src[len - 4])
    + 4 * u32::from(src[len - 3])
    + 6 * u32::from(src[len - 2])
    + 5 * u32::from(src[len - 1])
    + 8)
    >> 4) as u8;
  dst[len - 1] =
    ((u32::from(src[len - 3]) + 4 * u32::from(src[len - 2]) + 11 * u32::from(src[len - 1]) + 8) >> 4) as u8;
}
