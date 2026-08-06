// Standalone codegen probes for rustc/LLVM.
//
// Example:
//   rustc -O --crate-type=lib --emit=asm probes/rust/clamp.rs
//
// Stable Rust has no stable portable SIMD API, so these probes intentionally
// exercise scalar source/autovectorization. ISA-specific versions belong in
// the main Rust crate beside the existing std::arch kernels.

#[no_mangle]
pub extern "C" fn clamp_u8(
    c: u8,
    a1: u8,
    a2: u8,
    a3: u8,
    a4: u8,
    a5: u8,
    a6: u8,
    a7: u8,
    a8: u8,
) -> u8 {
    let lo = a1.min(a2).min(a3).min(a4).min(a5).min(a6).min(a7).min(a8);
    let hi = a1.max(a2).max(a3).max(a4).max(a5).max(a6).max(a7).max(a8);
    c.clamp(lo, hi)
}

#[no_mangle]
pub extern "C" fn clamp_u16(
    c: u16,
    a1: u16,
    a2: u16,
    a3: u16,
    a4: u16,
    a5: u16,
    a6: u16,
    a7: u16,
    a8: u16,
) -> u16 {
    let lo = a1.min(a2).min(a3).min(a4).min(a5).min(a6).min(a7).min(a8);
    let hi = a1.max(a2).max(a3).max(a4).max(a5).max(a6).max(a7).max(a8);
    c.clamp(lo, hi)
}

#[no_mangle]
pub extern "C" fn clamp_f32(
    c: f32,
    a1: f32,
    a2: f32,
    a3: f32,
    a4: f32,
    a5: f32,
    a6: f32,
    a7: f32,
    a8: f32,
) -> f32 {
    let lo = a1.min(a2).min(a3).min(a4).min(a5).min(a6).min(a7).min(a8);
    let hi = a1.max(a2).max(a3).max(a4).max(a5).max(a6).max(a7).max(a8);
    c.max(lo).min(hi)
}

#[no_mangle]
pub extern "C" fn clamp_f64(
    c: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    a7: f64,
    a8: f64,
) -> f64 {
    let lo = a1.min(a2).min(a3).min(a4).min(a5).min(a6).min(a7).min(a8);
    let hi = a1.max(a2).max(a3).max(a4).max(a5).max(a6).max(a7).max(a8);
    c.max(lo).min(hi)
}
