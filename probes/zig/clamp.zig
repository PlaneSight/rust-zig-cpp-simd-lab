// Tiny exported functions intended for assembly/codegen inspection.
// Keep these free of allocation, slices, and benchmark harness overhead.
//
// Example:
//   zig build-obj probes/zig/clamp.zig -O ReleaseFast -femit-asm=clamp.s

const builtin = @import("builtin");

fn min8(a1: anytype, a2: anytype, a3: anytype, a4: anytype, a5: anytype, a6: anytype, a7: anytype, a8: anytype) @TypeOf(a1) {
    return @min(@min(@min(a1, a2), @min(a3, a4)), @min(@min(a5, a6), @min(a7, a8)));
}

fn max8(a1: anytype, a2: anytype, a3: anytype, a4: anytype, a5: anytype, a6: anytype, a7: anytype, a8: anytype) @TypeOf(a1) {
    return @max(@max(@max(a1, a2), @max(a3, a4)), @max(@max(a5, a6), @max(a7, a8)));
}

pub export fn clamp_u8x32(
    c: @Vector(32, u8),
    a1: @Vector(32, u8),
    a2: @Vector(32, u8),
    a3: @Vector(32, u8),
    a4: @Vector(32, u8),
    a5: @Vector(32, u8),
    a6: @Vector(32, u8),
    a7: @Vector(32, u8),
    a8: @Vector(32, u8),
) @Vector(32, u8) {
    const lo = min8(a1, a2, a3, a4, a5, a6, a7, a8);
    const hi = max8(a1, a2, a3, a4, a5, a6, a7, a8);
    return @max(lo, @min(c, hi));
}

pub export fn clamp_u16x16(
    c: @Vector(16, u16),
    a1: @Vector(16, u16),
    a2: @Vector(16, u16),
    a3: @Vector(16, u16),
    a4: @Vector(16, u16),
    a5: @Vector(16, u16),
    a6: @Vector(16, u16),
    a7: @Vector(16, u16),
    a8: @Vector(16, u16),
) @Vector(16, u16) {
    const lo = min8(a1, a2, a3, a4, a5, a6, a7, a8);
    const hi = max8(a1, a2, a3, a4, a5, a6, a7, a8);
    return @max(lo, @min(c, hi));
}

pub export fn clamp_f32x8(
    c: @Vector(8, f32),
    a1: @Vector(8, f32),
    a2: @Vector(8, f32),
    a3: @Vector(8, f32),
    a4: @Vector(8, f32),
    a5: @Vector(8, f32),
    a6: @Vector(8, f32),
    a7: @Vector(8, f32),
    a8: @Vector(8, f32),
) @Vector(8, f32) {
    const lo = min8(a1, a2, a3, a4, a5, a6, a7, a8);
    const hi = max8(a1, a2, a3, a4, a5, a6, a7, a8);
    return @max(lo, @min(c, hi));
}

// Reproduction-shaped native f16 path from ziglang/zig#19550.
fn clamp_f16x16_native(
    c: @Vector(16, f16),
    a1: @Vector(16, f16),
    a2: @Vector(16, f16),
    a3: @Vector(16, f16),
    a4: @Vector(16, f16),
    a5: @Vector(16, f16),
    a6: @Vector(16, f16),
    a7: @Vector(16, f16),
    a8: @Vector(16, f16),
) callconv(.c) @Vector(16, f16) {
    const lo = min8(a1, a2, a3, a4, a5, a6, a7, a8);
    const hi = max8(a1, a2, a3, a4, a5, a6, a7, a8);
    return @max(lo, @min(c, hi));
}

// Application-level semantic tradeoff: widen each input once, perform the
// complete clamp in f32, then narrow exactly once at the output boundary.
fn clamp_f16x16_promote_once(
    c: @Vector(16, f16),
    a1: @Vector(16, f16),
    a2: @Vector(16, f16),
    a3: @Vector(16, f16),
    a4: @Vector(16, f16),
    a5: @Vector(16, f16),
    a6: @Vector(16, f16),
    a7: @Vector(16, f16),
    a8: @Vector(16, f16),
) callconv(.c) @Vector(16, f16) {
    const F32x16 = @Vector(16, f32);
    const wc: F32x16 = @floatCast(c);
    const w1: F32x16 = @floatCast(a1);
    const w2: F32x16 = @floatCast(a2);
    const w3: F32x16 = @floatCast(a3);
    const w4: F32x16 = @floatCast(a4);
    const w5: F32x16 = @floatCast(a5);
    const w6: F32x16 = @floatCast(a6);
    const w7: F32x16 = @floatCast(a7);
    const w8: F32x16 = @floatCast(a8);

    const lo = min8(w1, w2, w3, w4, w5, w6, w7, w8);
    const hi = max8(w1, w2, w3, w4, w5, w6, w7, w8);
    const out = @max(lo, @min(wc, hi));
    return @floatCast(out);
}

// WebAssembly SIMD has no native f16 vector ABI. Exporting these functions on
// wasm32 makes Zig 0.16 ask LLVM to perform an invalid backend cast, so retain
// the integer/f32 clamp probes there and expose f16 only on capable targets.
comptime {
    if (builtin.target.cpu.arch != .wasm32) {
        @export(&clamp_f16x16_native, .{ .name = "clamp_f16x16_native" });
        @export(&clamp_f16x16_promote_once, .{ .name = "clamp_f16x16_promote_once" });
    }
}
