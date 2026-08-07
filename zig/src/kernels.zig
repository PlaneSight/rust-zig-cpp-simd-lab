const std = @import("std");

pub fn axpyScalar(dst: []f32, x: []const f32, y: []const f32, a: f32) void {
    std.debug.assert(dst.len == x.len and x.len == y.len);
    for (dst, x, y) |*d, xv, yv| {
        d.* = @mulAdd(f32, a, xv, yv);
    }
}

/// Reference squared error for f32 inputs with f64 accumulation.
/// Subtraction keeps f32 semantics; products and the reduction are widened.
pub fn squaredErrorScalar(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0;
    for (a, b) |x, y| {
        const d: f64 = @floatCast(x - y);
        sum += d * d;
    }
    return sum;
}

pub fn squaredErrorVector(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);

    const Float = @Vector(8, f32);
    const Wide = @Vector(8, f64);
    var acc: Wide = @splat(0.0);
    var i: usize = 0;

    while (i + 8 <= a.len) : (i += 8) {
        const va: Float = a[i..][0..8].*;
        const vb: Float = b[i..][0..8].*;
        const difference: Float = va - vb;
        const wide: Wide = @floatCast(difference);
        acc += wide * wide;
    }

    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        const d: f64 = @floatCast(a[i] - b[i]);
        sum += d * d;
    }
    return sum;
}

pub fn sadU8Scalar(a: []const u8, b: []const u8) u64 {
    std.debug.assert(a.len == b.len);
    var sum: u64 = 0;
    for (a, b) |x, y| {
        sum += if (x > y) x - y else y - x;
    }
    return sum;
}

pub fn sadU8Vector(a: []const u8, b: []const u8) u64 {
    std.debug.assert(a.len == b.len);
    const Bytes = @Vector(32, u8);
    const Wide = @Vector(32, u16);
    var sum: u64 = 0;
    var i: usize = 0;

    while (i + 32 <= a.len) : (i += 32) {
        const va: Bytes = a[i..][0..32].*;
        const vb: Bytes = b[i..][0..32].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        const difference: Wide = @max(wide_a, wide_b) - @min(wide_a, wide_b);
        sum += @as(u64, @intCast(@reduce(.Add, difference)));
    }

    while (i < a.len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        sum += if (x > y) x - y else y - x;
    }
    return sum;
}

/// Adds unsigned bytes element-wise with saturation at 255.
/// `dst`, `a`, and `b` must have equal lengths, and `dst` must not partially
/// overlap either input.
pub fn satAddU8Scalar(dst: []u8, a: []const u8, b: []const u8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

/// Adds unsigned bytes in native 32-lane vectors with a scalar tail.
/// `dst`, `a`, and `b` must have equal lengths, and `dst` must not partially
/// overlap either input.
pub fn satAddU8Vector(dst: []u8, a: []const u8, b: []const u8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Bytes = @Vector(32, u8);
    var i: usize = 0;

    while (i + 32 <= dst.len) : (i += 32) {
        const va: Bytes = a[i..][0..32].*;
        const vb: Bytes = b[i..][0..32].*;
        dst[i..][0..32].* = va +| vb;
    }

    while (i < dst.len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub fn satAddI8Scalar(dst: []i8, a: []const i8, b: []const i8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

pub fn satAddU16Scalar(dst: []u16, a: []const u16, b: []const u16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

pub fn satAddI16Scalar(dst: []i16, a: []const i16, b: []const i16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

pub fn satAddU32Scalar(dst: []u32, a: []const u32, b: []const u32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

pub fn satAddI32Scalar(dst: []i32, a: []const i32, b: []const i32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

pub fn satAddU64Scalar(dst: []u64, a: []const u64, b: []const u64) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}

pub fn satAddI64Scalar(dst: []i64, a: []const i64, b: []const i64) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        out.* = x +| y;
    }
}
pub fn dotF32Scalar(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0;
    for (a, b) |x, y| {
        const wide_x: f64 = @floatCast(x);
        const wide_y: f64 = @floatCast(y);
        sum += wide_x * wide_y;
    }
    return sum;
}

pub fn dotF32Vector(a: []const f32, b: []const f32) f64 {
    std.debug.assert(a.len == b.len);
    const Float = @Vector(8, f32);
    const Wide = @Vector(8, f64);
    var acc: Wide = @splat(0.0);
    var i: usize = 0;

    while (i + 8 <= a.len) : (i += 8) {
        const va: Float = a[i..][0..8].*;
        const vb: Float = b[i..][0..8].*;
        const wide_a: Wide = @floatCast(va);
        const wide_b: Wide = @floatCast(vb);
        acc += wide_a * wide_b;
    }

    var sum: f64 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        const wide_x: f64 = @floatCast(a[i]);
        const wide_y: f64 = @floatCast(b[i]);
        sum += wide_x * wide_y;
    }
    return sum;
}

pub fn dotF64Scalar(a: []const f64, b: []const f64) f64 {
    std.debug.assert(a.len == b.len);
    var sum: f64 = 0;
    for (a, b) |x, y| {
        sum += x * y;
    }
    return sum;
}

pub fn dotF64Vector(a: []const f64, b: []const f64) f64 {
    std.debug.assert(a.len == b.len);
    const Vec = @Vector(4, f64);
    var acc: Vec = @splat(0.0);
    var i: usize = 0;

    while (i + 4 <= a.len) : (i += 4) {
        const va: Vec = a[i..][0..4].*;
        const vb: Vec = b[i..][0..4].*;
        acc += va * vb;
    }

    var sum: f64 = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

pub fn dotI16Scalar(a: []const i16, b: []const i16) i64 {
    std.debug.assert(a.len == b.len);
    var sum: i64 = 0;
    for (a, b) |x, y| {
        const wide_x: i64 = @intCast(x);
        const wide_y: i64 = @intCast(y);
        sum += wide_x * wide_y;
    }
    return sum;
}

pub fn dotI16Vector(a: []const i16, b: []const i16) i64 {
    std.debug.assert(a.len == b.len);
    const Input = @Vector(8, i16);
    const Wide = @Vector(8, i64);
    var sum: i64 = 0;
    var i: usize = 0;

    while (i + 8 <= a.len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        sum += @reduce(.Add, wide_a * wide_b);
    }

    while (i < a.len) : (i += 1) {
        const wide_x: i64 = @intCast(a[i]);
        const wide_y: i64 = @intCast(b[i]);
        sum += wide_x * wide_y;
    }
    return sum;
}

pub fn dotU8I8Scalar(a: []const u8, b: []const i8) i64 {
    std.debug.assert(a.len == b.len);
    var sum: i64 = 0;
    for (a, b) |x, y| {
        const unsigned_x: i64 = @intCast(x);
        const signed_y: i64 = @intCast(y);
        sum += unsigned_x * signed_y;
    }
    return sum;
}

pub fn dotU8I8Vector(a: []const u8, b: []const i8) i64 {
    std.debug.assert(a.len == b.len);
    const InputU8 = @Vector(8, u8);
    const InputI8 = @Vector(8, i8);
    const Wide = @Vector(8, i64);
    var sum: i64 = 0;
    var i: usize = 0;

    while (i + 8 <= a.len) : (i += 8) {
        const va: InputU8 = a[i..][0..8].*;
        const vb: InputI8 = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        sum += @reduce(.Add, wide_a * wide_b);
    }

    while (i < a.len) : (i += 1) {
        const unsigned_x: i64 = @intCast(a[i]);
        const signed_y: i64 = @intCast(b[i]);
        sum += unsigned_x * signed_y;
    }
    return sum;
}

pub fn widenMulU8U16Scalar(dst: []u16, a: []const u8, b: []const u8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        const wide_x: u16 = @intCast(x);
        const wide_y: u16 = @intCast(y);
        out.* = wide_x * wide_y;
    }
}

pub fn widenMulU8U16Vector(dst: []u16, a: []const u8, b: []const u8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Input = @Vector(16, u8);
    const Wide = @Vector(16, u16);
    var i: usize = 0;

    while (i + 16 <= dst.len) : (i += 16) {
        const va: Input = a[i..][0..16].*;
        const vb: Input = b[i..][0..16].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..16].* = wide_a * wide_b;
    }

    while (i < dst.len) : (i += 1) {
        const wide_x: u16 = @intCast(a[i]);
        const wide_y: u16 = @intCast(b[i]);
        dst[i] = wide_x * wide_y;
    }
}

pub fn widenMulI8I16Scalar(dst: []i16, a: []const i8, b: []const i8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        const wide_x: i16 = @intCast(x);
        const wide_y: i16 = @intCast(y);
        out.* = wide_x * wide_y;
    }
}

pub fn widenMulI8I16Vector(dst: []i16, a: []const i8, b: []const i8) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Input = @Vector(16, i8);
    const Wide = @Vector(16, i16);
    var i: usize = 0;

    while (i + 16 <= dst.len) : (i += 16) {
        const va: Input = a[i..][0..16].*;
        const vb: Input = b[i..][0..16].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..16].* = wide_a * wide_b;
    }

    while (i < dst.len) : (i += 1) {
        const wide_x: i16 = @intCast(a[i]);
        const wide_y: i16 = @intCast(b[i]);
        dst[i] = wide_x * wide_y;
    }
}

pub fn widenMulU16U32Scalar(dst: []u32, a: []const u16, b: []const u16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        const wide_x: u32 = @intCast(x);
        const wide_y: u32 = @intCast(y);
        out.* = wide_x * wide_y;
    }
}

pub fn widenMulU16U32Vector(dst: []u32, a: []const u16, b: []const u16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Input = @Vector(8, u16);
    const Wide = @Vector(8, u32);
    var i: usize = 0;

    while (i + 8 <= dst.len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..8].* = wide_a * wide_b;
    }

    while (i < dst.len) : (i += 1) {
        const wide_x: u32 = @intCast(a[i]);
        const wide_y: u32 = @intCast(b[i]);
        dst[i] = wide_x * wide_y;
    }
}

pub fn widenMulI16I32Scalar(dst: []i32, a: []const i16, b: []const i16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        const wide_x: i32 = @intCast(x);
        const wide_y: i32 = @intCast(y);
        out.* = wide_x * wide_y;
    }
}

pub fn widenMulI16I32Vector(dst: []i32, a: []const i16, b: []const i16) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Input = @Vector(8, i16);
    const Wide = @Vector(8, i32);
    var i: usize = 0;

    while (i + 8 <= dst.len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..8].* = wide_a * wide_b;
    }

    while (i < dst.len) : (i += 1) {
        const wide_x: i32 = @intCast(a[i]);
        const wide_y: i32 = @intCast(b[i]);
        dst[i] = wide_x * wide_y;
    }
}

pub fn widenMulU32U64Scalar(dst: []u64, a: []const u32, b: []const u32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        const wide_x: u64 = @intCast(x);
        const wide_y: u64 = @intCast(y);
        out.* = wide_x * wide_y;
    }
}

pub fn widenMulU32U64Vector(dst: []u64, a: []const u32, b: []const u32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Input = @Vector(4, u32);
    const Wide = @Vector(4, u64);
    var i: usize = 0;

    while (i + 4 <= dst.len) : (i += 4) {
        const va: Input = a[i..][0..4].*;
        const vb: Input = b[i..][0..4].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..4].* = wide_a * wide_b;
    }

    while (i < dst.len) : (i += 1) {
        const wide_x: u64 = @intCast(a[i]);
        const wide_y: u64 = @intCast(b[i]);
        dst[i] = wide_x * wide_y;
    }
}

pub fn widenMulI32I64Scalar(dst: []i64, a: []const i32, b: []const i32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    for (dst, a, b) |*out, x, y| {
        const wide_x: i64 = @intCast(x);
        const wide_y: i64 = @intCast(y);
        out.* = wide_x * wide_y;
    }
}

pub fn widenMulI32I64Vector(dst: []i64, a: []const i32, b: []const i32) void {
    std.debug.assert(dst.len == a.len and a.len == b.len);
    const Input = @Vector(4, i32);
    const Wide = @Vector(4, i64);
    var i: usize = 0;

    while (i + 4 <= dst.len) : (i += 4) {
        const va: Input = a[i..][0..4].*;
        const vb: Input = b[i..][0..4].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..4].* = wide_a * wide_b;
    }

    while (i < dst.len) : (i += 1) {
        const wide_x: i64 = @intCast(a[i]);
        const wide_y: i64 = @intCast(b[i]);
        dst[i] = wide_x * wide_y;
    }
}

pub fn widenU8ToU16Scalar(dst: []u16, src: []const u8) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @intCast(value);
    }
}

pub fn widenU8ToU16Vector(dst: []u16, src: []const u8) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(16, u8);
    const Output = @Vector(16, u16);
    var i: usize = 0;
    while (i + 16 <= dst.len) : (i += 16) {
        const values: Input = src[i..][0..16].*;
        const widened: Output = @intCast(values);
        dst[i..][0..16].* = widened;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @intCast(src[i]);
    }
}

pub fn widenU8ToU32Scalar(dst: []u32, src: []const u8) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @intCast(value);
    }
}

pub fn widenU8ToU32Vector(dst: []u32, src: []const u8) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(8, u8);
    const Output = @Vector(8, u32);
    var i: usize = 0;
    while (i + 8 <= dst.len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const widened: Output = @intCast(values);
        dst[i..][0..8].* = widened;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @intCast(src[i]);
    }
}

pub fn widenI8ToI16Scalar(dst: []i16, src: []const i8) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @intCast(value);
    }
}

pub fn widenI8ToI16Vector(dst: []i16, src: []const i8) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(16, i8);
    const Output = @Vector(16, i16);
    var i: usize = 0;
    while (i + 16 <= dst.len) : (i += 16) {
        const values: Input = src[i..][0..16].*;
        const widened: Output = @intCast(values);
        dst[i..][0..16].* = widened;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @intCast(src[i]);
    }
}

pub fn widenU16ToU32Scalar(dst: []u32, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @intCast(value);
    }
}

pub fn widenU16ToU32Vector(dst: []u32, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(8, u16);
    const Output = @Vector(8, u32);
    var i: usize = 0;
    while (i + 8 <= dst.len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const widened: Output = @intCast(values);
        dst[i..][0..8].* = widened;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @intCast(src[i]);
    }
}

pub fn widenI16ToI32Scalar(dst: []i32, src: []const i16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @intCast(value);
    }
}

pub fn widenI16ToI32Vector(dst: []i32, src: []const i16) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(8, i16);
    const Output = @Vector(8, i32);
    var i: usize = 0;
    while (i + 8 <= dst.len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const widened: Output = @intCast(values);
        dst[i..][0..8].* = widened;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @intCast(src[i]);
    }
}

pub fn convertU16ToF32Scalar(dst: []f32, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @floatFromInt(value);
    }
}

pub fn convertU16ToF32Vector(dst: []f32, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(8, u16);
    const Output = @Vector(8, f32);
    var i: usize = 0;
    while (i + 8 <= dst.len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const converted: Output = @floatFromInt(values);
        dst[i..][0..8].* = converted;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @floatFromInt(src[i]);
    }
}

pub fn convertI16ToF32Scalar(dst: []f32, src: []const i16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @floatFromInt(value);
    }
}

pub fn convertI16ToF32Vector(dst: []f32, src: []const i16) void {
    std.debug.assert(dst.len == src.len);
    const Input = @Vector(8, i16);
    const Output = @Vector(8, f32);
    var i: usize = 0;
    while (i + 8 <= dst.len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const converted: Output = @floatFromInt(values);
        dst[i..][0..8].* = converted;
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @floatFromInt(src[i]);
    }
}

pub fn convertU8F32AffineScalar(dst: []f32, src: []const u8, scale: f32, bias: f32) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @as(f32, @floatFromInt(value)) * scale + bias;
    }
}

pub fn convertF32U8TruncScalar(dst: []u8, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        std.debug.assert(!std.math.isNan(value) and value >= 0.0 and value <= 255.0);
        out.* = @intFromFloat(value);
    }
}

pub fn convertF32U8RoundScalar(dst: []u8, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        std.debug.assert(!std.math.isNan(value) and value >= 0.0 and value <= 255.0);
        const rounded: f32 = @floor(value + 0.5);
        out.* = @intFromFloat(rounded);
    }
}

pub fn convertF32U8SatScalar(dst: []u8, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    const max_value: f32 = @floatFromInt(std.math.maxInt(u8));
    for (dst, src) |*out, value| {
        if (std.math.isNan(value) or value <= 0.0) {
            out.* = 0;
        } else if (value >= max_value) {
            out.* = std.math.maxInt(u8);
        } else {
            out.* = @intFromFloat(value);
        }
    }
}

pub fn f32ToU16SatScalar(dst: []u16, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    const max_value: f32 = @floatFromInt(std.math.maxInt(u16));
    for (dst, src) |*out, value| {
        if (std.math.isNan(value) or value <= 0.0) {
            out.* = 0;
        } else if (value >= max_value) {
            out.* = std.math.maxInt(u16);
        } else {
            out.* = @intFromFloat(value);
        }
    }
}

pub fn narrowU16ToU8TruncScalar(dst: []u8, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = @intCast(value & 0xff);
    }
}

pub fn narrowU16ToU8RoundScalar(dst: []u8, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        const rounded: u32 = (@as(u32, value) + 128) / 257;
        out.* = @intCast(rounded);
    }
}

pub fn narrowU16ToU8SatScalar(dst: []u8, src: []const u16) void {
    std.debug.assert(dst.len == src.len);
    for (dst, src) |*out, value| {
        out.* = if (value > std.math.maxInt(u8)) std.math.maxInt(u8) else @intCast(value);
    }
}

pub fn packU8x4ToU32Scalar(dst: []u32, src: []const u8) void {
    std.debug.assert(src.len == dst.len * 4);
    for (dst, 0..) |*out, group| {
        const offset = group * 4;
        const b0: u32 = @intCast(src[offset]);
        const b1: u32 = @intCast(src[offset + 1]);
        const b2: u32 = @intCast(src[offset + 2]);
        const b3: u32 = @intCast(src[offset + 3]);
        out.* = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

pub fn unpackU32ToU8x4Scalar(dst: []u8, src: []const u32) void {
    std.debug.assert(dst.len == src.len * 4);
    for (src, 0..) |value, group| {
        const offset = group * 4;
        dst[offset] = @truncate(value);
        dst[offset + 1] = @truncate(value >> 8);
        dst[offset + 2] = @truncate(value >> 16);
        dst[offset + 3] = @truncate(value >> 24);
    }
}

pub fn clampF16Native(dst: []f16, c: []const f16, lo: []const f16, hi: []const f16) void {
    std.debug.assert(dst.len == c.len and c.len == lo.len and lo.len == hi.len);
    const Vec = @Vector(16, f16);
    var i: usize = 0;
    while (i + 16 <= dst.len) : (i += 16) {
        const vc: Vec = c[i..][0..16].*;
        const vlo: Vec = lo[i..][0..16].*;
        const vhi: Vec = hi[i..][0..16].*;
        dst[i..][0..16].* = @max(vlo, @min(vc, vhi));
    }
    while (i < dst.len) : (i += 1) {
        dst[i] = @max(lo[i], @min(c[i], hi[i]));
    }
}

pub fn clampF16PromoteOnce(dst: []f16, c: []const f16, lo: []const f16, hi: []const f16) void {
    std.debug.assert(dst.len == c.len and c.len == lo.len and lo.len == hi.len);
    const Half = @Vector(16, f16);
    const Float = @Vector(16, f32);
    var i: usize = 0;
    while (i + 16 <= dst.len) : (i += 16) {
        const hc: Half = c[i..][0..16].*;
        const hlo: Half = lo[i..][0..16].*;
        const hhi: Half = hi[i..][0..16].*;
        const vc: Float = @floatCast(hc);
        const vlo: Float = @floatCast(hlo);
        const vhi: Float = @floatCast(hhi);
        const out: Float = @max(vlo, @min(vc, vhi));
        const narrowed: Half = @floatCast(out);
        dst[i..][0..16].* = narrowed;
    }
    while (i < dst.len) : (i += 1) {
        const wc: f32 = @floatCast(c[i]);
        const wlo: f32 = @floatCast(lo[i]);
        const whi: f32 = @floatCast(hi[i]);
        dst[i] = @floatCast(@max(wlo, @min(wc, whi)));
    }
}

const XorShift64 = struct {
    state: u64,

    fn next(self: *XorShift64) u64 {
        var value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        return value;
    }

    fn nextF32(self: *XorShift64) f32 {
        const top: u32 = @intCast(self.next() >> 40);
        const unit = @as(f32, @floatFromInt(top)) / @as(f32, @floatFromInt(@as(u32, 1) << 24));
        return @mulAdd(f32, unit, 8.0, -4.0);
    }
};

test "axpy" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const y = [_]f32{ 5, 6, 7, 8 };
    var dst = [_]f32{0} ** 4;
    axpyScalar(&dst, &x, &y, 2.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7, 10, 13, 16 }, &dst);
}

test "randomized vector paths match scalar references across tails" {
    const max_len = 2049;
    var floats_a: [max_len]f32 = undefined;
    var floats_b: [max_len]f32 = undefined;
    var bytes_a: [max_len]u8 = undefined;
    var bytes_b: [max_len]u8 = undefined;
    var sat_expected: [max_len]u8 = undefined;
    var sat_candidate: [max_len]u8 = undefined;
    var rng = XorShift64{ .state = 0x8f3c_a516_d27b_49e1 };

    for (0..256) |trial| {
        const len: usize = if (trial < 64) trial else @intCast(rng.next() % max_len);
        for (floats_a[0..len], floats_b[0..len], bytes_a[0..len], bytes_b[0..len]) |*fa, *fb, *ba, *bb| {
            fa.* = rng.nextF32();
            fb.* = rng.nextF32();
            ba.* = @truncate(rng.next());
            bb.* = @truncate(rng.next());
        }
        const reference = squaredErrorScalar(floats_a[0..len], floats_b[0..len]);
        const candidate = squaredErrorVector(floats_a[0..len], floats_b[0..len]);
        const relative_error = @abs(reference - candidate) / @max(@abs(reference), 1.0);
        if (!(relative_error <= 1e-12)) return error.SquaredErrorMismatch;
        if (sadU8Scalar(bytes_a[0..len], bytes_b[0..len]) != sadU8Vector(bytes_a[0..len], bytes_b[0..len])) {
            return error.SadMismatch;
        }
        for (sat_expected[0..len], bytes_a[0..len], bytes_b[0..len]) |*out, x, y| {
            const widened_sum = @as(u16, x) + @as(u16, y);
            out.* = @intCast(@min(widened_sum, 255));
        }
        satAddU8Vector(
            sat_candidate[0..len],
            bytes_a[0..len],
            bytes_b[0..len],
        );
        if (!std.mem.eql(u8, sat_expected[0..len], sat_candidate[0..len])) {
            return error.SaturatingAddMismatch;
        }
    }
}

test "saturating add covers every u8 pair" {
    const pair_count = 256 * 256;
    const allocator = std.testing.allocator;
    const a = try allocator.alloc(u8, pair_count);
    defer allocator.free(a);
    const b = try allocator.alloc(u8, pair_count);
    defer allocator.free(b);
    const expected = try allocator.alloc(u8, pair_count);
    defer allocator.free(expected);
    const candidate = try allocator.alloc(u8, pair_count);
    defer allocator.free(candidate);

    for (0..256) |x| {
        for (0..256) |y| {
            const index = x * 256 + y;
            a[index] = @intCast(x);
            b[index] = @intCast(y);
            expected[index] = @intCast(@min(x + y, 255));
        }
    }

    satAddU8Scalar(candidate, a, b);
    try std.testing.expectEqualSlices(u8, expected, candidate);
    @memset(candidate, 0);
    satAddU8Vector(candidate, a, b);
    try std.testing.expectEqualSlices(u8, expected, candidate);
}

test "randomized f16 clamp strategies match across vector tails" {
    const half_values = [_]f16{ 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0 };
    var c: [257]f16 = undefined;
    var lo: [257]f16 = undefined;
    var hi: [257]f16 = undefined;
    var native: [257]f16 = undefined;
    var promoted: [257]f16 = undefined;
    var rng = XorShift64{ .state = 0xc672_b9e4_215d_8a3f };

    for (0..192) |trial| {
        const len: usize = if (trial < 64) trial else @intCast(rng.next() % 258);
        for (c[0..len], lo[0..len], hi[0..len]) |*value, *low, *high| {
            const index: usize = @intCast(rng.next() & 7);
            value.* = half_values[index];
            low.* = 0.5;
            high.* = 2.0;
        }
        clampF16Native(native[0..len], c[0..len], lo[0..len], hi[0..len]);
        clampF16PromoteOnce(promoted[0..len], c[0..len], lo[0..len], hi[0..len]);
        try std.testing.expectEqualSlices(f16, native[0..len], promoted[0..len]);
    }
}

test "dot products and widening multiplies match independent references" {
    const max_len = 2049;
    var f32_a: [max_len]f32 = undefined;
    var f32_b: [max_len]f32 = undefined;
    var f64_a: [max_len]f64 = undefined;
    var f64_b: [max_len]f64 = undefined;
    var i16_a: [max_len]i16 = undefined;
    var i16_b: [max_len]i16 = undefined;
    var u8_a: [max_len]u8 = undefined;
    var u8_b: [max_len]u8 = undefined;
    var i8_a: [max_len]i8 = undefined;
    var i8_b: [max_len]i8 = undefined;
    var u16_a: [max_len]u16 = undefined;
    var u16_b: [max_len]u16 = undefined;
    var u32_a: [max_len]u32 = undefined;
    var u32_b: [max_len]u32 = undefined;
    var i32_a: [max_len]i32 = undefined;
    var i32_b: [max_len]i32 = undefined;

    var expected_u16: [max_len]u16 = undefined;
    var scalar_u16: [max_len]u16 = undefined;
    var vector_u16: [max_len]u16 = undefined;
    var expected_i16: [max_len]i16 = undefined;
    var scalar_i16: [max_len]i16 = undefined;
    var vector_i16: [max_len]i16 = undefined;
    var expected_u32: [max_len]u32 = undefined;
    var scalar_u32: [max_len]u32 = undefined;
    var vector_u32: [max_len]u32 = undefined;
    var expected_i32: [max_len]i32 = undefined;
    var scalar_i32: [max_len]i32 = undefined;
    var vector_i32: [max_len]i32 = undefined;
    var expected_u64: [max_len]u64 = undefined;
    var scalar_u64: [max_len]u64 = undefined;
    var vector_u64: [max_len]u64 = undefined;
    var expected_i64: [max_len]i64 = undefined;
    var scalar_i64: [max_len]i64 = undefined;
    var vector_i64: [max_len]i64 = undefined;

    const deterministic_lengths = [_]usize{ 0, 1, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65, 127 };
    var lengths: [deterministic_lengths.len + 256]usize = undefined;
    for (deterministic_lengths, 0..) |length, index| {
        lengths[index] = length;
    }
    var rng = XorShift64{ .state = 0x1b6e_4a92_7d03_c5f1 };
    for (lengths[deterministic_lengths.len..], 0..) |*length, trial| {
        length.* = if (trial < 64) trial else @intCast(rng.next() % (max_len + 1));
    }

    const i16_extrema = [_]i16{ -32768, -32767, -1, 0, 1, 32767 };
    const i8_extrema = [_]i8{ -128, -127, -1, 0, 1, 127 };
    const u8_extrema = [_]u8{ 0, 1, 127, 128, 254, 255 };
    const u16_extrema = [_]u16{ 0, 1, 255, 256, 65534, 65535 };
    const i32_extrema = [_]i32{ -2147483648, -2147483647, -1, 0, 1, 2147483647 };
    const u32_extrema = [_]u32{ 0, 1, 255, 256, 4294967294, 4294967295 };

    for (lengths, 0..) |len, trial| {
        for (0..len) |i| {
            const lane = (i + trial) % 11;
            const extrema_lane = lane % 6;
            if (lane < 6) {
                f32_a[i] = switch (lane) {
                    0 => -3.5,
                    1 => -1.0,
                    2 => -0.0,
                    3 => 0.0,
                    4 => 1.0,
                    5 => 3.5,
                    else => unreachable,
                };
                f32_b[i] = switch (lane) {
                    0 => 3.5,
                    1 => 1.0,
                    2 => -0.0,
                    3 => 0.0,
                    4 => -1.0,
                    5 => -3.5,
                    else => unreachable,
                };
            } else {
                f32_a[i] = rng.nextF32();
                f32_b[i] = rng.nextF32();
            }

            f64_a[i] = @as(f64, @floatCast(f32_a[i])) * 0.125;
            f64_b[i] = @as(f64, @floatCast(f32_b[i])) * -0.25;

            if (lane < 6) {
                i16_a[i] = i16_extrema[extrema_lane];
                i16_b[i] = i16_extrema[(extrema_lane * 5 + 1) % i16_extrema.len];
                i8_a[i] = i8_extrema[extrema_lane];
                i8_b[i] = i8_extrema[(extrema_lane * 5 + 1) % i8_extrema.len];
                u8_a[i] = u8_extrema[extrema_lane];
                u8_b[i] = u8_extrema[(extrema_lane * 5 + 1) % u8_extrema.len];
                u16_a[i] = u16_extrema[extrema_lane];
                u16_b[i] = u16_extrema[(extrema_lane * 5 + 1) % u16_extrema.len];
                i32_a[i] = i32_extrema[extrema_lane];
                i32_b[i] = i32_extrema[(extrema_lane * 5 + 1) % i32_extrema.len];
                u32_a[i] = u32_extrema[extrema_lane];
                u32_b[i] = u32_extrema[(extrema_lane * 5 + 1) % u32_extrema.len];
            } else {
                i16_a[i] = @bitCast(@as(u16, @truncate(rng.next())));
                i16_b[i] = @bitCast(@as(u16, @truncate(rng.next())));
                i8_a[i] = @bitCast(@as(u8, @truncate(rng.next())));
                i8_b[i] = @bitCast(@as(u8, @truncate(rng.next())));
                u8_a[i] = @truncate(rng.next());
                u8_b[i] = @truncate(rng.next());
                u16_a[i] = @truncate(rng.next());
                u16_b[i] = @truncate(rng.next());
                i32_a[i] = @bitCast(@as(u32, @truncate(rng.next())));
                i32_b[i] = @bitCast(@as(u32, @truncate(rng.next())));
                u32_a[i] = @truncate(rng.next());
                u32_b[i] = @truncate(rng.next());
            }
        }
        var expected_dot_f32: f64 = 0;
        var expected_dot_f64: f64 = 0;
        var expected_dot_i16: i64 = 0;
        var expected_dot_u8_i8: i64 = 0;
        for (f32_a[0..len], f32_b[0..len], f64_a[0..len], f64_b[0..len], i16_a[0..len], i16_b[0..len], u8_a[0..len], i8_b[0..len]) |fa, fb, da, db, ia, ib, ua, sb| {
            const wide_fa: f64 = @floatCast(fa);
            const wide_fb: f64 = @floatCast(fb);
            expected_dot_f32 += wide_fa * wide_fb;
            expected_dot_f64 += da * db;
            const wide_ia: i64 = @intCast(ia);
            const wide_ib: i64 = @intCast(ib);
            expected_dot_i16 += wide_ia * wide_ib;
            const wide_ua: i64 = @intCast(ua);
            const wide_sb: i64 = @intCast(sb);
            expected_dot_u8_i8 += wide_ua * wide_sb;
        }

        const scalar_dot_f32 = dotF32Scalar(f32_a[0..len], f32_b[0..len]);
        const vector_dot_f32 = dotF32Vector(f32_a[0..len], f32_b[0..len]);
        const f32_scalar_error = @abs(scalar_dot_f32 - expected_dot_f32) /
            @max(@abs(expected_dot_f32), 1.0);
        const f32_vector_error = @abs(vector_dot_f32 - expected_dot_f32) /
            @max(@abs(expected_dot_f32), 1.0);
        try std.testing.expect(f32_scalar_error <= 1e-12);
        try std.testing.expect(f32_vector_error <= 1e-12);

        const scalar_dot_f64 = dotF64Scalar(f64_a[0..len], f64_b[0..len]);
        const vector_dot_f64 = dotF64Vector(f64_a[0..len], f64_b[0..len]);
        const f64_vector_error = @abs(vector_dot_f64 - scalar_dot_f64) /
            @max(@abs(scalar_dot_f64), 1.0);
        try std.testing.expect(f64_vector_error <= 1e-12);

        try std.testing.expectEqual(expected_dot_i16, dotI16Scalar(i16_a[0..len], i16_b[0..len]));
        try std.testing.expectEqual(expected_dot_i16, dotI16Vector(i16_a[0..len], i16_b[0..len]));
        try std.testing.expectEqual(expected_dot_u8_i8, dotU8I8Scalar(u8_a[0..len], i8_b[0..len]));
        try std.testing.expectEqual(expected_dot_u8_i8, dotU8I8Vector(u8_a[0..len], i8_b[0..len]));

        for (expected_u16[0..len], u8_a[0..len], u8_b[0..len]) |*out, x, y| {
            const wide_x: u16 = @intCast(x);
            const wide_y: u16 = @intCast(y);
            out.* = wide_x * wide_y;
        }
        widenMulU8U16Scalar(scalar_u16[0..len], u8_a[0..len], u8_b[0..len]);
        widenMulU8U16Vector(vector_u16[0..len], u8_a[0..len], u8_b[0..len]);
        try std.testing.expectEqualSlices(u16, expected_u16[0..len], scalar_u16[0..len]);
        try std.testing.expectEqualSlices(u16, expected_u16[0..len], vector_u16[0..len]);

        for (expected_i16[0..len], i8_a[0..len], i8_b[0..len]) |*out, x, y| {
            const wide_x: i16 = @intCast(x);
            const wide_y: i16 = @intCast(y);
            out.* = wide_x * wide_y;
        }
        widenMulI8I16Scalar(scalar_i16[0..len], i8_a[0..len], i8_b[0..len]);
        widenMulI8I16Vector(vector_i16[0..len], i8_a[0..len], i8_b[0..len]);
        try std.testing.expectEqualSlices(i16, expected_i16[0..len], scalar_i16[0..len]);
        try std.testing.expectEqualSlices(i16, expected_i16[0..len], vector_i16[0..len]);

        for (expected_u32[0..len], u16_a[0..len], u16_b[0..len]) |*out, x, y| {
            const wide_x: u32 = @intCast(x);
            const wide_y: u32 = @intCast(y);
            out.* = wide_x * wide_y;
        }
        widenMulU16U32Scalar(scalar_u32[0..len], u16_a[0..len], u16_b[0..len]);
        widenMulU16U32Vector(vector_u32[0..len], u16_a[0..len], u16_b[0..len]);
        try std.testing.expectEqualSlices(u32, expected_u32[0..len], scalar_u32[0..len]);
        try std.testing.expectEqualSlices(u32, expected_u32[0..len], vector_u32[0..len]);

        for (expected_i32[0..len], i16_a[0..len], i16_b[0..len]) |*out, x, y| {
            const wide_x: i32 = @intCast(x);
            const wide_y: i32 = @intCast(y);
            out.* = wide_x * wide_y;
        }
        widenMulI16I32Scalar(scalar_i32[0..len], i16_a[0..len], i16_b[0..len]);
        widenMulI16I32Vector(vector_i32[0..len], i16_a[0..len], i16_b[0..len]);
        try std.testing.expectEqualSlices(i32, expected_i32[0..len], scalar_i32[0..len]);
        try std.testing.expectEqualSlices(i32, expected_i32[0..len], vector_i32[0..len]);

        for (expected_u64[0..len], u32_a[0..len], u32_b[0..len]) |*out, x, y| {
            const wide_x: u64 = @intCast(x);
            const wide_y: u64 = @intCast(y);
            out.* = wide_x * wide_y;
        }
        widenMulU32U64Scalar(scalar_u64[0..len], u32_a[0..len], u32_b[0..len]);
        widenMulU32U64Vector(vector_u64[0..len], u32_a[0..len], u32_b[0..len]);
        try std.testing.expectEqualSlices(u64, expected_u64[0..len], scalar_u64[0..len]);
        try std.testing.expectEqualSlices(u64, expected_u64[0..len], vector_u64[0..len]);

        for (expected_i64[0..len], i32_a[0..len], i32_b[0..len]) |*out, x, y| {
            const wide_x: i64 = @intCast(x);
            const wide_y: i64 = @intCast(y);
            out.* = wide_x * wide_y;
        }
        widenMulI32I64Scalar(scalar_i64[0..len], i32_a[0..len], i32_b[0..len]);
        widenMulI32I64Vector(vector_i64[0..len], i32_a[0..len], i32_b[0..len]);
        try std.testing.expectEqualSlices(i64, expected_i64[0..len], scalar_i64[0..len]);
        try std.testing.expectEqualSlices(i64, expected_i64[0..len], vector_i64[0..len]);
    }
}

test "saturating add scalar paths match widened checked references" {
    const max_len = 2049;
    var u8_a: [max_len]u8 = undefined;
    var u8_b: [max_len]u8 = undefined;
    var u8_expected: [max_len]u8 = undefined;
    var u8_candidate: [max_len]u8 = undefined;
    var i8_a: [max_len]i8 = undefined;
    var i8_b: [max_len]i8 = undefined;
    var i8_expected: [max_len]i8 = undefined;
    var i8_candidate: [max_len]i8 = undefined;
    var u16_a: [max_len]u16 = undefined;
    var u16_b: [max_len]u16 = undefined;
    var u16_expected: [max_len]u16 = undefined;
    var u16_candidate: [max_len]u16 = undefined;
    var i16_a: [max_len]i16 = undefined;
    var i16_b: [max_len]i16 = undefined;
    var i16_expected: [max_len]i16 = undefined;
    var i16_candidate: [max_len]i16 = undefined;
    var u32_a: [max_len]u32 = undefined;
    var u32_b: [max_len]u32 = undefined;
    var u32_expected: [max_len]u32 = undefined;
    var u32_candidate: [max_len]u32 = undefined;
    var i32_a: [max_len]i32 = undefined;
    var i32_b: [max_len]i32 = undefined;
    var i32_expected: [max_len]i32 = undefined;
    var i32_candidate: [max_len]i32 = undefined;
    var u64_a: [max_len]u64 = undefined;
    var u64_b: [max_len]u64 = undefined;
    var u64_expected: [max_len]u64 = undefined;
    var u64_candidate: [max_len]u64 = undefined;
    var i64_a: [max_len]i64 = undefined;
    var i64_b: [max_len]i64 = undefined;
    var i64_expected: [max_len]i64 = undefined;
    var i64_candidate: [max_len]i64 = undefined;

    const deterministic_lengths = [_]usize{
        0,    1,  2,  3,   7,   8,   9,   15,  16,  17,  31,   32,   33,
        63,   64, 65, 127, 128, 129, 255, 256, 257, 511, 1023, 1024, 2048,
        2049,
    };
    var lengths: [deterministic_lengths.len + 256]usize = undefined;
    for (deterministic_lengths, 0..) |length, index| {
        lengths[index] = length;
    }
    var rng = XorShift64{ .state = 0x4a1d_83c7_6e20_b5f9 };
    for (lengths[deterministic_lengths.len..], 0..) |*length, trial| {
        length.* = if (trial < 64) trial else @intCast(rng.next() % (max_len + 1));
    }

    const u8_extrema = [_]u8{ 0, 1, 127, 128, 254, 255, 255 };
    const i8_extrema = [_]i8{ -128, -127, -1, 0, 1, 126, 127 };
    const u16_extrema = [_]u16{ 0, 1, 32767, 32768, 65534, 65535, 65535 };
    const i16_extrema = [_]i16{ -32768, -32767, -1, 0, 1, 32766, 32767 };
    const u32_extrema = [_]u32{ 0, 1, 2147483647, 2147483648, 4294967294, 4294967295, 4294967295 };
    const i32_extrema = [_]i32{ -2147483648, -2147483647, -1, 0, 1, 2147483646, 2147483647 };
    const u64_extrema = [_]u64{
        0,
        1,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        std.math.maxInt(u64) - 1,
        std.math.maxInt(u64),
        std.math.maxInt(u64),
    };
    const i64_extrema = [_]i64{
        std.math.minInt(i64),
        std.math.minInt(i64) + 1,
        -1,
        0,
        1,
        std.math.maxInt(i64) - 1,
        std.math.maxInt(i64),
    };

    const pair_indices = [_]usize{ 1, 0, 6, 2, 3, 5, 6 };

    for (lengths, 0..) |len, trial| {
        for (0..len) |i| {
            const lane = (i + trial) % 11;
            if (lane < 7) {
                const pair = pair_indices[lane];
                u8_a[i] = u8_extrema[lane];
                u8_b[i] = u8_extrema[pair];
                i8_a[i] = i8_extrema[lane];
                i8_b[i] = i8_extrema[pair];
                u16_a[i] = u16_extrema[lane];
                u16_b[i] = u16_extrema[pair];
                i16_a[i] = i16_extrema[lane];
                i16_b[i] = i16_extrema[pair];
                u32_a[i] = u32_extrema[lane];
                u32_b[i] = u32_extrema[pair];
                i32_a[i] = i32_extrema[lane];
                i32_b[i] = i32_extrema[pair];
                u64_a[i] = u64_extrema[lane];
                u64_b[i] = u64_extrema[pair];
                i64_a[i] = i64_extrema[lane];
                i64_b[i] = i64_extrema[pair];
            } else {
                u8_a[i] = @truncate(rng.next());
                u8_b[i] = @truncate(rng.next());
                i8_a[i] = @bitCast(@as(u8, @truncate(rng.next())));
                i8_b[i] = @bitCast(@as(u8, @truncate(rng.next())));
                u16_a[i] = @truncate(rng.next());
                u16_b[i] = @truncate(rng.next());
                i16_a[i] = @bitCast(@as(u16, @truncate(rng.next())));
                i16_b[i] = @bitCast(@as(u16, @truncate(rng.next())));
                u32_a[i] = @truncate(rng.next());
                u32_b[i] = @truncate(rng.next());
                i32_a[i] = @bitCast(@as(u32, @truncate(rng.next())));
                i32_b[i] = @bitCast(@as(u32, @truncate(rng.next())));
                u64_a[i] = rng.next();
                u64_b[i] = rng.next();
                i64_a[i] = @bitCast(rng.next());
                i64_b[i] = @bitCast(rng.next());
            }
        }

        for (u8_expected[0..len], u8_a[0..len], u8_b[0..len]) |*out, x, y| {
            const sum: u16 = @as(u16, x) + @as(u16, y);
            out.* = if (sum > @as(u16, std.math.maxInt(u8))) std.math.maxInt(u8) else @intCast(sum);
        }
        for (i8_expected[0..len], i8_a[0..len], i8_b[0..len]) |*out, x, y| {
            const sum: i16 = @as(i16, x) + @as(i16, y);
            const min_value: i16 = @intCast(std.math.minInt(i8));
            const max_value: i16 = @intCast(std.math.maxInt(i8));
            out.* = if (sum < min_value) std.math.minInt(i8) else if (sum > max_value) std.math.maxInt(i8) else @intCast(sum);
        }
        for (u16_expected[0..len], u16_a[0..len], u16_b[0..len]) |*out, x, y| {
            const sum: u32 = @as(u32, x) + @as(u32, y);
            out.* = if (sum > @as(u32, std.math.maxInt(u16))) std.math.maxInt(u16) else @intCast(sum);
        }
        for (i16_expected[0..len], i16_a[0..len], i16_b[0..len]) |*out, x, y| {
            const sum: i32 = @as(i32, x) + @as(i32, y);
            const min_value: i32 = @intCast(std.math.minInt(i16));
            const max_value: i32 = @intCast(std.math.maxInt(i16));
            out.* = if (sum < min_value) std.math.minInt(i16) else if (sum > max_value) std.math.maxInt(i16) else @intCast(sum);
        }
        for (u32_expected[0..len], u32_a[0..len], u32_b[0..len]) |*out, x, y| {
            const sum: u64 = @as(u64, x) + @as(u64, y);
            out.* = if (sum > @as(u64, std.math.maxInt(u32))) std.math.maxInt(u32) else @intCast(sum);
        }
        for (i32_expected[0..len], i32_a[0..len], i32_b[0..len]) |*out, x, y| {
            const sum: i64 = @as(i64, x) + @as(i64, y);
            const min_value: i64 = @intCast(std.math.minInt(i32));
            const max_value: i64 = @intCast(std.math.maxInt(i32));
            out.* = if (sum < min_value) std.math.minInt(i32) else if (sum > max_value) std.math.maxInt(i32) else @intCast(sum);
        }
        for (u64_expected[0..len], u64_a[0..len], u64_b[0..len]) |*out, x, y| {
            const sum: u128 = @as(u128, x) + @as(u128, y);
            out.* = if (sum > @as(u128, std.math.maxInt(u64))) std.math.maxInt(u64) else @intCast(sum);
        }
        for (i64_expected[0..len], i64_a[0..len], i64_b[0..len]) |*out, x, y| {
            const sum: i128 = @as(i128, x) + @as(i128, y);
            const min_value: i128 = @intCast(std.math.minInt(i64));
            const max_value: i128 = @intCast(std.math.maxInt(i64));
            out.* = if (sum < min_value) std.math.minInt(i64) else if (sum > max_value) std.math.maxInt(i64) else @intCast(sum);
        }

        satAddU8Scalar(u8_candidate[0..len], u8_a[0..len], u8_b[0..len]);
        satAddI8Scalar(i8_candidate[0..len], i8_a[0..len], i8_b[0..len]);
        satAddU16Scalar(u16_candidate[0..len], u16_a[0..len], u16_b[0..len]);
        satAddI16Scalar(i16_candidate[0..len], i16_a[0..len], i16_b[0..len]);
        satAddU32Scalar(u32_candidate[0..len], u32_a[0..len], u32_b[0..len]);
        satAddI32Scalar(i32_candidate[0..len], i32_a[0..len], i32_b[0..len]);
        satAddU64Scalar(u64_candidate[0..len], u64_a[0..len], u64_b[0..len]);
        satAddI64Scalar(i64_candidate[0..len], i64_a[0..len], i64_b[0..len]);

        try std.testing.expectEqualSlices(u8, u8_expected[0..len], u8_candidate[0..len]);
        try std.testing.expectEqualSlices(i8, i8_expected[0..len], i8_candidate[0..len]);
        try std.testing.expectEqualSlices(u16, u16_expected[0..len], u16_candidate[0..len]);
        try std.testing.expectEqualSlices(i16, i16_expected[0..len], i16_candidate[0..len]);
        try std.testing.expectEqualSlices(u32, u32_expected[0..len], u32_candidate[0..len]);
        try std.testing.expectEqualSlices(i32, i32_expected[0..len], i32_candidate[0..len]);
        try std.testing.expectEqualSlices(u64, u64_expected[0..len], u64_candidate[0..len]);
        try std.testing.expectEqualSlices(i64, i64_expected[0..len], i64_candidate[0..len]);
    }
}
test "mixed-width widening and integer-to-f32 paths cover vector tails" {
    const max_len = 33;
    var u8_values: [max_len]u8 = undefined;
    var i8_values: [max_len]i8 = undefined;
    var u16_values: [max_len]u16 = undefined;
    var i16_values: [max_len]i16 = undefined;
    var u8_u16_scalar: [max_len]u16 = undefined;
    var u8_u16_vector: [max_len]u16 = undefined;
    var u8_u32_scalar: [max_len]u32 = undefined;
    var u8_u32_vector: [max_len]u32 = undefined;
    var i8_i16_scalar: [max_len]i16 = undefined;
    var i8_i16_vector: [max_len]i16 = undefined;
    var u16_u32_scalar: [max_len]u32 = undefined;
    var u16_u32_vector: [max_len]u32 = undefined;
    var i16_i32_scalar: [max_len]i32 = undefined;
    var i16_i32_vector: [max_len]i32 = undefined;
    var u16_f32_scalar: [max_len]f32 = undefined;
    var u16_f32_vector: [max_len]f32 = undefined;
    var i16_f32_scalar: [max_len]f32 = undefined;
    var i16_f32_vector: [max_len]f32 = undefined;
    var affine: [max_len]f32 = undefined;
    const lengths = [_]usize{ 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33 };
    const u8_edges = [_]u8{ 0, 1, 127, 128, 254, 255 };
    const i8_edges = [_]i8{ std.math.minInt(i8), -127, -1, 0, 1, std.math.maxInt(i8) };
    const u16_edges = [_]u16{ 0, 1, 255, 256, 65534, std.math.maxInt(u16) };
    const i16_edges = [_]i16{ std.math.minInt(i16), -32767, -1, 0, 1, std.math.maxInt(i16) };
    var rng = XorShift64{ .state = 0x7a4c_91e2_d635_f0b1 };

    for (lengths, 0..) |len, trial| {
        for (0..len) |i| {
            const lane = (i + trial) % 11;
            if (lane < 6) {
                u8_values[i] = u8_edges[(i + trial) % u8_edges.len];
                i8_values[i] = i8_edges[(i + trial) % i8_edges.len];
                u16_values[i] = u16_edges[(i + trial) % u16_edges.len];
                i16_values[i] = i16_edges[(i + trial) % i16_edges.len];
            } else {
                u8_values[i] = @truncate(rng.next());
                i8_values[i] = @bitCast(@as(u8, @truncate(rng.next())));
                u16_values[i] = @truncate(rng.next());
                i16_values[i] = @bitCast(@as(u16, @truncate(rng.next())));
            }
        }

        widenU8ToU16Scalar(u8_u16_scalar[0..len], u8_values[0..len]);
        widenU8ToU16Vector(u8_u16_vector[0..len], u8_values[0..len]);
        widenU8ToU32Scalar(u8_u32_scalar[0..len], u8_values[0..len]);
        widenU8ToU32Vector(u8_u32_vector[0..len], u8_values[0..len]);
        widenI8ToI16Scalar(i8_i16_scalar[0..len], i8_values[0..len]);
        widenI8ToI16Vector(i8_i16_vector[0..len], i8_values[0..len]);
        widenU16ToU32Scalar(u16_u32_scalar[0..len], u16_values[0..len]);
        widenU16ToU32Vector(u16_u32_vector[0..len], u16_values[0..len]);
        widenI16ToI32Scalar(i16_i32_scalar[0..len], i16_values[0..len]);
        widenI16ToI32Vector(i16_i32_vector[0..len], i16_values[0..len]);
        convertU16ToF32Scalar(u16_f32_scalar[0..len], u16_values[0..len]);
        convertU16ToF32Vector(u16_f32_vector[0..len], u16_values[0..len]);
        convertI16ToF32Scalar(i16_f32_scalar[0..len], i16_values[0..len]);
        convertI16ToF32Vector(i16_f32_vector[0..len], i16_values[0..len]);
        convertU8F32AffineScalar(affine[0..len], u8_values[0..len], 1.25, -2.5);

        try std.testing.expectEqualSlices(u16, u8_u16_scalar[0..len], u8_u16_vector[0..len]);
        try std.testing.expectEqualSlices(u32, u8_u32_scalar[0..len], u8_u32_vector[0..len]);
        try std.testing.expectEqualSlices(i16, i8_i16_scalar[0..len], i8_i16_vector[0..len]);
        try std.testing.expectEqualSlices(u32, u16_u32_scalar[0..len], u16_u32_vector[0..len]);
        try std.testing.expectEqualSlices(i32, i16_i32_scalar[0..len], i16_i32_vector[0..len]);
        try std.testing.expectEqualSlices(f32, u16_f32_scalar[0..len], u16_f32_vector[0..len]);
        try std.testing.expectEqualSlices(f32, i16_f32_scalar[0..len], i16_f32_vector[0..len]);
        for (0..len) |i| {
            try std.testing.expectEqual(@as(u16, u8_values[i]), u8_u16_scalar[i]);
            try std.testing.expectEqual(@as(u32, u8_values[i]), u8_u32_scalar[i]);
            try std.testing.expectEqual(@as(i16, i8_values[i]), i8_i16_scalar[i]);
            try std.testing.expectEqual(@as(u32, u16_values[i]), u16_u32_scalar[i]);
            try std.testing.expectEqual(@as(i32, i16_values[i]), i16_i32_scalar[i]);
            try std.testing.expectEqual(@as(f32, @floatFromInt(u16_values[i])), u16_f32_scalar[i]);
            try std.testing.expectEqual(@as(f32, @floatFromInt(i16_values[i])), i16_f32_scalar[i]);
            const expected_affine = @as(f32, @floatFromInt(u8_values[i])) * 1.25 - 2.5;
            try std.testing.expectEqual(expected_affine, affine[i]);
        }
    }
}

test "mixed-width float conversions cover fractions, NaNs, infinities, and extrema" {
    const max_len = 33;
    var values: [max_len]f32 = undefined;
    var trunc: [max_len]u8 = undefined;
    var rounded: [max_len]u8 = undefined;
    var saturated: [max_len]u8 = undefined;
    const lengths = [_]usize{ 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33 };
    const valid_edges = [_]f32{ 0.0, 0.49, 0.5, 1.49, 1.5, 127.49, 127.5, 254.5, 255.0 };
    var rng = XorShift64{ .state = 0x11f0_5cc3_a927_6d84 };

    for (lengths, 0..) |len, trial| {
        for (0..len) |i| {
            if ((i + trial) % 10 < valid_edges.len) {
                values[i] = valid_edges[(i + trial) % valid_edges.len];
            } else {
                const integer_part: u32 = @intCast(rng.next() % 255);
                values[i] = @as(f32, @floatFromInt(integer_part)) + 0.25;
            }
        }
        convertF32U8TruncScalar(trunc[0..len], values[0..len]);
        convertF32U8RoundScalar(rounded[0..len], values[0..len]);
        convertF32U8SatScalar(saturated[0..len], values[0..len]);
        for (0..len) |i| {
            try std.testing.expectEqual(@as(u8, @intFromFloat(values[i])), trunc[i]);
            try std.testing.expectEqual(@as(u8, @intFromFloat(@floor(values[i] + 0.5))), rounded[i]);
            try std.testing.expectEqual(trunc[i], saturated[i]);
        }
    }

    const nan: f32 = @bitCast(@as(u32, 0x7fc0_0000));
    const positive_infinity: f32 = @bitCast(@as(u32, 0x7f80_0000));
    const negative_infinity: f32 = @bitCast(@as(u32, 0xff80_0000));
    const pathological_u8 = [_]f32{
        nan, negative_infinity, -1.0, -0.0, 0.0, 0.5, 1.5, 254.9, 255.0, 255.1, positive_infinity,
    };
    const expected_u8 = [_]u8{ 0, 0, 0, 0, 0, 0, 1, 254, 255, 255, 255 };
    convertF32U8SatScalar(saturated[0..pathological_u8.len], pathological_u8[0..]);
    try std.testing.expectEqualSlices(u8, expected_u8[0..], saturated[0..pathological_u8.len]);

    const pathological_u16 = [_]f32{
        nan, negative_infinity, -1.0, -0.0, 0.0, 0.5, 1.5, 65534.5, 65535.0, 65535.5, positive_infinity,
    };
    const expected_u16 = [_]u16{ 0, 0, 0, 0, 0, 0, 1, 65534, 65535, 65535, 65535 };
    var u16_output: [pathological_u16.len]u16 = undefined;
    f32ToU16SatScalar(&u16_output, pathological_u16[0..]);
    try std.testing.expectEqualSlices(u16, expected_u16[0..], u16_output[0..]);

    const narrow_input = [_]u16{ 0, 1, 127, 128, 255, 256, 257, 511, 512, 1023, 65534, 65535 };
    var narrow_trunc: [narrow_input.len]u8 = undefined;
    var narrow_round: [narrow_input.len]u8 = undefined;
    var narrow_sat: [narrow_input.len]u8 = undefined;
    narrowU16ToU8TruncScalar(&narrow_trunc, narrow_input[0..]);
    narrowU16ToU8RoundScalar(&narrow_round, narrow_input[0..]);
    narrowU16ToU8SatScalar(&narrow_sat, narrow_input[0..]);
    for (narrow_input, narrow_trunc, narrow_round, narrow_sat) |value, trunc_value, round_value, sat_value| {
        try std.testing.expectEqual(@as(u8, @intCast(value & 0xff)), trunc_value);
        try std.testing.expectEqual(@as(u8, @intCast((@as(u32, value) + 128) / 257)), round_value);
        try std.testing.expectEqual(if (value > 255) @as(u8, 255) else @as(u8, @intCast(value)), sat_value);
    }
}

test "mixed-width packing and unpacking use logical little-endian groups" {
    const max_groups = 9;
    var bytes: [max_groups * 4]u8 = undefined;
    var unpacked: [max_groups * 4]u8 = undefined;
    var words: [max_groups]u32 = undefined;
    const group_counts = [_]usize{ 0, 1, 2, 5, 9 };
    const edge_bytes = [_]u8{ 0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff, 0x12, 0x34 };
    var rng = XorShift64{ .state = 0x92b3_4de1_0f67_a8c5 };

    for (group_counts, 0..) |groups, trial| {
        for (0..groups * 4) |i| {
            bytes[i] = if ((i + trial) % 3 == 0) edge_bytes[(i + trial) % edge_bytes.len] else @truncate(rng.next());
        }
        packU8x4ToU32Scalar(words[0..groups], bytes[0 .. groups * 4]);
        for (0..groups) |group| {
            const offset = group * 4;
            const expected: u32 = @as(u32, bytes[offset]) |
                (@as(u32, bytes[offset + 1]) << 8) |
                (@as(u32, bytes[offset + 2]) << 16) |
                (@as(u32, bytes[offset + 3]) << 24);
            try std.testing.expectEqual(expected, words[group]);
        }
        unpackU32ToU8x4Scalar(unpacked[0 .. groups * 4], words[0..groups]);
        try std.testing.expectEqualSlices(u8, bytes[0 .. groups * 4], unpacked[0 .. groups * 4]);
    }
}
