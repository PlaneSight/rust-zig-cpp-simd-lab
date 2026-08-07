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
        const difference: Bytes = @max(va, vb) - @min(va, vb);
        const wide: Wide = @intCast(difference);
        sum += @as(u64, @intCast(@reduce(.Add, wide)));
    }

    while (i < a.len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        sum += if (x > y) x - y else y - x;
    }
    return sum;
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
        try std.testing.expect(relative_error <= 1e-12);
        try std.testing.expectEqual(
            sadU8Scalar(bytes_a[0..len], bytes_b[0..len]),
            sadU8Vector(bytes_a[0..len], bytes_b[0..len]),
        );
    }
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
