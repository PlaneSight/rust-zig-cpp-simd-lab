const std = @import("std");

pub fn axpyScalar(dst: []f32, x: []const f32, y: []const f32, a: f32) void {
    std.debug.assert(dst.len == x.len and x.len == y.len);
    for (dst, x, y) |*d, xv, yv| {
        d.* = @mulAdd(f32, a, xv, yv);
    }
}

pub fn squaredErrorScalar(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (a, b) |x, y| {
        const d = x - y;
        sum += d * d;
    }
    return sum;
}

pub fn squaredErrorVector(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    const Vec = @Vector(8, f32);
    var acc: Vec = @splat(0.0);
    var i: usize = 0;

    while (i + 8 <= a.len) : (i += 8) {
        const va: Vec = a[i..][0..8].*;
        const vb: Vec = b[i..][0..8].*;
        const d = va - vb;
        acc += d * d;
    }

    var sum = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        const d = a[i] - b[i];
        sum += d * d;
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
    const H = @Vector(16, f16);
    const F = @Vector(16, f32);
    var i: usize = 0;
    while (i + 16 <= dst.len) : (i += 16) {
        const hc: H = c[i..][0..16].*;
        const hlo: H = lo[i..][0..16].*;
        const hhi: H = hi[i..][0..16].*;
        const vc: F = @floatCast(hc);
        const vlo: F = @floatCast(hlo);
        const vhi: F = @floatCast(hhi);
        const out: F = @max(vlo, @min(vc, vhi));
        const narrowed: H = @floatCast(out);
        dst[i..][0..16].* = narrowed;
    }
    while (i < dst.len) : (i += 1) {
        const wc: f32 = @floatCast(c[i]);
        const wlo: f32 = @floatCast(lo[i]);
        const whi: f32 = @floatCast(hi[i]);
        dst[i] = @floatCast(@max(wlo, @min(wc, whi)));
    }
}

test "axpy" {
    const x = [_]f32{ 1, 2, 3, 4 };
    const y = [_]f32{ 5, 6, 7, 8 };
    var dst = [_]f32{0} ** 4;
    axpyScalar(&dst, &x, &y, 2.0);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 7, 10, 13, 16 }, &dst);
}

test "vector squared error matches scalar" {
    var a: [257]f32 = undefined;
    var b: [257]f32 = undefined;
    for (&a, &b, 0..) |*av, *bv, i| {
        av.* = @as(f32, @floatFromInt(i)) * 0.25;
        bv.* = @as(f32, @floatFromInt(i)) * 0.125 + 1.0;
    }
    const scalar = squaredErrorScalar(&a, &b);
    const vector = squaredErrorVector(&a, &b);
    const tolerance = @max(@abs(scalar), 1.0) * 1e-5;
    try std.testing.expect(@abs(scalar - vector) <= tolerance);
}

test "promoted f16 clamp matches native for finite exact inputs" {
    var c: [32]f16 = undefined;
    var lo: [32]f16 = undefined;
    var hi: [32]f16 = undefined;
    var native: [32]f16 = undefined;
    var promoted: [32]f16 = undefined;
    for (&c, &lo, &hi, 0..) |*cv, *lv, *hv, i| {
        const v: f32 = @floatFromInt(i % 8);
        cv.* = @floatCast(v * 0.5);
        lv.* = 0.5;
        hv.* = 2.5;
    }
    clampF16Native(&native, &c, &lo, &hi);
    clampF16PromoteOnce(&promoted, &c, &lo, &hi);
    try std.testing.expectEqualSlices(f16, &native, &promoted);
}
