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
