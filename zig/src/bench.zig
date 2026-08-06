const std = @import("std");
const kernels = @import("kernels.zig");

const n: usize = 1 << 20;
const warmup: usize = 8;
const iterations: usize = 64;
const half_values = [_]u16{ 0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400 };

fn report(name: []const u8, elapsed_ns: u64, bytes_per_iter: usize) void {
    const elapsed: f64 = @floatFromInt(elapsed_ns);
    const ns_elem = elapsed / @as(f64, @floatFromInt(iterations * n));
    const seconds = elapsed / 1e9;
    const gib_s = @as(f64, @floatFromInt(bytes_per_iter * iterations)) / seconds /
        @as(f64, @floatFromInt(@as(u64, 1) << 30));
    std.debug.print("{s: <28} {d:9.4} ns/elem  {d:8.2} GiB/s\n", .{ name, ns_elem, gib_s });
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const x = try allocator.alloc(f32, n);
    defer allocator.free(x);
    const y = try allocator.alloc(f32, n);
    defer allocator.free(y);
    const dst = try allocator.alloc(f32, n);
    defer allocator.free(dst);

    for (x, y, 0..) |*xv, *yv, i| {
        xv.* = @as(f32, @floatFromInt(i)) * 0.001;
        yv.* = 1.0 + @as(f32, @floatFromInt(i)) * 0.0005;
    }

    for (0..warmup) |_| kernels.axpyScalar(dst, x, y, 0.75);
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| kernels.axpyScalar(dst, x, y, 0.75);
    report("axpy/scalar-autovec", timer.read(), n * 12);

    var sink: f32 = 0;
    for (0..warmup) |_| sink += kernels.squaredErrorScalar(x, y);
    timer.reset();
    for (0..iterations) |_| sink += kernels.squaredErrorScalar(x, y);
    report("sqerr/scalar-autovec", timer.read(), n * 8);

    for (0..warmup) |_| sink += kernels.squaredErrorVector(x, y);
    timer.reset();
    for (0..iterations) |_| sink += kernels.squaredErrorVector(x, y);
    report("sqerr/native-vector", timer.read(), n * 8);

    const bytes_a = try allocator.alloc(u8, n);
    defer allocator.free(bytes_a);
    const bytes_b = try allocator.alloc(u8, n);
    defer allocator.free(bytes_b);
    for (bytes_a, bytes_b, 0..) |*av, *bv, i| {
        av.* = @intCast((i * 17 + 3) & 255);
        bv.* = @intCast((i * 29 + 11) & 255);
    }
    std.debug.assert(kernels.sadU8Scalar(bytes_a, bytes_b) == kernels.sadU8Vector(bytes_a, bytes_b));

    var sink_u64: u64 = 0;
    for (0..warmup) |_| sink_u64 +%= kernels.sadU8Scalar(bytes_a, bytes_b);
    timer.reset();
    for (0..iterations) |_| sink_u64 +%= kernels.sadU8Scalar(bytes_a, bytes_b);
    report("sad-u8/scalar-autovec", timer.read(), n * 2);

    for (0..warmup) |_| sink_u64 +%= kernels.sadU8Vector(bytes_a, bytes_b);
    timer.reset();
    for (0..iterations) |_| sink_u64 +%= kernels.sadU8Vector(bytes_a, bytes_b);
    report("sad-u8/native-vector", timer.read(), n * 2);

    const c = try allocator.alloc(f16, n);
    defer allocator.free(c);
    const lo = try allocator.alloc(f16, n);
    defer allocator.free(lo);
    const hi = try allocator.alloc(f16, n);
    defer allocator.free(hi);
    const native = try allocator.alloc(f16, n);
    defer allocator.free(native);
    const promoted = try allocator.alloc(f16, n);
    defer allocator.free(promoted);

    for (c, lo, hi, 0..) |*cv, *lv, *hv, i| {
        cv.* = @bitCast(half_values[i & 7]);
        lv.* = @bitCast(@as(u16, 0x3800)); // 0.5
        hv.* = @bitCast(@as(u16, 0x4000)); // 2.0
    }

    kernels.clampF16Native(native, c, lo, hi);
    kernels.clampF16PromoteOnce(promoted, c, lo, hi);
    for (native, promoted) |a, b| {
        std.debug.assert(@as(u16, @bitCast(a)) == @as(u16, @bitCast(b)));
    }

    for (0..warmup) |_| kernels.clampF16Native(native, c, lo, hi);
    timer.reset();
    for (0..iterations) |_| kernels.clampF16Native(native, c, lo, hi);
    report("clamp-f16/native", timer.read(), n * 8);

    for (0..warmup) |_| kernels.clampF16PromoteOnce(promoted, c, lo, hi);
    timer.reset();
    for (0..iterations) |_| kernels.clampF16PromoteOnce(promoted, c, lo, hi);
    report("clamp-f16/promote-f32", timer.read(), n * 8);

    std.mem.doNotOptimizeAway(&sink);
    std.mem.doNotOptimizeAway(&sink_u64);
    std.mem.doNotOptimizeAway(native.ptr);
    std.mem.doNotOptimizeAway(promoted.ptr);
    std.debug.print("N={d} warmup={d} iterations={d}\n", .{ n, warmup, iterations });
}
