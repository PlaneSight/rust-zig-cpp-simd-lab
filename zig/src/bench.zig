const std = @import("std");
const kernels = @import("kernels.zig");

const n: usize = 1 << 20;
const warmup: usize = 8;
const iterations: usize = 64;

fn report(name: []const u8, elapsed_ns: i128, bytes_per_iter: usize) void {
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
    for (0..warmup) |_| sink +%= kernels.squaredErrorScalar(x, y);
    timer.reset();
    for (0..iterations) |_| sink +%= kernels.squaredErrorScalar(x, y);
    report("sqerr/scalar-autovec", timer.read(), n * 8);

    for (0..warmup) |_| sink +%= kernels.squaredErrorVector(x, y);
    timer.reset();
    for (0..iterations) |_| sink +%= kernels.squaredErrorVector(x, y);
    report("sqerr/native-vector", timer.read(), n * 8);

    std.mem.doNotOptimizeAway(&sink);
    std.debug.print("N={d} warmup={d} iterations={d}\n", .{ n, warmup, iterations });
}
