const std = @import("std");
const kernels = @import("kernels.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const n: usize = 1 << 20;

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

    kernels.axpyScalar(dst, x, y, 0.75);
    const scalar = kernels.squaredErrorScalar(x, y);
    const vector = kernels.squaredErrorVector(x, y);

    var checksum: f64 = 0;
    for (dst) |v| checksum += v;

    std.debug.print("Zig SIMD lab smoke test\n", .{});
    std.debug.print("AXPY checksum: {d:.6}\n", .{checksum});
    std.debug.print("Squared error scalar: {d:.6}\n", .{scalar});
    std.debug.print("Squared error vector: {d:.6}\n", .{vector});
}
