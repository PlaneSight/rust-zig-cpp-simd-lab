const std = @import("std");
const kernels = @import("kernels.zig");

const cases = [_]u16{
    0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x37ff, 0x3800, 0x3801,
    0x3bff, 0x3c00, 0x3c01, 0x4000, 0x7bff, 0x7c00, 0x7e01, 0x7fff,
    0x8001, 0xb800, 0xbc00, 0xc000, 0xfbff, 0xfc00, 0xfe01,
};

fn dump(name: []const u8, values: []const f16) void {
    std.debug.print("{{\"strategy\":\"{s}\",\"available\":true,\"outputs\":[", .{name});
    for (values, 0..) |v, i| {
        if (i != 0) std.debug.print(",", .{});
        const bits: u16 = @bitCast(v);
        std.debug.print("\"{x:0>4}\"", .{bits});
    }
    std.debug.print("]}}\n", .{});
}

pub fn main() void {
    var c: [cases.len]f16 = undefined;
    var lo: [cases.len]f16 = undefined;
    var hi: [cases.len]f16 = undefined;
    var native: [cases.len]f16 = undefined;
    var promoted: [cases.len]f16 = undefined;

    for (&c, &lo, &hi, 0..) |*cv, *lv, *hv, i| {
        cv.* = @bitCast(cases[i]);
        lv.* = @bitCast(@as(u16, 0xb800)); // -0.5
        hv.* = @bitCast(@as(u16, 0x4000)); // +2.0
    }

    kernels.clampF16Native(&native, &c, &lo, &hi);
    kernels.clampF16PromoteOnce(&promoted, &c, &lo, &hi);
    dump("zig-native-f16", &native);
    dump("zig-promote-f32", &promoted);
}
