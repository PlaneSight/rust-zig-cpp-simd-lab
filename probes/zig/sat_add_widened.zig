const std = @import("std");

pub export fn sat_add_u8_widened(dst: [*]u8, a: [*]const u8, b: [*]const u8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: u16 = @as(u16, a[i]) + @as(u16, b[i]);
        dst[i] = if (sum > @as(u16, std.math.maxInt(u8))) std.math.maxInt(u8) else @intCast(sum);
    }
}

pub export fn sat_add_i8_widened(dst: [*]i8, a: [*]const i8, b: [*]const i8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: i16 = @as(i16, a[i]) + @as(i16, b[i]);
        const min_value: i16 = @intCast(std.math.minInt(i8));
        const max_value: i16 = @intCast(std.math.maxInt(i8));
        dst[i] = if (sum < min_value) std.math.minInt(i8) else if (sum > max_value) std.math.maxInt(i8) else @intCast(sum);
    }
}

pub export fn sat_add_u16_widened(dst: [*]u16, a: [*]const u16, b: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: u32 = @as(u32, a[i]) + @as(u32, b[i]);
        dst[i] = if (sum > @as(u32, std.math.maxInt(u16))) std.math.maxInt(u16) else @intCast(sum);
    }
}

pub export fn sat_add_i16_widened(dst: [*]i16, a: [*]const i16, b: [*]const i16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: i32 = @as(i32, a[i]) + @as(i32, b[i]);
        const min_value: i32 = @intCast(std.math.minInt(i16));
        const max_value: i32 = @intCast(std.math.maxInt(i16));
        dst[i] = if (sum < min_value) std.math.minInt(i16) else if (sum > max_value) std.math.maxInt(i16) else @intCast(sum);
    }
}

pub export fn sat_add_u32_widened(dst: [*]u32, a: [*]const u32, b: [*]const u32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: u64 = @as(u64, a[i]) + @as(u64, b[i]);
        dst[i] = if (sum > @as(u64, std.math.maxInt(u32))) std.math.maxInt(u32) else @intCast(sum);
    }
}

pub export fn sat_add_i32_widened(dst: [*]i32, a: [*]const i32, b: [*]const i32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: i64 = @as(i64, a[i]) + @as(i64, b[i]);
        const min_value: i64 = @intCast(std.math.minInt(i32));
        const max_value: i64 = @intCast(std.math.maxInt(i32));
        dst[i] = if (sum < min_value) std.math.minInt(i32) else if (sum > max_value) std.math.maxInt(i32) else @intCast(sum);
    }
}

pub export fn sat_add_u64_widened(dst: [*]u64, a: [*]const u64, b: [*]const u64, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: u128 = @as(u128, a[i]) + @as(u128, b[i]);
        dst[i] = if (sum > @as(u128, std.math.maxInt(u64))) std.math.maxInt(u64) else @intCast(sum);
    }
}

pub export fn sat_add_i64_widened(dst: [*]i64, a: [*]const i64, b: [*]const i64, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const sum: i128 = @as(i128, a[i]) + @as(i128, b[i]);
        const min_value: i128 = @intCast(std.math.minInt(i64));
        const max_value: i128 = @intCast(std.math.maxInt(i64));
        dst[i] = if (sum < min_value) std.math.minInt(i64) else if (sum > max_value) std.math.maxInt(i64) else @intCast(sum);
    }
}
