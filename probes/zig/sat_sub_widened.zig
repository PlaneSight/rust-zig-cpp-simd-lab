const std = @import("std");

pub export fn sat_sub_u8_widened(dst: [*]u8, a: [*]const u8, b: [*]const u8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i16 = @as(i16, a[i]) - @as(i16, b[i]);
        dst[i] = if (difference < 0) 0 else @intCast(difference);
    }
}

pub export fn sat_sub_i8_widened(dst: [*]i8, a: [*]const i8, b: [*]const i8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i16 = @as(i16, a[i]) - @as(i16, b[i]);
        const min_value: i16 = @intCast(std.math.minInt(i8));
        const max_value: i16 = @intCast(std.math.maxInt(i8));
        dst[i] = if (difference < min_value) std.math.minInt(i8) else if (difference > max_value) std.math.maxInt(i8) else @intCast(difference);
    }
}

pub export fn sat_sub_u16_widened(dst: [*]u16, a: [*]const u16, b: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i32 = @as(i32, a[i]) - @as(i32, b[i]);
        dst[i] = if (difference < 0) 0 else @intCast(difference);
    }
}

pub export fn sat_sub_i16_widened(dst: [*]i16, a: [*]const i16, b: [*]const i16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i32 = @as(i32, a[i]) - @as(i32, b[i]);
        const min_value: i32 = @intCast(std.math.minInt(i16));
        const max_value: i32 = @intCast(std.math.maxInt(i16));
        dst[i] = if (difference < min_value) std.math.minInt(i16) else if (difference > max_value) std.math.maxInt(i16) else @intCast(difference);
    }
}

pub export fn sat_sub_u32_widened(dst: [*]u32, a: [*]const u32, b: [*]const u32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i64 = @as(i64, a[i]) - @as(i64, b[i]);
        dst[i] = if (difference < 0) 0 else @intCast(difference);
    }
}

pub export fn sat_sub_i32_widened(dst: [*]i32, a: [*]const i32, b: [*]const i32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i64 = @as(i64, a[i]) - @as(i64, b[i]);
        const min_value: i64 = @intCast(std.math.minInt(i32));
        const max_value: i64 = @intCast(std.math.maxInt(i32));
        dst[i] = if (difference < min_value) std.math.minInt(i32) else if (difference > max_value) std.math.maxInt(i32) else @intCast(difference);
    }
}

pub export fn sat_sub_u64_widened(dst: [*]u64, a: [*]const u64, b: [*]const u64, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i128 = @as(i128, a[i]) - @as(i128, b[i]);
        dst[i] = if (difference < 0) 0 else @intCast(difference);
    }
}

pub export fn sat_sub_i64_widened(dst: [*]i64, a: [*]const i64, b: [*]const i64, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const difference: i128 = @as(i128, a[i]) - @as(i128, b[i]);
        const min_value: i128 = @intCast(std.math.minInt(i64));
        const max_value: i128 = @intCast(std.math.maxInt(i64));
        dst[i] = if (difference < min_value) std.math.minInt(i64) else if (difference > max_value) std.math.maxInt(i64) else @intCast(difference);
    }
}
