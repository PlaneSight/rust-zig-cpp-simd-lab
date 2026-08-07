const std = @import("std");

// Raw-pointer probes mirror the slice kernels. Vector loops always finish with
// the same scalar conversion used by their scalar counterpart.
pub export fn widen_u8_to_u16_scalar(dst: [*]u16, src: [*]const u8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_u8_to_u16_vector(dst: [*]u16, src: [*]const u8, len: usize) void {
    const Input = @Vector(16, u8);
    const Output = @Vector(16, u16);
    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        const values: Input = src[i..][0..16].*;
        const widened: Output = @intCast(values);
        dst[i..][0..16].* = widened;
    }
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_u8_to_u32_scalar(dst: [*]u32, src: [*]const u8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_u8_to_u32_vector(dst: [*]u32, src: [*]const u8, len: usize) void {
    const Input = @Vector(8, u8);
    const Output = @Vector(8, u32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const widened: Output = @intCast(values);
        dst[i..][0..8].* = widened;
    }
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_i8_to_i16_scalar(dst: [*]i16, src: [*]const i8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_i8_to_i16_vector(dst: [*]i16, src: [*]const i8, len: usize) void {
    const Input = @Vector(16, i8);
    const Output = @Vector(16, i16);
    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        const values: Input = src[i..][0..16].*;
        const widened: Output = @intCast(values);
        dst[i..][0..16].* = widened;
    }
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_u16_to_u32_scalar(dst: [*]u32, src: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_u16_to_u32_vector(dst: [*]u32, src: [*]const u16, len: usize) void {
    const Input = @Vector(8, u16);
    const Output = @Vector(8, u32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const widened: Output = @intCast(values);
        dst[i..][0..8].* = widened;
    }
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_i16_to_i32_scalar(dst: [*]i32, src: [*]const i16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn widen_i16_to_i32_vector(dst: [*]i32, src: [*]const i16, len: usize) void {
    const Input = @Vector(8, i16);
    const Output = @Vector(8, i32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const widened: Output = @intCast(values);
        dst[i..][0..8].* = widened;
    }
    while (i < len) : (i += 1) dst[i] = @intCast(src[i]);
}

pub export fn convert_u16_to_f32_scalar(dst: [*]f32, src: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @floatFromInt(src[i]);
}

pub export fn convert_u16_to_f32_vector(dst: [*]f32, src: [*]const u16, len: usize) void {
    const Input = @Vector(8, u16);
    const Output = @Vector(8, f32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const converted: Output = @floatFromInt(values);
        dst[i..][0..8].* = converted;
    }
    while (i < len) : (i += 1) dst[i] = @floatFromInt(src[i]);
}

pub export fn convert_i16_to_f32_scalar(dst: [*]f32, src: [*]const i16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @floatFromInt(src[i]);
}

pub export fn convert_i16_to_f32_vector(dst: [*]f32, src: [*]const i16, len: usize) void {
    const Input = @Vector(8, i16);
    const Output = @Vector(8, f32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const values: Input = src[i..][0..8].*;
        const converted: Output = @floatFromInt(values);
        dst[i..][0..8].* = converted;
    }
    while (i < len) : (i += 1) dst[i] = @floatFromInt(src[i]);
}

pub export fn convert_u8_f32_affine_scalar(
    dst: [*]f32,
    src: [*]const u8,
    scale: f32,
    bias: f32,
    len: usize,
) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = @as(f32, @floatFromInt(src[i])) * scale + bias;
    }
}

pub export fn convert_f32_u8_trunc_scalar(dst: [*]u8, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        std.debug.assert(!std.math.isNan(src[i]) and src[i] >= 0.0 and src[i] <= 255.0);
        dst[i] = @intFromFloat(src[i]);
    }
}

pub export fn convert_f32_u8_round_scalar(dst: [*]u8, src: [*]const f32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        std.debug.assert(!std.math.isNan(src[i]) and src[i] >= 0.0 and src[i] <= 255.0);
        dst[i] = @intFromFloat(@floor(src[i] + 0.5));
    }
}

pub export fn convert_f32_u8_sat_scalar(dst: [*]u8, src: [*]const f32, len: usize) void {
    const max_value: f32 = @floatFromInt(std.math.maxInt(u8));
    var i: usize = 0;
    while (i < len) : (i += 1) {
        if (std.math.isNan(src[i]) or src[i] <= 0.0) {
            dst[i] = 0;
        } else if (src[i] >= max_value) {
            dst[i] = std.math.maxInt(u8);
        } else {
            dst[i] = @intFromFloat(src[i]);
        }
    }
}

pub export fn f32_to_u16_sat_scalar(dst: [*]u16, src: [*]const f32, len: usize) void {
    const max_value: f32 = @floatFromInt(std.math.maxInt(u16));
    var i: usize = 0;
    while (i < len) : (i += 1) {
        if (std.math.isNan(src[i]) or src[i] <= 0.0) {
            dst[i] = 0;
        } else if (src[i] >= max_value) {
            dst[i] = std.math.maxInt(u16);
        } else {
            dst[i] = @intFromFloat(src[i]);
        }
    }
}

pub export fn narrow_u16_to_u8_trunc_scalar(dst: [*]u8, src: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = @intCast(src[i] & 0xff);
}

pub export fn narrow_u16_to_u8_round_scalar(dst: [*]u8, src: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = @intCast((@as(u32, src[i]) + 128) / 257);
    }
}

pub export fn narrow_u16_to_u8_sat_scalar(dst: [*]u8, src: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = if (src[i] > std.math.maxInt(u8)) std.math.maxInt(u8) else @intCast(src[i]);
    }
}

// For packing and unpacking, len is the number of four-byte groups. The
// pointed-to byte ranges therefore have length len * 4.
pub export fn pack_u8x4_to_u32_scalar(dst: [*]u32, src: [*]const u8, len: usize) void {
    var group: usize = 0;
    while (group < len) : (group += 1) {
        const offset = group * 4;
        const b0: u32 = @intCast(src[offset]);
        const b1: u32 = @intCast(src[offset + 1]);
        const b2: u32 = @intCast(src[offset + 2]);
        const b3: u32 = @intCast(src[offset + 3]);
        dst[group] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

pub export fn unpack_u32_to_u8x4_scalar(dst: [*]u8, src: [*]const u32, len: usize) void {
    var group: usize = 0;
    while (group < len) : (group += 1) {
        const value = src[group];
        const offset = group * 4;
        dst[offset] = @truncate(value);
        dst[offset + 1] = @truncate(value >> 8);
        dst[offset + 2] = @truncate(value >> 16);
        dst[offset + 3] = @truncate(value >> 24);
    }
}
