const std = @import("std");

fn convolve3_at(src: [*]const u8, len: usize, i: usize) u8 {
    const left = if (i == 0) 0 else i - 1;
    const right = if (i < len - 1) i + 1 else i;
    const sum: u32 = @as(u32, src[left]) + 2 * @as(u32, src[i]) + @as(u32, src[right]) + 2;
    return @intCast(sum >> 2);
}

fn convolve5_at(src: [*]const u8, len: usize, i: usize) u8 {
    const left2 = if (i < 2) 0 else i - 2;
    const left1 = if (i == 0) 0 else i - 1;
    const right1 = if (i < len - 1) i + 1 else i;
    const right2 = if (len >= 3 and i < len - 2) i + 2 else len - 1;
    const sum: u32 = @as(u32, src[left2]) + 4 * @as(u32, src[left1]) + 6 * @as(u32, src[i]) + 4 * @as(u32, src[right1]) + @as(u32, src[right2]) + 8;
    return @intCast(sum >> 4);
}

pub export fn blend_u8_scalar(
    dst: [*]u8,
    a: [*]const u8,
    b: [*]const u8,
    weight: u16,
    len: usize,
) void {
    std.debug.assert(weight <= 256);
    if (len == 0) return;
    const wide_weight: u32 = @intCast(weight);
    const inverse_weight: u32 = 256 - wide_weight;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const weighted_a = @as(u32, a[i]) * inverse_weight;
        const weighted_b = @as(u32, b[i]) * wide_weight;
        dst[i] = @intCast((weighted_a + weighted_b + 128) >> 8);
    }
}

pub export fn blend_u8_vector(
    dst: [*]u8,
    a: [*]const u8,
    b: [*]const u8,
    weight: u16,
    len: usize,
) void {
    std.debug.assert(weight <= 256);
    if (len == 0) return;
    const Lanes = 16;
    const Input = @Vector(Lanes, u8);
    const Wide = @Vector(Lanes, u32);
    const wide_weight: u32 = @intCast(weight);
    const inverse_weight: u32 = 256 - wide_weight;
    const weight_vec: Wide = @splat(wide_weight);
    const inverse_vec: Wide = @splat(inverse_weight);
    const rounding: Wide = @splat(128);
    var i: usize = 0;
    while (i + Lanes <= len) : (i += Lanes) {
        const va: Input = a[i..][0..Lanes].*;
        const vb: Input = b[i..][0..Lanes].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        const sum = wide_a * inverse_vec + wide_b * weight_vec + rounding;
        const result: Input = @intCast(sum >> @splat(8));
        dst[i..][0..Lanes].* = result;
    }
    while (i < len) : (i += 1) {
        const weighted_a = @as(u32, a[i]) * inverse_weight;
        const weighted_b = @as(u32, b[i]) * wide_weight;
        dst[i] = @intCast((weighted_a + weighted_b + 128) >> 8);
    }
}

pub export fn convolve3_u8_scalar(
    dst: [*]u8,
    src: [*]const u8,
    len: usize,
) void {
    if (len == 0) return;
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = convolve3_at(src, len, i);
}

pub export fn convolve3_u8_vector(
    dst: [*]u8,
    src: [*]const u8,
    len: usize,
) void {
    if (len == 0) return;
    if (len == 1) {
        dst[0] = convolve3_at(src, len, 0);
        return;
    }

    dst[0] = convolve3_at(src, len, 0);
    const Lanes = 16;
    const Input = @Vector(Lanes, u8);
    const Wide = @Vector(Lanes, u16);
    const two: Wide = @splat(2);
    const rounding: Wide = @splat(2);
    var i: usize = 1;
    while (i + Lanes <= len - 1) : (i += Lanes) {
        const left_bytes: Input = src[i - 1 ..][0..Lanes].*;
        const center_bytes: Input = src[i..][0..Lanes].*;
        const right_bytes: Input = src[i + 1 ..][0..Lanes].*;
        const left: Wide = @intCast(left_bytes);
        const center: Wide = @intCast(center_bytes);
        const right: Wide = @intCast(right_bytes);
        const sum = left + center * two + right + rounding;
        const result: Input = @intCast(sum >> @splat(2));
        dst[i..][0..Lanes].* = result;
    }
    while (i < len - 1) : (i += 1) dst[i] = convolve3_at(src, len, i);
    dst[len - 1] = convolve3_at(src, len, len - 1);
}

pub export fn convolve5_u8_scalar(
    dst: [*]u8,
    src: [*]const u8,
    len: usize,
) void {
    if (len == 0) return;
    var i: usize = 0;
    while (i < len) : (i += 1) dst[i] = convolve5_at(src, len, i);
}

pub export fn convolve5_u8_vector(
    dst: [*]u8,
    src: [*]const u8,
    len: usize,
) void {
    if (len == 0) return;
    if (len < 5) {
        var short_i: usize = 0;
        while (short_i < len) : (short_i += 1) dst[short_i] = convolve5_at(src, len, short_i);
        return;
    }

    dst[0] = convolve5_at(src, len, 0);
    dst[1] = convolve5_at(src, len, 1);
    const Lanes = 16;
    const Input = @Vector(Lanes, u8);
    const Wide = @Vector(Lanes, u16);
    const four: Wide = @splat(4);
    const six: Wide = @splat(6);
    const rounding: Wide = @splat(8);
    var i: usize = 2;
    while (i + Lanes <= len - 2) : (i += Lanes) {
        const left2_bytes: Input = src[i - 2 ..][0..Lanes].*;
        const left1_bytes: Input = src[i - 1 ..][0..Lanes].*;
        const center_bytes: Input = src[i..][0..Lanes].*;
        const right1_bytes: Input = src[i + 1 ..][0..Lanes].*;
        const right2_bytes: Input = src[i + 2 ..][0..Lanes].*;
        const left2: Wide = @intCast(left2_bytes);
        const left1: Wide = @intCast(left1_bytes);
        const center: Wide = @intCast(center_bytes);
        const right1: Wide = @intCast(right1_bytes);
        const right2: Wide = @intCast(right2_bytes);
        const sum = left2 + left1 * four + center * six + right1 * four + right2 + rounding;
        const result: Input = @intCast(sum >> @splat(4));
        dst[i..][0..Lanes].* = result;
    }
    while (i < len - 2) : (i += 1) dst[i] = convolve5_at(src, len, i);
    dst[len - 2] = convolve5_at(src, len, len - 2);
    dst[len - 1] = convolve5_at(src, len, len - 1);
}
