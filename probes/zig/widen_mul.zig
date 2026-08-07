// Scalar and explicit-vector widening-multiply probes.
// No allocation or I/O: compile this file directly when inspecting codegen.

pub export fn widen_mul_u8_u16_scalar(
    dst: [*]u16,
    a: [*]const u8,
    b: [*]const u8,
    len: usize,
) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: u16 = @intCast(a[i]);
        const wide_b: u16 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_u8_u16_vector(
    dst: [*]u16,
    a: [*]const u8,
    b: [*]const u8,
    len: usize,
) void {
    const Input = @Vector(16, u8);
    const Wide = @Vector(16, u16);
    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        const va: Input = a[i..][0..16].*;
        const vb: Input = b[i..][0..16].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..16].* = wide_a * wide_b;
    }
    while (i < len) : (i += 1) {
        const wide_a: u16 = @intCast(a[i]);
        const wide_b: u16 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_i8_i16_scalar(
    dst: [*]i16,
    a: [*]const i8,
    b: [*]const i8,
    len: usize,
) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: i16 = @intCast(a[i]);
        const wide_b: i16 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_i8_i16_vector(
    dst: [*]i16,
    a: [*]const i8,
    b: [*]const i8,
    len: usize,
) void {
    const Input = @Vector(16, i8);
    const Wide = @Vector(16, i16);
    var i: usize = 0;
    while (i + 16 <= len) : (i += 16) {
        const va: Input = a[i..][0..16].*;
        const vb: Input = b[i..][0..16].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..16].* = wide_a * wide_b;
    }
    while (i < len) : (i += 1) {
        const wide_a: i16 = @intCast(a[i]);
        const wide_b: i16 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_u16_u32_scalar(
    dst: [*]u32,
    a: [*]const u16,
    b: [*]const u16,
    len: usize,
) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: u32 = @intCast(a[i]);
        const wide_b: u32 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_u16_u32_vector(
    dst: [*]u32,
    a: [*]const u16,
    b: [*]const u16,
    len: usize,
) void {
    const Input = @Vector(8, u16);
    const Wide = @Vector(8, u32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..8].* = wide_a * wide_b;
    }
    while (i < len) : (i += 1) {
        const wide_a: u32 = @intCast(a[i]);
        const wide_b: u32 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_i16_i32_scalar(
    dst: [*]i32,
    a: [*]const i16,
    b: [*]const i16,
    len: usize,
) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: i32 = @intCast(a[i]);
        const wide_b: i32 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}

pub export fn widen_mul_i16_i32_vector(
    dst: [*]i32,
    a: [*]const i16,
    b: [*]const i16,
    len: usize,
) void {
    const Input = @Vector(8, i16);
    const Wide = @Vector(8, i32);
    var i: usize = 0;
    while (i + 8 <= len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        dst[i..][0..8].* = wide_a * wide_b;
    }
    while (i < len) : (i += 1) {
        const wide_a: i32 = @intCast(a[i]);
        const wide_b: i32 = @intCast(b[i]);
        dst[i] = wide_a * wide_b;
    }
}
