// Scalar and explicit-vector dot-product probes.
// These functions intentionally use only raw pointers and fixed vectors so they
// can be compiled directly for code-generation inspection.

pub export fn dot_f32_scalar(a: [*]const f32, b: [*]const f32, len: usize) f64 {
    var sum: f64 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: f64 = @floatCast(a[i]);
        const wide_b: f64 = @floatCast(b[i]);
        sum += wide_a * wide_b;
    }
    return sum;
}

pub export fn dot_f32_vector(a: [*]const f32, b: [*]const f32, len: usize) f64 {
    const Input = @Vector(8, f32);
    const Wide = @Vector(8, f64);
    var acc: Wide = @splat(0.0);
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @floatCast(va);
        const wide_b: Wide = @floatCast(vb);
        acc += wide_a * wide_b;
    }

    var sum: f64 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        const wide_a: f64 = @floatCast(a[i]);
        const wide_b: f64 = @floatCast(b[i]);
        sum += wide_a * wide_b;
    }
    return sum;
}

pub export fn dot_f64_scalar(a: [*]const f64, b: [*]const f64, len: usize) f64 {
    var sum: f64 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

pub export fn dot_f64_vector(a: [*]const f64, b: [*]const f64, len: usize) f64 {
    const Vec = @Vector(4, f64);
    var acc: Vec = @splat(0.0);
    var i: usize = 0;

    while (i + 4 <= len) : (i += 4) {
        const va: Vec = a[i..][0..4].*;
        const vb: Vec = b[i..][0..4].*;
        acc += va * vb;
    }

    var sum: f64 = @reduce(.Add, acc);
    while (i < len) : (i += 1) {
        sum += a[i] * b[i];
    }
    return sum;
}

pub export fn dot_i16_scalar(a: [*]const i16, b: [*]const i16, len: usize) i64 {
    var sum: i64 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: i64 = @intCast(a[i]);
        const wide_b: i64 = @intCast(b[i]);
        sum += wide_a * wide_b;
    }
    return sum;
}

pub export fn dot_i16_vector(a: [*]const i16, b: [*]const i16, len: usize) i64 {
    const Input = @Vector(8, i16);
    const Wide = @Vector(8, i64);
    var sum: i64 = 0;
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: Input = a[i..][0..8].*;
        const vb: Input = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        sum += @reduce(.Add, wide_a * wide_b);
    }

    while (i < len) : (i += 1) {
        const wide_a: i64 = @intCast(a[i]);
        const wide_b: i64 = @intCast(b[i]);
        sum += wide_a * wide_b;
    }
    return sum;
}

pub export fn dot_u8_i8_scalar(a: [*]const u8, b: [*]const i8, len: usize) i64 {
    var sum: i64 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const wide_a: i64 = @intCast(a[i]);
        const wide_b: i64 = @intCast(b[i]);
        sum += wide_a * wide_b;
    }
    return sum;
}

pub export fn dot_u8_i8_vector(a: [*]const u8, b: [*]const i8, len: usize) i64 {
    const InputU8 = @Vector(8, u8);
    const InputI8 = @Vector(8, i8);
    const Wide = @Vector(8, i64);
    var sum: i64 = 0;
    var i: usize = 0;

    while (i + 8 <= len) : (i += 8) {
        const va: InputU8 = a[i..][0..8].*;
        const vb: InputI8 = b[i..][0..8].*;
        const wide_a: Wide = @intCast(va);
        const wide_b: Wide = @intCast(vb);
        sum += @reduce(.Add, wide_a * wide_b);
    }

    while (i < len) : (i += 1) {
        const wide_a: i64 = @intCast(a[i]);
        const wide_b: i64 = @intCast(b[i]);
        sum += wide_a * wide_b;
    }
    return sum;
}
